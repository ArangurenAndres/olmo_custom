import os
import torch
import time
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig, InitMethod
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from olmo_core.utils import seed_all
from olmo_core.data import NumpyDatasetConfig, NumpyDataLoaderConfig, TokenizerConfig, NumpyDatasetType
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer import TransformerTrainModule
from olmo_core.train.train_module.transformer.config import TransformerActivationCheckpointingConfig, TransformerActivationCheckpointingMode
from olmo_core.train.callbacks import Callback

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--inference-interval", type=int, default=5)
parser.add_argument("--prompt", type=str, default="Dutch is ")
args = parser.parse_args()

# Constants
SEQUENCE_LENGTH = 1024
GLOBAL_BATCH_SIZE = args.batch_size * SEQUENCE_LENGTH
TOTAL_TOKENS = args.steps * GLOBAL_BATCH_SIZE
TOTAL_TOKENS_MARGIN = int(TOTAL_TOKENS * 1.1)
SEQUENCES_NEEDED = (TOTAL_TOKENS_MARGIN + SEQUENCE_LENGTH - 1) // SEQUENCE_LENGTH

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)

# Data paths
data_dir = "./data_local_test"
os.makedirs(data_dir, exist_ok=True)
wiki_tokens_path = os.path.join(data_dir, "wiki_tokens.npy")

# Tokenizer
tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")

# Tokenization
if not os.path.exists(wiki_tokens_path):
    print("Downloading and tokenizing a small Wikipedia subset...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train")
    all_tokens = []
    for article in tqdm(dataset.select(range(1000)), desc="Tokenizing"):
        tokens = tokenizer.encode(article["text"])
        tokens = [t for t in tokens if t != 0]
        all_tokens.extend(tokens)
        if len(all_tokens) >= TOTAL_TOKENS_MARGIN:
            break
    total_sequences = len(all_tokens) // SEQUENCE_LENGTH
    tokens = all_tokens[:total_sequences * SEQUENCE_LENGTH]
    np.save(wiki_tokens_path, np.array(tokens, dtype=np.int32).reshape(-1, SEQUENCE_LENGTH))
else:
    print("Using existing tokenized data.")

# Load tokens
wiki_tokens = np.load(wiki_tokens_path)

# Dataset & dataloader
dataset_config = NumpyDatasetConfig(
    tokenizer=tokenizer_config,
    name=NumpyDatasetType.fsl,
    paths=[wiki_tokens_path],
    sequence_length=SEQUENCE_LENGTH,
    work_dir=os.path.join(data_dir, "dataset_work")
)
dataset = dataset_config.build()

dataloader_config = NumpyDataLoaderConfig(
    global_batch_size=GLOBAL_BATCH_SIZE,
    seed=42,
    num_workers=0,
)
dataloader = dataloader_config.build(dataset)

# Model config
model_config = TransformerConfig.olmo2_190M(
    vocab_size=tokenizer_config.padded_vocab_size(),
    dtype=DType.bfloat16 if device.type == "cuda" else DType.float32,
    init_method=InitMethod.normal
)

# Build model
model = model_config.build(init_device=device)

# ✅ Set activation checkpointing after building the model
model.activation_checkpointing_config = TransformerActivationCheckpointingConfig(
    mode=TransformerActivationCheckpointingMode.full
)

# Token ID 0 handling
with torch.no_grad():
    model.embeddings.weight[0].zero_()
    if hasattr(model.lm_head, "w_out") and model.lm_head.w_out.bias is not None:
        model.lm_head.w_out.bias[0] = -100.0

# Optimizer
optim_config = AdamWConfig(
    lr=4e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    fused=False,
    group_overrides=[
        OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
    ]
)

# Training module
train_module_config = TransformerTrainModuleConfig(
    rank_microbatch_size=GLOBAL_BATCH_SIZE,
    max_sequence_length=SEQUENCE_LENGTH,
    optim=optim_config,
    compile_model=False
)
train_module = train_module_config.build(model=model, device=device)

# Inference callback
class InferenceCallback(Callback):
    def __init__(self, model, tokenizer_config, prompt, interval):
        self.model = model
        self.tokenizer_config = tokenizer_config
        self.prompt = prompt
        self.interval = interval
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")

    def pre_train(self):
        self.run_inference(0)

    def post_step(self):
        if self.trainer.global_step % self.interval == 0 and self.trainer.global_step > 0:
            self.run_inference(self.trainer.global_step)

    def run_inference(self, step):
        self.model.eval()
        tokens = [t for t in self.tokenizer.encode(self.prompt) if t != 0]
        input_tensor = torch.tensor([tokens], device=self.model.device)
        generated = tokens.copy()

        with torch.no_grad():
            logits = self.model(input_tensor)
            for _ in range(50):
                next_token_logits = logits[0, -1, :] / 0.8
                next_token_logits[0] = -float("inf")
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                if next_token == self.tokenizer_config.eos_token_id:
                    break
                generated.append(next_token)
                input_tensor = torch.tensor([generated], device=self.model.device)
                logits = self.model(input_tensor)

        decoded = self.tokenizer.decode(generated)
        print(f"[Step {step}] Generated: {decoded}")
        self.model.train()

# Trainer config (✅ Add work_dir to avoid Checkpointer error)
trainer_config = TrainerConfig(
    save_folder=os.path.join(data_dir, "checkpoints"),  # ✅ Required, can't be None
    save_overwrite=True,
    work_dir=os.path.join(data_dir, "trainer_work_dir"),
    metrics_collect_interval=10,
    cancel_check_interval=5,
    max_duration=Duration.steps(args.steps),
    device=str(device),
).with_callback("inference", InferenceCallback(
    model=model,
    tokenizer_config=tokenizer_config,
    prompt=args.prompt,
    interval=args.inference_interval
))

trainer = trainer_config.build(train_module=train_module, data_loader=dataloader)

# Train
print(f"🚀 Training for {args.steps} steps on device: {device}")
trainer.fit()
print("✅ Training complete")
