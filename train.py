import os
import torch
from olmo_core.train import TrainerConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from olmo_core.utils import seed_all

from utils.dataloader import prepare_data
from utils.model import build_model
from utils.inference import InferenceCallback

def run(data_dir="./data", steps=10, batch_size=1, prompt="Dutch is ", sequence_length=1024):
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    total_tokens = steps * batch_size * sequence_length
    total_sequences = int(total_tokens * 1.1 / sequence_length)

    loader, tokenizer_config = prepare_data(data_dir, total_sequences)
    model, train_module = build_model(
        vocab_size=tokenizer_config.padded_vocab_size(),
        device=device,
        sequence_length=sequence_length
    )

    save_dir = os.path.join(data_dir, "checkpoints")
    work_dir = os.path.join(data_dir, "trainer_work_dir")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    callback = InferenceCallback(model, tokenizer_config, prompt, interval=5)

    trainer_config = TrainerConfig(
        save_folder=save_dir,
        save_overwrite=True,
        work_dir=work_dir,
        metrics_collect_interval=10,
        cancel_check_interval=5,
        max_duration=Duration.steps(steps),
        device=str(device),
    ).with_callback("inference", callback)

    trainer = trainer_config.build(train_module=train_module, data_loader=loader)
    trainer.fit()

if __name__ == "__main__":
    run()
