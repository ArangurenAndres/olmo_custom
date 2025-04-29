import sys
import os
import torch

# === Fix the path so we can import olmo_core and utils ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Import utility function to load config ===
from utils.load_config import load_config

# === Now import olmo_core ===
from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig, InitMethod
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train.train_module.transformer.config import TransformerActivationCheckpointingConfig, TransformerActivationCheckpointingMode

# === Build model ===
def build_model(vocab_size, device, sequence_length, lr, weight_decay, betas, n_kv_heads):
    # Create transformer configuration with specified number of key/value heads (n_kv_heads)
    model_config = TransformerConfig.olmo2_190M(
        vocab_size=vocab_size,
        dtype=DType.bfloat16 if device.type == "cuda" else DType.float32,
        init_method=InitMethod.normal,
        n_kv_heads=n_kv_heads  
    )

    # Instantiate the model from the config
    model = model_config.build(init_device=device)
    model.activation_checkpointing_config = TransformerActivationCheckpointingConfig(
        mode=TransformerActivationCheckpointingMode.full
    )

    with torch.no_grad():
        model.embeddings.weight[0].zero_()
        if hasattr(model.lm_head, "w_out") and model.lm_head.w_out.bias is not None:
            model.lm_head.w_out.bias[0] = -100.0

    # Configure the optimizer
    optim_config = AdamWConfig(
        lr=lr,
        weight_decay=weight_decay,
        betas=tuple(betas),
        fused=False,
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ]
    )
     # Build the training module
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=sequence_length,
        max_sequence_length=sequence_length,
        optim=optim_config,
        compile_model=False
    )

    train_module = train_module_config.build(model=model, device=device)
    return model, train_module

# === GQA assertion ===
def assert_gqa(model, expected_n_heads=12):
    """
    Test whether GQA is correctly applied by verifying:
    n_kv_heads < n_heads.
    """
    # Access the attention layer of the first transformer block
    attention_layer = model.blocks["0"].attention
    # === Extract attention dimensions ===
    hidden_dim = attention_layer.w_q.in_features # Total input dimension to queries
    q_out_features = attention_layer.w_q.out_features # Total output dimension for queries
    k_out_features = attention_layer.w_k.out_features # Total output dimension for keys
    v_out_features = attention_layer.w_v.out_features # Total output dimension for values
    # === Infer number of key/value heads ===
    n_heads = expected_n_heads
    head_dim = hidden_dim // n_heads
    n_kv_heads = k_out_features // head_dim # Each KV head has head_dim; compute how many there are


    print(f"Hidden dim: {hidden_dim}, head dim: {head_dim}")
    print(f"q_out_features: {q_out_features}, k_out_features: {k_out_features}, v_out_features: {v_out_features}")
    print(f"Detected n_heads: {n_heads}, n_kv_heads: {n_kv_heads}")

    # ===  GQA check ===
    # If n_kv_heads < n_heads, then GQA is active (KV heads are shared across Q heads)
    assert n_kv_heads < n_heads, f" GQA not active! n_kv_heads ({n_kv_heads}) is not smaller than n_heads ({n_heads})."
    print(" GQA is correctly active (n_kv_heads < n_heads).")

# === Main ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Load config
    config = load_config()

    # Read config values
    vocab_size = 50257  # Static
    sequence_length = config["sequence_length"]
    lr = config["learning_rate"]
    weight_decay = config["weight_decay"]
    betas = config["betas"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]

    #  Build the model using config
    model, train_module = build_model(vocab_size, device, sequence_length, lr, weight_decay, betas, n_kv_heads)

    #  Assert GQA is active
    assert_gqa(model, expected_n_heads=n_heads)

    #  Dummy forward pass to verify model works
    batch_size = 2
    dummy_input = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)
    outputs = model(dummy_input)
    print(" Dummy forward pass successful!")
