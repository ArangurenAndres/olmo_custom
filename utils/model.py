import torch
from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig, InitMethod
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train.train_module.transformer.config import TransformerActivationCheckpointingConfig, TransformerActivationCheckpointingMode

def build_model(vocab_size, device, sequence_length, lr, weight_decay, betas, config):
    # Extract optional GQA-related parameters from config
    n_kv_heads = config.get("n_kv_heads", 3)
    n_heads = config.get("n_heads", 12)

    print(f"Building model with n_heads={n_heads}, n_kv_heads={n_kv_heads}, device={device}")

    model_config = TransformerConfig.olmo2_190M(
        vocab_size=vocab_size,
        dtype=DType.bfloat16 if device.type == "cuda" else DType.float32,
        init_method=InitMethod.normal,
        n_kv_heads=n_kv_heads,
        n_heads=n_heads
    )

    model = model_config.build(init_device=device)

    model.activation_checkpointing_config = TransformerActivationCheckpointingConfig(
        mode=TransformerActivationCheckpointingMode.full
    )

    # Optional: zero out [PAD] token embedding
    with torch.no_grad():
        model.embeddings.weight[0].zero_()
        if hasattr(model.lm_head, "w_out") and model.lm_head.w_out.bias is not None:
            model.lm_head.w_out.bias[0] = -100.0

    optim_config = AdamWConfig(
        lr=lr,
        weight_decay=weight_decay,
        betas=tuple(betas),
        fused=False,
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ]
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=sequence_length,
        max_sequence_length=sequence_length,
        optim=optim_config,
        compile_model=False
    )

    train_module = train_module_config.build(model=model, device=device)

    return model, train_module
