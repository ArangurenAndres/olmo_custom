import torch
from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig, InitMethod
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train.train_module.transformer.config import TransformerActivationCheckpointingConfig, TransformerActivationCheckpointingMode

def build_model(vocab_size, device, config):
    sequence_length = config["sequence_length"]
    n_kv_heads = config["n_kv_heads"]

    model_config = TransformerConfig.olmo2_190M(
        vocab_size=vocab_size,
        dtype=DType.bfloat16 if device.type == "cuda" else DType.float32,
        n_kv_heads=n_kv_heads,
       # flash_attn=config["flash_attn"],
        #init_method=InitMethod.normal   # GIVES AN ERROR IF SET   
    )

    model = model_config.build(init_device=device)
    model.activation_checkpointing_config = TransformerActivationCheckpointingConfig(
        mode=TransformerActivationCheckpointingMode.full
    )

#    with torch.no_grad():
 #       model.embeddings.weight[0].zero_()
 #       if hasattr(model.lm_head, "w_out") and model.lm_head.w_out.bias is not None:
 #           model.lm_head.w_out.bias[0] = -100.0

    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    betas = tuple(config["betas"])

    optim_config = AdamWConfig(
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        fused=False,
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ]
    )

    rank_microbatch_size_in_tokens = config["micro_batch_size"] * sequence_length

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size_in_tokens,
        max_sequence_length=sequence_length,
        optim=optim_config,
        compile_model=False
    )

    train_module = train_module_config.build(model=model, device=device)
    return model, train_module
