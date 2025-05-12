import torch
from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig, InitMethod
#highlight-start
from olmo_core.train.train_module.transformer import (
    TransformerTrainModuleConfig,
    TransformerDataParallelConfig,  # Import FSDP related config
    TransformerDataParallelWrappingStrategy  # Import FSDP wrapping strategy
)
#highlight-end
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train.train_module.transformer.config import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode
)
#highlight-start
from olmo_core.distributed.parallel import DataParallelType # Import DataParallelType
#highlight-end

def build_model(vocab_size, device, sequence_length, lr, weight_decay, betas):
    model_config = TransformerConfig.olmo2_190M(
        vocab_size=vocab_size,
        dtype=DType.bfloat16 if device.type == "cuda" else DType.float32,
        init_method=InitMethod.normal,
        #This activates GQA: mulitple query heads will share the same key value heads
        # If n_heads is larger than n_kv_heads, then GQA is used
        # To enable GQA set nkv_heads to a smaller number than n_heads
        n_kv_heads=3
    )

    model = model_config.build(init_device=device)
    # The activation checkpointing config should ideally be part of the TransformerTrainModuleConfig
    # model.activation_checkpointing_config = TransformerActivationCheckpointingConfig(
    # mode=TransformerActivationCheckpointingMode.full
    # )

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

    #highlight-start
    # Configure FSDP
    # You might need to adjust these FSDP parameters based on your specific setup and needs.
    # For example, `param_dtype` and `reduce_dtype` can be set to `DType.bfloat16`
    # if your hardware supports it and you want to save memory.
    fsdp_config = TransformerDataParallelConfig(
        name=DataParallelType.fsdp,  # Specify FSDP
        wrapping_strategy=TransformerDataParallelWrappingStrategy.full, # Or .blocks / .fine_grained
        # param_dtype=DType.bfloat16, # Optional: if you want params in bfloat16
        # reduce_dtype=DType.bfloat16, # Optional: if you want reductions in bfloat16
    )
    #highlight-end

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=sequence_length,
        max_sequence_length=sequence_length,
        optim=optim_config,
        compile_model=False,
        #highlight-start
        dp_config=fsdp_config,  # Add the FSDP config here
        ac_config=TransformerActivationCheckpointingConfig( # Activation checkpointing config
             mode=TransformerActivationCheckpointingMode.full
        )
        #highlight-end
    )

    train_module = train_module_config.build(model=model, device=device)
    return model, train_module