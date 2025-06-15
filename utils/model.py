import torch
from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig, InitMethod
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train.train_module.transformer.config import TransformerActivationCheckpointingConfig, TransformerActivationCheckpointingMode

from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer.config import TransformerDataParallelConfig
from olmo_core.distributed.parallel import DataParallelType

from olmo_core.distributed.utils import is_distributed, get_rank, get_world_size

import logging


def build_model(vocab_size, device, config):
    sequence_length = config["sequence_length"]
    n_kv_heads = config["n_kv_heads"]

    model_config = TransformerConfig.olmo2_190M(
        vocab_size=vocab_size,
        dtype=DType.bfloat16 if device.type == "cuda" else DType.float32,
        n_kv_heads=n_kv_heads,
        use_flash=config["use_flash_attn"],
        #init_method=InitMethod.normal   # GIVES AN ERROR IF SET   
    )

    model = model_config.build(init_device=device)
    model.activation_checkpointing_config = TransformerActivationCheckpointingConfig(
        mode=TransformerActivationCheckpointingMode.full
    )

    # ---------------------------------------------------------
    # Flash-Attention sanity check (easy-to-spot log message)
    # ---------------------------------------------------------
    def _log_flash_attention_status(m):
        """Walk through all sub-modules, collect `module.attn.use_flash_attn` flags,
        and emit a clear log. This is agnostic to model internals and works even
        when layers are stored in `ModuleList`s or custom containers."""
        flags = []
        for sub in m.modules():
            if hasattr(sub, "use_flash"):
                flags.append(getattr(sub, "use_flash", None))
                continue
            attn = getattr(sub, "attn", None)
            if attn is not None and hasattr(attn, "use_flash"):
                flags.append(getattr(attn, "use_flash", None))

        if not flags:  # could not find any attention modules
            logging.warning("⚠️  Could not locate attention modules to verify Flash-Attention; skipping check.")
            return

        if all(flags):
            logging.info("\n🔆 FLASH-ATTENTION ENABLED on ALL %d attention layers — fast kernels will be used.\n", len(flags))
        else:
            disabled_layers = [idx for idx, f in enumerate(flags) if not f]
            message = f"Flash-Attention disabled on layers: {disabled_layers}"
            logging.warning("\n❌ %s\n", message)

    # Run the check only once (rank 0) to avoid duplicated messages in DDP
    if not is_distributed() or get_rank() == 0:
        _log_flash_attention_status(model)

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

def build_train_module_with_fsdp(model, config):
    # Define optimizer configuration
    optimizer_config = AdamWConfig(
        lr=config.get("learning_rate", 1e-4),
        betas=tuple(config.get("betas", [0.9, 0.95])),
        eps=config.get("eps", 1e-8),
        weight_decay=config.get("weight_decay", 0.1),
        group_overrides=[
            OptimGroupOverride(
                params=["embeddings.weight"],  # Fixed: Only embeddings.weight pattern
                opts=dict(weight_decay=0.0)
            )
        ]
    )
    

    # Configure DDP instead of FSDP
    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.ddp,
        param_dtype=DType.bfloat16 if config.get("param_dtype", "bfloat16") == "bfloat16" else None,
        reduce_dtype=DType.float32,
    )
    
    train_module_config = TransformerTrainModuleConfig(
        optim=optimizer_config,
        dp_config=dp_config,
        max_sequence_length=config["sequence_length"],
        # Convert micro_batch_size (sequences) -> tokens for rank_microbatch_size
        rank_microbatch_size=config["micro_batch_size"] * config["sequence_length"],
        compile_model=False,
        max_grad_norm=1.0  # Add gradient clipping
    )
    
    # Pass model to build() method, not to constructor
    return train_module_config.build(model=model)

# def create_distributed_trainer(train_module, dataloader, config):
#     # Configure trainer for distributed training
#     trainer_config = TrainerConfig(
#         save_folder=config["save_dir"],
#         save_overwrite=True,
#         max_duration=Duration.steps(config["steps"]),
        
#         # Distributed-specific settings
#         save_interval=config.get("save_interval", 100),
#         log_interval=config.get("log_interval", 10),
#         metrics_collect_interval=config.get("metrics_interval", 10),
        
#         # Enable async checkpointing for better performance
#         async_bookkeeping=True,
#     )
    
#     # Add callbacks (only on rank 0 for logging)
#     if not is_distributed() or get_rank() == 0:
#         # Add your inference callback, wandb, etc.
#         pass
    
#     return trainer_config.build(train_module, dataloader)