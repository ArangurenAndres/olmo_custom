import torch
from olmo_core.config import ( # type: ignore
    DType,
    TransformerDataParallelConfig, # For dp_config
    DataParallelType # For specifying DDP
)
from olmo_core.nn.transformer import TransformerConfig, InitMethod # type: ignore
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig # type: ignore
from olmo_core.optim import AdamWConfig, OptimGroupOverride # type: ignore
from olmo_core.train.train_module.transformer.config import ( # type: ignore
    TransformerActivationCheckpointingConfig, TransformerActivationCheckpointingMode
)

# Helper function (can be defined here or imported if it's in a common utils)
def is_main_process_for_logging(device: torch.device) -> bool:
    """Checks if the current process is the main process for logging purposes."""
    if device.type == 'cuda':
        # In DDP, rank 0 is usually on cuda:0 if local_rank is 0
        # This is a heuristic for logging before full dist init or if global_rank is not passed
        return device.index == 0
    return True # For CPU, assume it's main or only process

def get_model_and_train_module_config(
    vocab_size: int,
    device: torch.device,
    sequence_length: int,
    lr: float,
    weight_decay: float,
    betas: list[float],
    batch_size_per_rank: int,
    is_distributed: bool, # Added flag to know if we are in DDP mode
    n_kv_heads_gqa: int | None = 3
):
    """
    Builds the core transformer model and the configuration for the TransformerTrainModule.
    If is_distributed is True, configures TransformerTrainModuleConfig for DDP.
    """
    model_dtype = DType.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else DType.float32
    if device.type == "cuda" and not torch.cuda.is_bf16_supported() and model_dtype == DType.bfloat16:
        print(f"Warning: Device {device} selected for bfloat16, but bfloat16 is not supported. Falling back to float32.")
        model_dtype = DType.float32
    
    log_rank_tag = device.index if device.type == 'cuda' and device.index is not None else "CPU"
    if is_main_process_for_logging(device):
        print(f"[Rank {log_rank_tag}] Model will be initialized with dtype: {model_dtype}")

    model_config = TransformerConfig.olmo2_190M(
        vocab_size=vocab_size,
        dtype=model_dtype,
        init_method=InitMethod.normal,
        n_kv_heads=n_kv_heads_gqa
    )

    if is_main_process_for_logging(device):
        print(f"[Rank {log_rank_tag}] Building model on device: {device}")
    model = model_config.build(init_device=device)

    model.activation_checkpointing_config = TransformerActivationCheckpointingConfig(
        mode=TransformerActivationCheckpointingMode.full
    )

    with torch.no_grad():
        if vocab_size > 0 :
             model.embeddings.weight[0].zero_()
        if hasattr(model.lm_head, "w_out") and model.lm_head.w_out is not None and \
           hasattr(model.lm_head.w_out, "bias") and model.lm_head.w_out.bias is not None:
            if model.lm_head.w_out.bias.numel() > 0:
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

    # --- Configure Data Parallelism for OLMo ---
    dp_train_config: TransformerDataParallelConfig | None = None
    if is_distributed:
        if is_main_process_for_logging(device):
            print(f"[Rank {log_rank_tag}] Configuring OLMo for DDP data parallelism.")
        # This tells OLMo to use DDP.
        # Other dp_config parameters (like reduce_dtype, param_dtype for FSDP) are not needed for basic DDP.
        dp_train_config = TransformerDataParallelConfig(name=DataParallelType.ddp)
    
    rank_microbatch_size_in_tokens = batch_size_per_rank * sequence_length
    if is_main_process_for_logging(device):
        print(f"[Rank {log_rank_tag}] Configuring TransformerTrainModule with rank_microbatch_size: {rank_microbatch_size_in_tokens} tokens "
              f"({batch_size_per_rank} sequences * {sequence_length} seq_len)")

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size_in_tokens,
        max_sequence_length=sequence_length,
        optim=optim_config,
        compile_model=False,
        dp_config=dp_train_config  # Pass the DDP configuration to OLMo
    )

    return model, train_module_config
