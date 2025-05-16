"""
Utilities for Fully Sharded Data Parallel (FSDP) training with OLMo.
This file integrates with OLMo's built-in distributed training support.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, List
from datetime import timedelta

import torch.distributed.fsdp as fsdp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy
)

from olmo_core.distributed.utils import (
    init_distributed,
    is_distributed,
    get_rank,
    get_world_size,
    get_local_rank,
    barrier,
    backend_supports_cuda
)
from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerBlock

logger = logging.getLogger(__name__)

def setup_fsdp_environment(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "29500",
    backend: str = "nccl",
    timeout: timedelta = timedelta(minutes=30)
) -> None:
    """
    Set up the distributed environment for FSDP training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        master_addr: Master address for rendezvous
        master_port: Master port for rendezvous
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        timeout: Timeout for initialization
    """
    # Set required environment variables
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank % torch.cuda.device_count())
    os.environ['LOCAL_WORLD_SIZE'] = str(torch.cuda.device_count())
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize distributed process group using OLMo's utility
    init_distributed(backend=backend, timeout=timeout)
    
    logger.info(f"Initialized FSDP process group: rank={get_rank()}, world_size={get_world_size()}")

def get_mixed_precision_policy(dtype: str = "bfloat16") -> MixedPrecision:
    """
    Get mixed precision policy for FSDP.
    
    Args:
        dtype: Data type for mixed precision ('bfloat16' or 'float16')
        
    Returns:
        MixedPrecision policy for FSDP
    """
    if dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        )
    elif dtype == "float16" and torch.cuda.is_available():
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
    else:
        return MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32
        )

def get_transformer_wrap_policy():
    """
    Get FSDP auto wrap policy for transformer blocks.
    
    Returns:
        auto_wrap_policy for FSDP
    """
    # Define a custom policy function that matches PyTorch's expected signature
    def custom_auto_wrap_policy(module, recurse, unwrapped_params):
        return isinstance(module, TransformerBlock)
    
    return custom_auto_wrap_policy

def wrap_model_with_fsdp(
    model: torch.nn.Module, 
    mixed_precision: bool = True,
    shard_strategy: str = "FULL_SHARD",
    cpu_offload: bool = False,
) -> torch.nn.Module:
    """
    Wrap model with FSDP.
    
    Args:
        model: OLMo model to wrap
        mixed_precision: Whether to use mixed precision
        shard_strategy: Sharding strategy ('FULL_SHARD', 'SHARD_GRAD_OP', or 'NO_SHARD')
        cpu_offload: Whether to offload parameters to CPU
        
    Returns:
        wrapped_model: FSDP-wrapped model
    """
    if not is_distributed():
        logger.warning("Not in distributed mode, returning unwrapped model")
        return model
    
    # Determine sharding strategy
    if shard_strategy.upper() == "FULL_SHARD":
        strategy = ShardingStrategy.FULL_SHARD
    elif shard_strategy.upper() == "SHARD_GRAD_OP":
        strategy = ShardingStrategy.SHARD_GRAD_OP
    else:
        strategy = ShardingStrategy.NO_SHARD
    
    # Set up mixed precision policy
    mp_policy = get_mixed_precision_policy("bfloat16") if mixed_precision else None
    
    # Auto wrap policy for transformer blocks
    auto_wrap_policy = get_transformer_wrap_policy()
    
    # Wrap model with FSDP
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=strategy,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
    )
    
    return fsdp_model

def save_fsdp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    step: int,
    loss: float,
    save_path: str,
) -> None:
    """
    Save checkpoint with FSDP support.
    
    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        step: Current training step
        loss: Training loss
        save_path: Path to save the checkpoint
    """
    # Wait for all processes to reach this point
    barrier()
    
    # Only save from rank 0
    if get_rank() == 0:
        # Create checkpoint dictionary
        checkpoint = {
            'step': step,
            'loss': loss,
        }
        
        # Add optimizer and scheduler states if available
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        # Get model state dict with FSDP.state_dict_type context
        with FSDP.state_dict_type(model, fsdp.StateDictType.FULL_STATE_DICT):
            checkpoint['model_state_dict'] = model.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        logger.info(f"Saved FSDP checkpoint to {save_path}")
    
    # Wait for checkpoint to be saved before continuing
    barrier()

def load_fsdp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    checkpoint_path: str,
) -> Dict[str, Any]:
    """
    Load checkpoint with FSDP support.
    
    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        checkpoint_path: Path to checkpoint
        
    Returns:
        Dict containing training metadata (step, loss, etc.)
    """
    # Load checkpoint on rank 0 and broadcast to all ranks
    if get_rank() == 0:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        checkpoint = None
    
    # Broadcast checkpoint from rank 0 to all ranks
    if is_distributed():
        object_list = [checkpoint]
        torch.distributed.broadcast_object_list(object_list, src=0)
        checkpoint = object_list[0]
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        with FSDP.state_dict_type(model, fsdp.StateDictType.FULL_STATE_DICT):
            model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Return training metadata
    metadata = {
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', float('inf')),
    }
    
    return metadata

def broadcast_model_parameters(model: torch.nn.Module, rank: int = 0) -> None:
    """
    Broadcast model parameters from a specific rank to all other ranks.
    
    Args:
        model: Model to synchronize
        rank: Source rank for parameters
    """
    if not is_distributed():
        return
    
    for param in model.parameters():
        torch.distributed.broadcast(param.data, src=rank)