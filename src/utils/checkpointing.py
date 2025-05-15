"""
FSDP-compatible checkpointing utilities for OLMo training.
"""

import os
import json
import time
import logging
import torch
from typing import Dict, Any, Optional, List, Union, Tuple
import copy
import shutil

from olmo_core.distributed.utils import (
    is_distributed,
    get_rank,
    barrier
)

logger = logging.getLogger(__name__)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    step: int,
    loss: float,
    save_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    save_optimizer: bool = True,
    extra_data: Optional[Dict[str, Any]] = None,
    is_fsdp: bool = False,
    max_to_keep: int = 3,
    verbose: bool = True
) -> str:
    """
    Save a checkpoint with support for FSDP.
    
    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        scheduler: Scheduler to save.
        step: Training step.
        loss: Training loss.
        save_dir: Directory to save checkpoint in.
        metrics: Optional dictionary of metrics to save.
        filename: Optional filename to use. If None, use step number.
        save_optimizer: Whether to save optimizer state.
        extra_data: Optional additional data to include in checkpoint.
        is_fsdp: Whether model is wrapped with FSDP.
        max_to_keep: Maximum number of checkpoints to keep.
        verbose: Whether to log checkpoint information.
        
    Returns:
        str: Path to saved checkpoint.
    """
    # Only save on the main process
    if is_distributed() and get_rank() != 0:
        barrier()  # Wait for primary rank to finish saving
        return ""
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up checkpoint path
    if filename is None:
        filename = f"checkpoint_{step:07d}.pt"
    checkpoint_path = os.path.join(save_dir, filename)
    
    # Set up metadata
    metadata = {
        'step': step,
        'loss': float(loss),
        'timestamp': time.time(),
        'metrics': metrics or {}
    }
    
    # Add any extra data
    if extra_data:
        metadata.update(extra_data)
    
    # Create checkpoint dictionary
    checkpoint = {
        'metadata': metadata,
        'model_config': getattr(model, 'config', None),  # Save model config if available
    }
    
    # Save model state
    if is_fsdp:
        # For FSDP, we need to use special handling of state_dict
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType
        )
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            checkpoint['model_state_dict'] = model.state_dict()
    else:
        # Regular model
        checkpoint['model_state_dict'] = model.state_dict()
    
    # Save optimizer and scheduler if requested
    if save_optimizer and optimizer is not None:
        if is_fsdp:
            # For FSDP optimizer states
            from torch.distributed.fsdp.api import (
                FullOptimStateDictConfig,
                FullStateDictConfig
            )
            optim_state_dict_config = FullOptimStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True
            )
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT,
                optim_state_dict_config=optim_state_dict_config
            ):
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        else:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Also save metadata separately as JSON for easy inspection
    metadata_path = os.path.join(save_dir, f"metadata_{step:07d}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Create a symlink to the latest checkpoint
    latest_path = os.path.join(save_dir, "checkpoint_latest.pt")
    if os.path.exists(latest_path):
        if os.path.islink(latest_path):
            os.unlink(latest_path)
        else:
            os.remove(latest_path)
    
    # Create relative symlink
    os.symlink(os.path.basename(checkpoint_path), latest_path)
    
    # Clean up old checkpoints if needed
    if max_to_keep > 0:
        _cleanup_old_checkpoints(save_dir, max_to_keep)
    
    if verbose:
        logger.info(f"Saved checkpoint at step {step} to {checkpoint_path}")
    
    # Wait for all processes
    if is_distributed():
        barrier()
    
    return checkpoint_path

def _cleanup_old_checkpoints(save_dir: str, max_to_keep: int) -> None:
    """
    Clean up old checkpoints, keeping only the latest max_to_keep.
    
    Args:
        save_dir: Directory containing checkpoints.
        max_to_keep: Maximum number of checkpoints to keep.
    """
    # Get all checkpoint files
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Remove old checkpoints
    checkpoints_to_remove = checkpoints[:-max_to_keep]
    
    for checkpoint in checkpoints_to_remove:
        checkpoint_path = os.path.join(save_dir, checkpoint)
        
        # Skip latest symlink
        if os.path.islink(checkpoint_path) and os.path.basename(checkpoint_path) == "checkpoint_latest.pt":
            continue
        
        # Remove checkpoint file
        os.remove(checkpoint_path)
        
        # Also remove corresponding metadata file if it exists
        metadata_file = checkpoint.replace("checkpoint_", "metadata_")
        metadata_path = os.path.join(save_dir, metadata_file)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    checkpoint_path: str,
    strict: bool = True,
    load_optimizer: bool = True,
    is_fsdp: bool = False,
    device: Optional[Union[torch.device, str]] = None
) -> Dict[str, Any]:
    """
    Load a checkpoint with support for FSDP.
    
    Args:
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        checkpoint_path: Path to checkpoint file.
        strict: Whether to require that the keys in the model state dict match.
        load_optimizer: Whether to load optimizer state.
        is_fsdp: Whether model is wrapped with FSDP.
        device: Device to load tensors onto.
        
    Returns:
        Dict: Metadata from the checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load for all ranks in FSDP
    device_map = {'cuda': 'cuda', 'cpu': 'cpu'}
    map_location = device_map.get(device, None) if isinstance(device, str) else device
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Load model state
    if is_fsdp:
        # For FSDP, we need to use special handling of state_dict
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType
        )
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        # Regular model
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state if requested
    if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        if is_fsdp:
            # For FSDP optimizer states
            from torch.distributed.fsdp.api import (
                FullOptimStateDictConfig,
                FullStateDictConfig
            )
            optim_state_dict_config = FullOptimStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True
            )
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT,
                optim_state_dict_config=optim_state_dict_config
            ):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Wait for all processes in distributed setting
    if is_distributed():
        barrier()
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Return metadata
    return checkpoint.get('metadata', {})

def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        save_dir: Directory to search in.
        
    Returns:
        Optional[str]: Path to latest checkpoint or None if not found.
    """
    # Check for symlink first
    latest_link = os.path.join(save_dir, "checkpoint_latest.pt")
    if os.path.islink(latest_link):
        target = os.readlink(latest_link)
        target_path = os.path.join(save_dir, target)
        if os.path.exists(target_path):
            return target_path
    
    # Otherwise, find the checkpoint with the highest step number
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    
    if not checkpoints:
        return None
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    latest_checkpoint = checkpoints[-1]
    return os.path.join(save_dir, latest_checkpoint)

def convert_checkpoint_to_hf(
    checkpoint_path: str,
    output_dir: str,
    config_override: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convert an OLMo checkpoint to a format compatible with Hugging Face Transformers.
    
    Args:
        checkpoint_path: Path to OLMo checkpoint.
        output_dir: Directory to save HF model.
        config_override: Optional configuration overrides.
        
    Returns:
        str: Path to saved HF model.
    """
    try:
        import transformers
        from transformers import AutoConfig
    except ImportError:
        raise ImportError("transformers library is required for convert_checkpoint_to_hf")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration
    model_config = checkpoint.get('model_config', None)
    
    if model_config is None:
        raise ValueError("Model configuration not found in checkpoint")
    
    # Set up HF config - this is a simplified version that may need customization
    config_dict = {
        "vocab_size": model_config.vocab_size,
        "hidden_size": model_config.global_dim,
        "num_hidden_layers": len(model_config.blocks),
        "num_attention_heads": model_config.n_heads,
        "intermediate_size": model_config.ffn_dim,
        "num_key_value_heads": getattr(model_config, 'n_kv_heads', model_config.n_heads),  # GQA support
        "hidden_act": "silu",  # OLMo uses SwiGLU
        "max_position_embeddings": 2048,  # Default, may need to be adjusted
        "torch_dtype": "bfloat16" if model_config.dtype.name == "bfloat16" else "float32",
        "model_type": "olmo",
        "architectures": ["OLMoForCausalLM"]
    }
    
    # Apply any configuration overrides
    if config_override:
        config_dict.update(config_override)
    
    # Create HF config
    hf_config = AutoConfig.for_model(**config_dict)
    
    # Save config
    hf_config.save_pretrained(output_dir)
    
    # Extract and convert model weights - this is a placeholder
    # A complete implementation would map OLMo weights to HF model structure
    logger.warning("Full weight conversion not implemented - saving raw state dict")
    
    # Save model state dict
    torch.save(checkpoint['model_state_dict'], os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save metadata
    metadata = checkpoint.get('metadata', {})
    with open(os.path.join(output_dir, "olmo_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Converted checkpoint saved to {output_dir}")
    
    return output_dir