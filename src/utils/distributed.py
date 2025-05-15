"""
Distributed training helpers for OLMo.
"""

import os
import torch
import random
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple

from olmo_core.distributed.utils import (
    is_distributed,
    get_rank,
    get_world_size,
    get_local_rank,
    get_local_world_size,
    backend_supports_cuda
)

import logging
logger = logging.getLogger(__name__)

def setup_device(
    device_id: Optional[int] = None,
    use_cuda: bool = True,
    use_cuda_amp: bool = True
) -> Tuple[torch.device, Dict[str, Any]]:
    """
    Set up the device for training.
    
    Args:
        device_id: Specific device ID to use. If None, use local rank in distributed mode.
        use_cuda: Whether to use CUDA if available.
        use_cuda_amp: Whether to use AMP with CUDA.
        
    Returns:
        Tuple of:
            - torch.device: Device to use for training.
            - Dict[str, Any]: Mixed precision settings.
    """
    # Determine available devices
    cuda_available = torch.cuda.is_available() and use_cuda
    
    # Set device
    if cuda_available:
        if is_distributed():
            # In distributed mode, use the local rank as the device ID
            if device_id is None:
                device_id = get_local_rank()
            torch.cuda.set_device(device_id)
            device = torch.device("cuda", device_id)
        else:
            # In non-distributed mode, use the specified device or device 0
            if device_id is None:
                device_id = 0
            torch.cuda.set_device(device_id)
            device = torch.device("cuda", device_id)
    else:
        device = torch.device("cpu")
        if is_distributed():
            logger.warning("CUDA is not available, but distributed training is enabled. This might be very slow.")
    
    # Set up mixed precision
    mixed_precision = {}
    if cuda_available and use_cuda_amp:
        # Check if we can use bfloat16
        if torch.cuda.is_bf16_supported():
            mixed_precision["dtype"] = torch.bfloat16
        else:
            mixed_precision["dtype"] = torch.float16
    
    logger.info(f"Using device: {device}")
    
    return device, mixed_precision

def init_random_seeds(seed: int = 42, set_torch_deterministic: bool = False) -> None:
    """
    Initialize random seeds for reproducibility.
    
    Args:
        seed: Base seed to use.
        set_torch_deterministic: Whether to set torch.backends.cudnn.deterministic.
            This can significantly slow down training.
    """
    # In distributed mode, add rank to seed to avoid identical initialization across processes
    if is_distributed():
        seed += get_rank()
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Additional settings for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if set_torch_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.warning(
                "torch.backends.cudnn.deterministic = True has been set. "
                "This can significantly slow down training."
            )
    
    logger.info(f"Random seeds initialized to {seed}")

def is_main_process() -> bool:
    """
    Check if the current process is the main process (rank 0).
    
    Returns:
        bool: True if the process is the main process, False otherwise.
    """
    return not is_distributed() or get_rank() == 0

def get_device_map(
    num_devices: Optional[int] = None,
    balanced: bool = True
) -> Dict[str, int]:
    """
    Get a device map for model parallel training.
    
    Args:
        num_devices: Number of devices to use. If None, use all available devices.
        balanced: Whether to balance layers across devices.
        
    Returns:
        Dict[str, int]: Device map for model parallel training.
    """
    if not torch.cuda.is_available():
        return {"": "cpu"}
    
    # Determine number of devices
    if num_devices is None:
        if is_distributed():
            num_devices = get_local_world_size()
        else:
            num_devices = torch.cuda.device_count()
    
    # Create device map
    if not balanced:
        # Simple device map, just alternate devices
        return {"cpu": "cpu"}  # Placeholder - this needs custom mapping logic
    
    # Placeholder for more sophisticated balanced device mapping
    # This would require knowledge of the model structure
    return {"cpu": "cpu"}  # Placeholder - this needs custom mapping logic

def synchronize_model_parameters(model: torch.nn.Module) -> None:
    """
    Synchronize model parameters across processes in distributed mode.
    
    Args:
        model: Model to synchronize.
    """
    if not is_distributed():
        return
    
    # Ensure all processes have the same model weights
    for param in model.parameters():
        torch.distributed.broadcast(param.data, src=0)

def gather_results(
    tensor: torch.Tensor,
    dst: int = 0
) -> Optional[torch.Tensor]:
    """
    Gather tensors from all processes at the destination process.
    
    Args:
        tensor: Tensor to gather.
        dst: Destination rank.
        
    Returns:
        Optional[torch.Tensor]: Gathered tensors on dst rank, None on other ranks.
    """
    if not is_distributed():
        return tensor
    
    # Move tensor to CPU to avoid device mismatch
    tensor = tensor.cpu()
    
    # Get tensor shape
    shape = list(tensor.shape)
    
    # Create output tensor
    if get_rank() == dst:
        result = torch.zeros(
            [get_world_size()] + shape,
            dtype=tensor.dtype
        )
    else:
        result = None
    
    # Gather tensors
    torch.distributed.gather(tensor, result, dst=dst)
    
    return result

def all_reduce_average(tensor: torch.Tensor) -> torch.Tensor:
    """
    Average tensor across all processes.
    
    Args:
        tensor: Tensor to average.
        
    Returns:
        torch.Tensor: Averaged tensor.
    """
    if not is_distributed():
        return tensor
    
    # Create a copy of the tensor
    result = tensor.clone()
    
    # All-reduce
    torch.distributed.all_reduce(result)
    
    # Average
    result /= get_world_size()
    
    return result