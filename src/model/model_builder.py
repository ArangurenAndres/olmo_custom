"""
OLMo model builder with GQA and Layer-Wise Scaling support.
"""

import torch
import logging
from typing import Dict, Optional, Union, Any, Tuple

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig, InitMethod
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer import TransformerTrainModule
from olmo_core.train.train_module.transformer.config import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode
)

from .lws_layer import apply_layer_wise_scaling, verify_lws_applied

logger = logging.getLogger(__name__)

def build_olmo_model(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    dtype: Optional[DType] = None
) -> torch.nn.Module:
    """
    Build OLMo model with Layer-Wise Scaling and GQA support.
    
    Args:
        config: Configuration dictionary
        device: Device to place the model on
        dtype: Data type for model parameters
        
    Returns:
        model: OLMo model with LWS and GQA
    """
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set dtype if not provided
    if dtype is None:
        dtype = DType.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else DType.float32
    
    # Extract model parameters from config
    vocab_size = config.get("vocab_size", 50304)
    n_heads = config.get("n_heads", 12)
    n_kv_heads = config.get("n_kv_heads", n_heads)  # Default to standard MHA if not specified
    global_dim = config.get("global_dim", 768)
    head_dim = config.get("head_dim", 64)
    
    logger.info(f"Building OLMo model with {n_heads} query heads and {n_kv_heads} KV heads")
    
    # Set up model configuration
    model_config = TransformerConfig.olmo2_190M(
        vocab_size=vocab_size,
        dtype=dtype,
        init_method=InitMethod.normal,
        # Support for GQA: n_kv_heads can be smaller than n_heads
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        # Set the global dimension and head dimension
        global_dim=global_dim,
        head_dim=head_dim
    )
    
    # Build the model
    model = model_config.build(init_device=device)
    
    # Configure activation checkpointing to save memory
    model.activation_checkpointing_config = TransformerActivationCheckpointingConfig(
        mode=TransformerActivationCheckpointingMode.full
    )
    
    # Apply Layer-Wise Scaling if configured
    if "fnn_scalars" in config or "qkv_scalars" in config or "output_scalars" in config:
        logger.info("Applying Layer-Wise Scaling")
        model = apply_layer_wise_scaling(
            model, 
            ffn_scalars=config.get("fnn_scalars", None),
            qkv_scalars=config.get("qkv_scalars", None),
            output_scalars=config.get("output_scalars", None),
            verbose=config.get("verbose_scaling", False)
        )
        
        # Verify LWS was applied correctly
        if verify_lws_applied(model, config.get("fnn_scalars", None), config.get("qkv_scalars", None)):
            logger.info("Layer-Wise Scaling verified successfully")
        else:
            logger.warning("Layer-Wise Scaling verification failed")
    
    # Handle token ID 0 (special token masking)
    with torch.no_grad():
        model.embeddings.weight[0].zero_()
        if hasattr(model.lm_head, "w_out") and model.lm_head.w_out.bias is not None:
            model.lm_head.w_out.bias[0] = -100.0
    
    # Verify GQA setup
    verify_gqa(model, n_heads, n_kv_heads)
    
    return model

def verify_gqa(
    model: torch.nn.Module,
    expected_n_heads: int,
    expected_n_kv_heads: int
) -> bool:
    """
    Verify that Grouped Query Attention is set up correctly.
    
    Args:
        model: OLMo model
        expected_n_heads: Expected number of query heads
        expected_n_kv_heads: Expected number of key/value heads
        
    Returns:
        bool: True if GQA is set up correctly
    """
    if not hasattr(model, 'blocks') or len(model.blocks) == 0:
        logger.error("Model has no transformer blocks")
        return False
    
    # Check the first block's attention layer
    first_block = model.blocks['0']
    attention = first_block.attention
    
    # Check attention head configuration
    if hasattr(attention, 'n_heads') and hasattr(attention, 'n_kv_heads'):
        actual_n_heads = attention.n_heads
        actual_n_kv_heads = attention.n_kv_heads
        
        if actual_n_heads != expected_n_heads or actual_n_kv_heads != expected_n_kv_heads:
            logger.error(f"GQA configuration mismatch: "
                       f"Expected {expected_n_heads} query heads and {expected_n_kv_heads} KV heads, "
                       f"got {actual_n_heads} query heads and {actual_n_kv_heads} KV heads")
            return False
        
        is_gqa = actual_n_kv_heads < actual_n_heads
        logger.info(f"{'GQA' if is_gqa else 'Standard MHA'} is active: "
                  f"{actual_n_heads} query heads, {actual_n_kv_heads} KV heads")
        
        return True
    else:
        logger.error("Attention layer doesn't have n_heads or n_kv_heads attributes")
        return False

def build_optimizer_config(
    config: Dict[str, Any]
) -> AdamWConfig:
    """
    Build optimizer configuration with weight decay and parameter groups.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        optimizer_config: Optimizer configuration
    """
    # Extract optimizer parameters from config
    lr = config.get("learning_rate", 4e-4)
    weight_decay = config.get("weight_decay", 0.1)
    betas = config.get("betas", (0.9, 0.95))
    
    # Set up optimizer configuration
    optimizer_config = AdamWConfig(
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        fused=torch.cuda.is_available(),  # Use fused implementation if CUDA is available
        group_overrides=[
            # No weight decay for normalization layers
            OptimGroupOverride("^(transformer|embeddings).*norm.*", weight_decay=0.0),
            # No weight decay for bias terms
            OptimGroupOverride("^.*bias", weight_decay=0.0),
            # Also exclude embedding weights from weight decay (following LLaMa recipe)
            OptimGroupOverride("^embeddings.weight", weight_decay=0.0)
        ]
    )
    
    return optimizer_config

def build_train_module(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> TransformerTrainModule:
    """
    Build the training module with the model and optimizer.
    
    Args:
        model: OLMo model
        config: Configuration dictionary
        device: Device to place the model on
        
    Returns:
        train_module: TransformerTrainModule for training
    """
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract training parameters
    sequence_length = config.get("sequence_length", 1024)
    batch_size = config.get("batch_size", 8)
    compile_model = config.get("compile_model", False)
    
    # Get optimizer config
    optimizer_config = build_optimizer_config(config)
    
    # Create train module config
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=batch_size,
        max_sequence_length=sequence_length,
        optim=optimizer_config,
        compile_model=compile_model
    )
    
    # Build train module
    train_module = train_module_config.build(model=model, device=device)
    
    return train_module