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
    d_model = config.get("d_model", 768)  # ADD THIS LINE
    n_heads = config.get("n_heads", 12)
    n_kv_heads = config.get("n_kv_heads", n_heads)  # Default to standard MHA if not specified
    n_layers = config.get("n_layers", 12)

    # IMPORTANT: Ensure head_dim is even for RoPE compatibility
    head_dim = d_model // n_heads
    if head_dim % 2 != 0:
        # Adjust d_model to ensure head_dim is even
        logger.warning(f"Adjusting d_model from {d_model} to {n_heads * (head_dim + 1)} to ensure even head_dim")
        d_model = n_heads * (head_dim + 1)
    
    logger.info(f"Building OLMo model with {n_heads} query heads and {n_kv_heads} KV heads")
    
    # Set up model configuration
    model_config = TransformerConfig.olmo2_190M(
        vocab_size=vocab_size,
        dtype=dtype,
        init_method=InitMethod.normal,
        # Support for GQA: n_kv_heads can be smaller than n_heads
        n_heads=n_heads,
        n_kv_heads=n_kv_heads
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
    
    # Check first and last blocks to account for layer-wise scaling
    first_block = model.blocks['0']
    last_block_idx = str(len(model.blocks) - 1)
    last_block = model.blocks[last_block_idx]

    # Examine attention mechanisms
    first_attention = first_block.attention
    last_attention = last_block.attention
    
    # Get actual head counts to verify GQA is active
    if hasattr(first_attention, 'w_q') and hasattr(first_attention, 'w_k'):
        # Derive head dimensions from the linear layer dimensions
        head_dim = first_attention.w_q.in_features // expected_n_heads
        
        first_q_heads = first_attention.w_q.out_features // head_dim
        first_kv_heads = first_attention.w_k.out_features // head_dim
        
        last_q_heads = last_attention.w_q.out_features // head_dim
        last_kv_heads = last_attention.w_k.out_features // head_dim
        
        # Check GQA is active (q heads > kv heads) in both first and last layer
        first_is_gqa = first_q_heads > first_kv_heads and first_kv_heads >= 2
        last_is_gqa = last_q_heads > last_kv_heads and last_kv_heads >= 2
        
        logger.info(f"First layer: {first_q_heads} query heads, {first_kv_heads} KV heads")
        logger.info(f"Last layer: {last_q_heads} query heads, {last_kv_heads} KV heads")
        
        if first_is_gqa and last_is_gqa:
            logger.info("GQA is active throughout the model with Layer-Wise Scaling")
            return True
        else:
            logger.error(f"GQA configuration invalid: first layer {first_is_gqa}, last layer {last_is_gqa}")
            return False
    else:
        logger.error("Cannot verify GQA: attention architecture not as expected")
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
        fused=torch.cuda.is_available(),
        group_overrides=[
            # Use parameter lists with proper patterns for normalization layers
            OptimGroupOverride(params=[
                "lm_head.norm.weight", 
                "blocks.*.attention_norm.weight", 
                "blocks.*.feed_forward_norm.weight", 
                "blocks.*.attention.q_norm.weight",
                "blocks.*.attention.k_norm.weight"
            ], opts=dict(weight_decay=0.0)),
            
            # For embeddings - use direct parameter name
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
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
        rank_microbatch_size=batch_size * sequence_length,
        max_sequence_length=sequence_length,
        optim=optimizer_config,
        compile_model=compile_model
    )
    
    # Build train module
    train_module = train_module_config.build(model=model, device=device)
    
    return train_module