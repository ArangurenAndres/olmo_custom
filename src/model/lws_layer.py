"""
Implementation of Layer-Wise Scaling for OLMo Transformer models.

Layer-Wise Scaling (LWS) applies different scaling factors to different layers of the model
based on their depth. This results in gradient flow improvement and better training dynamics.
"""

import torch
import logging
from typing import Dict, List, Union, Optional

logger = logging.getLogger(__name__)

def compute_layer_scaling_factor(
    layer_idx: int,
    num_layers: int,
    min_scale: float,
    max_scale: float
) -> float:
    """
    Compute a scaling factor for a layer based on its position.
    
    Args:
        layer_idx: Index of the layer
        num_layers: Total number of layers
        min_scale: Minimum scaling factor
        max_scale: Maximum scaling factor
        
    Returns:
        float: Scaling factor for the layer
    """
    if num_layers <= 1:
        return max_scale
        
    # Linear interpolation between min_scale and max_scale
    return min_scale + (max_scale - min_scale) * (layer_idx / (num_layers - 1))

def apply_layer_wise_scaling(
    model: torch.nn.Module,
    ffn_scalars: Optional[List[float]] = None,
    qkv_scalars: Optional[List[float]] = None,
    output_scalars: Optional[List[float]] = None,
    verbose: bool = False
) -> torch.nn.Module:
    """
    Apply Layer-Wise Scaling to transformer blocks.
    
    Args:
        model: OLMo model
        ffn_scalars: List of [min, max] scaling factors for feed-forward networks
        qkv_scalars: List of [min, max] scaling factors for query-key-value projections
        output_scalars: List of [min, max] scaling factors for output projections
        verbose: Whether to print detailed scaling information
        
    Returns:
        model: Model with Layer-Wise Scaling applied
    """
    # Get total number of layers
    num_layers = len(model.blocks) if hasattr(model, 'blocks') else 0
    
    if num_layers == 0:
        logger.warning("No transformer blocks found in model, skipping layer-wise scaling")
        return model
        
    if verbose:
        logger.info(f"Applying layer-wise scaling to {num_layers} transformer blocks")
    
    # Apply scaling to each transformer block
    for block_idx_str, block in model.blocks.items():
        block_idx = int(block_idx_str)
        
        # Apply FFN scaling
        if ffn_scalars and len(ffn_scalars) >= 2:
            ffn_scale = compute_layer_scaling_factor(
                block_idx, num_layers, ffn_scalars[0], ffn_scalars[1]
            )
            
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'w_gate'):
                with torch.no_grad():
                    block.ffn.w_gate.weight.data *= ffn_scale
                    
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'w_up'):
                with torch.no_grad():
                    block.ffn.w_up.weight.data *= ffn_scale
                    
            if verbose:
                logger.info(f"Applied FFN scaling factor {ffn_scale:.4f} to block {block_idx}")
        
        # Apply QKV scaling
        if qkv_scalars and len(qkv_scalars) >= 2:
            qkv_scale = compute_layer_scaling_factor(
                block_idx, num_layers, qkv_scalars[0], qkv_scalars[1]
            )
            
            if hasattr(block, 'attention') and hasattr(block.attention, 'wqkv'):
                with torch.no_grad():
                    block.attention.wqkv.weight.data *= qkv_scale
                    
            if verbose:
                logger.info(f"Applied QKV scaling factor {qkv_scale:.4f} to block {block_idx}")
        
        # Apply output projection scaling
        if output_scalars and len(output_scalars) >= 2:
            output_scale = compute_layer_scaling_factor(
                block_idx, num_layers, output_scalars[0], output_scalars[1]
            )
            
            if hasattr(block, 'attention') and hasattr(block.attention, 'wo'):
                with torch.no_grad():
                    block.attention.wo.weight.data *= output_scale
                    
            if verbose:
                logger.info(f"Applied output scaling factor {output_scale:.4f} to block {block_idx}")
    
    return model

def verify_lws_applied(
    model: torch.nn.Module,
    ffn_scalars: Optional[List[float]] = None,
    qkv_scalars: Optional[List[float]] = None
) -> bool:
    """
    Verify that Layer-Wise Scaling was applied correctly.
    
    Args:
        model: OLMo model
        ffn_scalars: List of [min, max] scaling factors for feed-forward networks
        qkv_scalars: List of [min, max] scaling factors for query-key-value projections
        
    Returns:
        bool: True if scaling was applied correctly
    """
    # This is a simplified verification that just checks a few layers
    # A complete verification would compare to an unscaled model
    
    if not hasattr(model, 'blocks') or len(model.blocks) < 2:
        return False
    
    # Check FFN scaling by comparing first and last layer norms
    if ffn_scalars and len(ffn_scalars) >= 2:
        first_block = model.blocks['0']
        last_block = model.blocks[str(len(model.blocks) - 1)]
        
        if (hasattr(first_block, 'ffn') and hasattr(last_block, 'ffn') and
            hasattr(first_block.ffn, 'w_gate') and hasattr(last_block.ffn, 'w_gate')):
            first_norm = first_block.ffn.w_gate.weight.data.norm().item()
            last_norm = last_block.ffn.w_gate.weight.data.norm().item()
            
            # If scaling is applied correctly, the last layer should have higher norm
            ratio = last_norm / first_norm
            expected_ratio = ffn_scalars[1] / ffn_scalars[0]
            
            # Check if ratio is close to expected (within 10%)
            if abs(ratio - expected_ratio) / expected_ratio > 0.1:
                logger.warning(f"FFN scaling might not have been applied correctly. "
                             f"Expected ratio: {expected_ratio:.4f}, Got: {ratio:.4f}")
                return False
    
    return True