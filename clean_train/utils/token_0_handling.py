"""
Module for handling special token ID 0 in the model.

IP ADRESS PROBLEM solved by this function.
"""

import torch

def apply_special_token_handling(model):
    """
    Apply special handling for token ID 0 - set embedding to zeros 
    and logit bias to large negative.
    
    Args:
        model: The OLMo model
    """
    with torch.no_grad():
        # Set embedding for token ID 0 to zeros
        model.embeddings.weight[0].zero_()
        
        # Set output layer bias for token ID 0 to large negative value
        if hasattr(model.lm_head, 'w_out') and model.lm_head.w_out.bias is not None:
            model.lm_head.w_out.bias[0] = -100.0
    
    print("Special handling for token ID 0 applied - zeroed embeddings and set negative bias") 