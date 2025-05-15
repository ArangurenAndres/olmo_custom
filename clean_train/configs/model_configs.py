"""
Model configuration module for OLMo models.
"""

import sys
import os
from olmo_core.nn.transformer import (
    TransformerConfig, 
    InitMethod, 
    TransformerActivationCheckpointingMode
)
from olmo_core.config import DType
from olmo_core.train.train_module.transformer.config import (
    TransformerActivationCheckpointingConfig
)


def get_model_config(tokenizer_config):
    """
    Create configuration for the OLMo model.
    
    Args:
        tokenizer_config: The tokenizer configuration
        
    Returns:
        TransformerConfig: Model configuration
    """
    # Set up model configuration
    model_config = TransformerConfig.olmo2_190M(
        vocab_size=tokenizer_config.padded_vocab_size(),
        dtype=DType.bfloat16
    )

    # Set activation checkpointing
    ac_config = TransformerActivationCheckpointingConfig(
        mode=TransformerActivationCheckpointingMode.full
    )
    model_config.activation_checkpointing = ac_config
    
    return model_config 

