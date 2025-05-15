"""
Tokenization utilities for OLMo training.
"""

import os
import logging
from typing import Optional, Union, Dict, Any

from transformers import AutoTokenizer
from olmo_core.data import TokenizerConfig

logger = logging.getLogger(__name__)

def get_tokenizer_config(
    cache_dir: Optional[str] = None,
    tokenizer_name: str = "allenai/gpt-neox-olmo-dolma-v1_5"
) -> TokenizerConfig:
    """
    Create a tokenizer configuration for OLMo.
    
    Args:
        cache_dir: Directory to cache tokenizer files
        tokenizer_name: Name of the tokenizer to use
        
    Returns:
        TokenizerConfig: Configuration for the OLMo tokenizer
    """
    if tokenizer_name == "allenai/gpt-neox-olmo-dolma-v1_5":
        # Use OLMo's built-in configuration for this tokenizer
        config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
        if cache_dir is not None:
            config.cache_dir = cache_dir
        return config
    else:
        # For other tokenizers, create a custom config
        return TokenizerConfig(
            name=tokenizer_name,
            cache_dir=cache_dir
        )

def load_tokenizer(
    tokenizer_name: str = "allenai/gpt-neox-olmo-dolma-v1_5",
    cache_dir: Optional[str] = None,
    additional_special_tokens: Optional[list] = None,
    **kwargs
) -> AutoTokenizer:
    """
    Load a tokenizer using the Hugging Face transformers library.
    
    Args:
        tokenizer_name: Name of the tokenizer to load
        cache_dir: Directory to cache tokenizer files
        additional_special_tokens: Additional special tokens to add
        **kwargs: Additional arguments to pass to AutoTokenizer.from_pretrained()
        
    Returns:
        AutoTokenizer: The loaded tokenizer
    """
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=cache_dir,
        **kwargs
    )
    
    # Add additional special tokens if provided
    if additional_special_tokens:
        special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {len(additional_special_tokens)} special tokens")
    
    return tokenizer

def get_vocab_size(tokenizer_config: TokenizerConfig) -> int:
    """
    Get the padded vocab size from a tokenizer configuration.
    
    Args:
        tokenizer_config: Tokenizer configuration
        
    Returns:
        int: Padded vocabulary size
    """
    return tokenizer_config.padded_vocab_size()