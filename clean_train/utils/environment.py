"""
Environment setup module for OLMo training.
"""

import os
WANDB_API_KEY = "8e07447fa3d6269331f7ecd0b27f8518c2a65855"


def setup_environment():
    """
    Set up environment variables for OLMo training.
    """
    # Disable tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set up cache directories
    cache_dir = os.path.join(os.getcwd(), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables for cache directories
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
    os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface")
    
    # Create cache directories
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # Set up Weights & Biases for detailed logging 
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    
    print("Environment setup complete") 