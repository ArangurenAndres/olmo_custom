"""
Path management module for OLMo training.
"""

import os
import time


### SETING UP DATA DIRECTORIES FOR TRAINING train.py ### 
def setup_train_directories(args):
    """
    Set up directories for data and model outputs.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (data_dir, save_folder)
    """
    # Set up data directories
    data_dir = args.data_dir or f"/home/{os.environ.get('USER')}/OLMo-custom/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Set up cache directories
    cache_dir = os.path.join(data_dir, "huggingface_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Setup output directory for checkpoints
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    save_folder = os.path.join(data_dir, f"olmo_training_output_{timestamp}")
    os.makedirs(save_folder, exist_ok=True)
    
    return data_dir, save_folder 


### SETING UP DATA DIRECTORIES FOR DATA PREPARATION data_prep.py ### 
def setup_data_directories(args):
    """
    Set up directories for data preparation.
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Data directory path
    """
    # Set up data directories
    data_dir = args.data_dir or f"/home/{os.environ.get('USER')}/OLMo-custom/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Set up cache directories
    cache_dir = os.path.join(data_dir, "huggingface_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Setup wiki data directory
    wiki_data_dir = os.path.join(data_dir, "wiki_data")
    os.makedirs(wiki_data_dir, exist_ok=True)
    
    return data_dir 