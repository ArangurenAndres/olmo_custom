"""
Main training script for OLMo models.

This script coordinates the training process by importing and using modules for:
1. Configuration parsing
2. Dataset loading from pre-prepared data
3. Model building
4. Training execution

Total tokens trained = batch_size * steps * sequence_length 
- batch size is dependent on the GPU memory (higher always better)
- steps is dependent on the number of tokens to train
- sequence length is fixed at 1024

"""

import os
import torch
import wandb

from configs.argparser import parse_args_train
from configs.model_configs import get_model_config
from configs.optimizer_configs import get_optimizer_config
from data_utils.load_dataset import load_prepared_dataset
from data_utils.dataloader import build_dataloader
from training.trainer import run_training
from utils.paths import setup_train_directories
from utils.environment import setup_environment
from utils.token_0_handling import apply_special_token_handling


def main():
    # Parse command line arguments
    args = parse_args_train()
    
    # Setup environment variables and directories
    setup_environment()
    data_dir, save_folder = setup_train_directories(args)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    
    # Load pre-prepared dataset + tokenizer config 
    tokenizer_config, dataset = load_prepared_dataset(
        args, data_dir, device
    )
    
    # Build data loader
    data_loader = build_dataloader(dataset, args.batch_size, args.sequence_length)
    
    model_config = get_model_config(tokenizer_config)
        
    model = model_config.build(init_device=device)

    
    # IP ADRESS PROBLEM SOLVED with setting the token ID 0 to zeros and logit bias to large negative
    apply_special_token_handling(model)
    
    # Configure optimizer
    optim_config = get_optimizer_config()
    
    run_training(
        model, optim_config, data_loader, save_folder, 
        args, tokenizer_config, device, args.sequence_length
    )
    
    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main() 