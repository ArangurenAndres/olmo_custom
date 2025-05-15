#!/usr/bin/env python3
"""
Main entry point for training OLMo models with Layer-Wise Scaling (LWS) and 
Group Query Attention (GQA) using Fully Sharded Data Parallel (FSDP).
"""

import os
import sys
import argparse
import yaml
import time
import logging
import torch
from pathlib import Path

# Import from OLMo core
from olmo_core.distributed.utils import (
    is_distributed,
    get_rank,
    get_world_size,
    barrier
)
from olmo_core.utils import seed_all

# Import from project modules
from src.data import (
    create_dataset_config,
    create_dataloader_config,
    build_dataloader,
    get_tokenizer_config,
    load_tokenizer
)
from src.model import (
    build_olmo_model, 
    build_optimizer_config, 
    build_train_module,
    apply_layer_wise_scaling
)
from src.training import (
    create_trainer,
    train_model,
    InferenceCallback,
    CheckpointCallback,
    WandBCallback
)
from src.training.fsdp_utils import (
    setup_fsdp_environment,
    wrap_model_with_fsdp,
    get_mixed_precision_policy
)
from src.utils import (
    setup_logging,
    get_logger,
    log_dict,
    setup_device,
    init_random_seeds,
    is_main_process
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train OLMo with Layer-Wise Scaling and Group Query Attention"
    )
    
    # Configuration files
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/base_config.yaml", 
        help="Path to base configuration file"
    )
    parser.add_argument(
        "--fsdp-config", 
        type=str, 
        default="config/fsdp_config.yaml", 
        help="Path to FSDP configuration file"
    )
    
    # Distributed training settings
    parser.add_argument(
        "--use-fsdp", 
        action="store_true", 
        help="Use Fully Sharded Data Parallel"
    )
    parser.add_argument(
        "--distributed-backend", 
        type=str, 
        choices=["nccl", "gloo"], 
        default="nccl", 
        help="Distributed backend"
    )
    parser.add_argument(
        "--world-size", 
        type=int, 
        default=1, 
        help="World size for distributed training"
    )
    parser.add_argument(
        "--rank", 
        type=int, 
        default=0, 
        help="Process rank for distributed training"
    )
    parser.add_argument(
        "--master-addr", 
        type=str, 
        default="localhost", 
        help="Master address for distributed training"
    )
    parser.add_argument(
        "--master-port", 
        type=str, 
        default="29500", 
        help="Master port for distributed training"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None, 
        help="Directory to save outputs, overrides config"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--wandb", 
        action="store_true", 
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--inference-prompt", 
        type=str, 
        default=None, 
        help="Text prompt for inference during training"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def load_config(path: str):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base_config, fsdp_config=None, args=None):
    """Merge configurations from different sources with priority."""
    config = base_config.copy()
    
    # Add FSDP configuration if provided
    if fsdp_config:
        config["fsdp"] = fsdp_config
    
    # Override with command-line arguments
    if args:
        # Update output directory if specified
        if args.output_dir:
            config["output_dir"] = args.output_dir
        
        # Update Weights & Biases setting
        if args.wandb:
            config["use_wandb"] = True
        
        # Update inference prompt if specified
        if args.inference_prompt:
            config["inference_prompt"] = args.inference_prompt
            
        # Update debugging settings
        if args.debug:
            config["debug"] = True
            config["verbose_scaling"] = True
            config["log_level"] = "DEBUG"
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configurations
    try:
        base_config = load_config(args.config)
        fsdp_config = load_config(args.fsdp_config) if args.fsdp_config and args.use_fsdp else None
        config = merge_configs(base_config, fsdp_config, args)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Set up directories
    output_dir = config.get("output_dir", "./output")
    data_dir = config.get("data_dir", "./data")
    log_dir = os.path.join(output_dir, "logs")
    save_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize distributed environment if using FSDP
    if args.use_fsdp:
        setup_fsdp_environment(
            rank=args.rank,
            world_size=args.world_size,
            master_addr=args.master_addr,
            master_port=args.master_port,
            backend=args.distributed_backend
        )
        distributed = True
    else:
        distributed = False
    
    # Set up logging
    log_file = os.path.join(log_dir, "training.log")
    log_level = getattr(logging, config.get("log_level", "INFO"))
    logger = setup_logging(
        log_file=log_file,
        log_level=log_level,
        main_rank_only=True,
        console=True
    )
    
    # Log configuration
    if is_main_process():
        log_dict(config, logger, prefix="Configuration:")
        
    # Set random seeds
    seed = config.get("seed", 42)
    init_random_seeds(seed)
    
    # Set up device
    device, mixed_precision = setup_device(
        use_cuda=torch.cuda.is_available(),
        use_cuda_amp=config.get("mixed_precision", True)
    )
    
    # Log device information
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
    
    # Build dataloader
    dataloader, tokenizer_config, token_file = build_dataloader(
        config=config,
        distributed=distributed
    )
    
    # Log data information
    logger.info(f"Data loaded from: {token_file}")
    logger.info(f"Sequence length: {config.get('sequence_length', 1024)}")
    logger.info(f"Batch size: {config.get('batch_size', 8)}")
    
    # Build model
    model = build_olmo_model(
        config=config,
        device=device
    )
    
    # Verify model structure
    logger.info(f"Built model with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Number of layers: {len(model.blocks)}")
    logger.info(f"Attention heads: {config.get('n_heads', 12)}")
    logger.info(f"KV heads: {config.get('n_kv_heads', config.get('n_heads', 12))}")
    
    # Apply Layer-Wise Scaling
    if any(k in config for k in ["fnn_scalars", "qkv_scalars", "output_scalars"]):
        logger.info("Applying Layer-Wise Scaling")
        model = apply_layer_wise_scaling(
            model=model,
            ffn_scalars=config.get("fnn_scalars"),
            qkv_scalars=config.get("qkv_scalars"),
            output_scalars=config.get("output_scalars"),
            verbose=config.get("verbose_scaling", False)
        )
    
    # Wrap model with FSDP if needed
    if args.use_fsdp:
        logger.info("Wrapping model with FSDP")
        model = wrap_model_with_fsdp(
            model=model,
            mixed_precision=config.get("mixed_precision", True),
            shard_strategy=config.get("fsdp", {}).get("shard_strategy", "FULL_SHARD"),
            cpu_offload=config.get("fsdp", {}).get("cpu_offload", False)
        )
    
    # Build training module
    train_module = build_train_module(
        model=model,
        config=config,
        device=device
    )
    
    # Define callbacks
    callbacks = []
    
    # Add wandb callback if enabled
    if config.get("use_wandb", False) and is_main_process():
        wandb_callback = WandBCallback(
            project_name=config.get("wandb_project", "olmo_training"),
            config=config
        )
        callbacks.append(wandb_callback)
    
    # Add inference callback if prompt is provided
    if config.get("inference_prompt") and is_main_process():
        inference_callback = InferenceCallback(
            train_module=train_module,
            tokenizer_name=config.get("tokenizer_name", "allenai/gpt-neox-olmo-dolma-v1_5"),
            prompt=config.get("inference_prompt"),
            interval=config.get("inference_interval", 50),
            max_new_tokens=config.get("max_new_tokens", 50),
            log_to_file=os.path.join(log_dir, "generations.txt")
        )
        callbacks.append(inference_callback)
    
    # Update configuration for training
    train_config = config.copy()
    train_config.update({
        "save_dir": save_dir,
        "log_dir": log_dir,
        "data_dir": data_dir
    })
    
    # Run training
    logger.info(f"Starting training for {config.get('steps', 1000)} steps")
    start_time = time.time()
    
    try:
        trained_model = train_model(
            train_module=train_module,
            dataloader=dataloader,
            config=train_config,
            callbacks=callbacks,
            device=device
        )
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save the final model in Hugging Face format if specified
        if config.get("save_hf_model", False) and is_main_process():
            from src.utils.checkpointing import convert_checkpoint_to_hf
            
            hf_path = os.path.join(output_dir, "hf_model")
            logger.info(f"Converting checkpoint to Hugging Face format: {hf_path}")
            
            convert_checkpoint_to_hf(
                checkpoint_path=os.path.join(save_dir, "final_model.pt"),
                output_dir=hf_path
            )
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1
    
    # Synchronize processes before exit
    if distributed:
        barrier()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())