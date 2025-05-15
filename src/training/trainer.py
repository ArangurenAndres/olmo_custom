"""
Enhanced trainer for OLMo with FSDP support.
"""

import os
import logging
import torch
from typing import Dict, Any, Optional, List, Union

from olmo_core.train import TrainerConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from olmo_core.distributed.utils import (
    is_distributed,
    get_rank,
    get_world_size,
    barrier
)

from .callbacks import InferenceCallback, CheckpointCallback, WandBCallback

logger = logging.getLogger(__name__)

def create_trainer(
    train_module: Any,  # TransformerTrainModule from olmo_core
    dataloader: Any,    # DataLoader from olmo_core
    config: Dict[str, Any],
    callbacks: Optional[List[Any]] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Trainer:
    """
    Create an OLMo trainer with proper configuration.
    
    Args:
        train_module: OLMo train module (has model, optimizer, etc.)
        dataloader: OLMo data loader
        config: Configuration dictionary
        callbacks: Optional list of additional callbacks
        device: Device to train on
        
    Returns:
        trainer: Configured OLMo trainer
    """
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device
    
    # Set up directories
    save_dir = config.get("save_dir", os.path.join(config.get("data_dir", "./data"), "checkpoints"))
    work_dir = config.get("work_dir", os.path.join(config.get("data_dir", "./data"), "trainer_work_dir"))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    
    # Configure duration
    steps = config.get("steps", 1000)
    duration = Duration.from_steps(steps)
    
    # Initialize trainer config
    trainer_config = TrainerConfig(
        save_folder=save_dir,
        save_overwrite=config.get("save_overwrite", True),
        work_dir=work_dir,
        metrics_collect_interval=config.get("metrics_interval", 10),
        cancel_check_interval=config.get("cancel_interval", 5),
        max_duration=duration,
        device=device_str,
    )
    
    # Add default callbacks
    if not is_distributed() or get_rank() == 0:
        # Only add inference and wandb callbacks on rank 0
        
        # Add inference callback if specified
        if config.get("inference_prompt"):
            inference_interval = config.get("inference_interval", 50)
            inference_cb = InferenceCallback(
                train_module=train_module,
                tokenizer_name=config.get("tokenizer_name", "allenai/gpt-neox-olmo-dolma-v1_5"),
                prompt=config.get("inference_prompt"),
                interval=inference_interval,
                max_new_tokens=config.get("max_new_tokens", 50)
            )
            trainer_config = trainer_config.with_callback("inference", inference_cb)
            
        # Add WandB callback if specified
        if config.get("use_wandb", False):
            wandb_cb = WandBCallback(
                project_name=config.get("wandb_project", "olmo_training"),
                config=config
            )
            trainer_config = trainer_config.with_callback("wandb", wandb_cb)
    
    # Add checkpoint callback on all ranks
    ckpt_interval = config.get("checkpoint_interval", 100)
    ckpt_cb = CheckpointCallback(
        save_dir=save_dir,
        interval=ckpt_interval,
        save_optimizer=config.get("save_optimizer", True),
        keep_last_k=config.get("keep_last_k", 3)
    )
    trainer_config = trainer_config.with_callback("checkpoint", ckpt_cb)
    
    # Add any additional user-provided callbacks
    if callbacks:
        for i, callback in enumerate(callbacks):
            trainer_config = trainer_config.with_callback(f"user_callback_{i}", callback)
    
    # Build and return the trainer
    trainer = trainer_config.build(train_module=train_module, data_loader=dataloader)
    return trainer

def train_model(
    train_module: Any,
    dataloader: Any,
    config: Dict[str, Any],
    callbacks: Optional[List[Any]] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Any:
    """
    Run the complete training process.
    
    Args:
        train_module: OLMo train module
        dataloader: OLMo data loader
        config: Configuration dictionary
        callbacks: Optional additional callbacks
        device: Device to train on
        
    Returns:
        model: Trained model
    """
    # Create trainer
    trainer = create_trainer(
        train_module=train_module,
        dataloader=dataloader,
        config=config,
        callbacks=callbacks,
        device=device
    )
    
    # Log training start
    logger.info(f"Starting training for {config.get('steps', 1000)} steps")
    
    # Run training
    trainer.fit()
    
    # Ensure all processes are synchronized after training
    if is_distributed():
        barrier()
    
    # Log completion
    logger.info("Training complete")
    
    # Return the trained model
    return train_module.model