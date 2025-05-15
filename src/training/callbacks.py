"""
Custom callbacks for OLMo training.
"""

import os
import logging
import torch
import time
from typing import Dict, Any, Optional, List, Union

from olmo_core.train.callbacks import Callback
from olmo_core.distributed.utils import is_distributed, get_rank, barrier

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class InferenceCallback(Callback):
    """
    Run text generation during training to monitor model progress.
    """
    
    def __init__(
        self,
        train_module: Any,
        tokenizer_name: str = "allenai/gpt-neox-olmo-dolma-v1_5",
        prompt: str = "The universe is",
        interval: int = 50,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        log_to_file: Optional[str] = None
    ):
        """
        Initialize the inference callback.
        
        Args:
            train_module: OLMo train module
            tokenizer_name: Name of the tokenizer to use
            prompt: Prompt to use for generation
            interval: How often to run inference (in steps)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            log_to_file: Optional file path to log generations
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for InferenceCallback")
        
        self.train_module = train_module
        self.tokenizer_name = tokenizer_name
        self.prompt = prompt
        self.interval = interval
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.log_to_file = log_to_file
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def pre_train(self, trainer):
        """Run inference before training starts."""
        self.trainer = trainer
        self.run_inference(trainer, step=0)
    
    def post_step(self):
        """Run inference periodically during training."""
        # step = trainer.global_step
        # if step % self.interval == 0 and step > 0:
        #     self.run_inference(trainer, step=step)
        if self.trainer.global_step % self.interval == 0 and self.trainer.global_step > 0:
            self.run_inference(self.trainer.global_step)
    
    def run_inference(self, trainer, step: int):
        """
        Run text generation with the current model.
        
        Args:
            trainer: OLMo trainer
            step: Current training step
        """
        # Only run on rank 0 if in distributed mode
        if is_distributed() and get_rank() != 0:
            return
            
        # Get the model from train_module
        model = self.train_module.model
        
        # Generate text
        try:
            # Prepare input
            input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt").to(trainer.device)
            
            # Set model to eval mode, generate text, then restore training mode
            model.eval()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            model.train()
            
            # Decode generated text
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Log the generation
            logger.info(f"\nStep {step}, Generated text: {generated_text}")
            
            # Log to file if specified
            if self.log_to_file:
                with open(self.log_to_file, 'a') as f:
                    f.write(f"Step {step} | {generated_text}\n\n")
            
            # Log to wandb if available
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"generated_text": generated_text, "step": step})
                
        except Exception as e:
            logger.error(f"Error during inference: {e}")

class CheckpointCallback(Callback):
    """
    Enhanced checkpoint callback with FSDP support.
    """
    
    def __init__(
        self,
        save_dir: str,
        interval: int = 100,
        save_optimizer: bool = True,
        keep_last_k: int = 3
    ):
        """
        Initialize the checkpoint callback.
        
        Args:
            save_dir: Directory to save checkpoints
            interval: How often to save checkpoints (in steps)
            save_optimizer: Whether to save optimizer state
            keep_last_k: Keep only the last k checkpoints
        """
        self.save_dir = save_dir
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.keep_last_k = keep_last_k
        self.saved_checkpoints = []
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def post_step(self, trainer):
        """Save checkpoints periodically during training."""
        step = trainer.global_step
        if step % self.interval == 0 and step > 0:
            self.save_checkpoint(trainer, step)
    
    def save_checkpoint(self, trainer, step: int):
        """
        Save a checkpoint of the model and optimizer.
        
        Args:
            trainer: OLMo trainer
            step: Current training step
        """
        # Synchronize processes in distributed setting
        if is_distributed():
            barrier()
            
        # Only save on rank 0
        if is_distributed() and get_rank() != 0:
            return
            
        # Create checkpoint path
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{step}.pt")
        
        # Create checkpoint dictionary
        checkpoint = {
            'step': step,
            'model_state_dict': trainer.train_module.model.state_dict(),
            'loss': trainer.train_state.loss
        }
        
        # Add optimizer state if requested
        if self.save_optimizer and hasattr(trainer, 'optimizer'):
            checkpoint['optimizer_state_dict'] = trainer.optimizer.state_dict()
        
        # Save checkpoint
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Keep track of saved checkpoints
            self.saved_checkpoints.append(checkpoint_path)
            
            # Remove old checkpoints if needed
            if self.keep_last_k > 0 and len(self.saved_checkpoints) > self.keep_last_k:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
                    
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def post_train(self, trainer):
        """Save final checkpoint after training."""
        step = trainer.global_step
        
        # Save with special name to indicate it's the final checkpoint
        if is_distributed() and get_rank() != 0:
            return
            
        final_checkpoint_path = os.path.join(self.save_dir, "final_model.pt")
        
        # Create checkpoint dictionary
        checkpoint = {
            'step': step,
            'model_state_dict': trainer.train_module.model.state_dict(),
            'loss': trainer.train_state.loss
        }
        
        # Add optimizer state if requested
        if self.save_optimizer and hasattr(trainer, 'optimizer'):
            checkpoint['optimizer_state_dict'] = trainer.optimizer.state_dict()
        
        # Save checkpoint
        try:
            torch.save(checkpoint, final_checkpoint_path)
            logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving final checkpoint: {e}")

class WandBCallback(Callback):
    """
    Weights & Biases integration for experiment tracking.
    """
    
    def __init__(
        self,
        project_name: str = "olmo_training",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the WandB callback.
        
        Args:
            project_name: WandB project name
            config: Configuration to log
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for WandBCallback")
        
        self.project_name = project_name
        self.config = config or {}
        self.initialized = False
        
        # Set default metrics to track
        self.tracked_metrics = [
            "loss"
        ]
    
    def pre_train(self, trainer):
        """Initialize WandB run before training."""
        # Only initialize on main process
        if is_distributed() and get_rank() != 0:
            return
            
        # Initialize wandb
        if not self.initialized:
            wandb.init(project=self.project_name, config=self.config)
            self.initialized = True
            
            # Log model details if available
            if hasattr(trainer.train_module, 'model'):
                model = trainer.train_module.model
                wandb.config.update({
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
                })
    
    def post_step(self, trainer):
        """Log metrics after each step."""
        if not self.initialized or (is_distributed() and get_rank() != 0):
            return
            
        # Extract and log metrics
        metrics = {}
        
        # Log loss
        if hasattr(trainer.train_state, 'loss'):
            metrics["loss"] = trainer.train_state.loss
            
        # Log learning rate if available
        if hasattr(trainer, 'optimizer') and hasattr(trainer.optimizer, 'param_groups'):
            metrics["learning_rate"] = trainer.optimizer.param_groups[0]['lr']
        
        # Log step
        metrics["step"] = trainer.global_step
        
        # Send to wandb
        if metrics:
            wandb.log(metrics)
    
    def post_train(self, trainer):
        """Finalize WandB run after training."""
        if not self.initialized or (is_distributed() and get_rank() != 0):
            return
        
        wandb.finish()