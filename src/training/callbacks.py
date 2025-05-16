"""
Custom callbacks for OLMo training.
"""

import os
import logging
import torch
import time
from typing import Dict, Any, Optional, List, Union
import math

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
        self.trainer = None
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def pre_train(self):  # IMPORTANT: No trainer parameter here
        """Run inference before training starts."""
        # self.trainer is already set by OLMo's Trainer in __post_init__
        if self.tokenizer is None:
            try:
                from src.data import load_tokenizer
                self.tokenizer = load_tokenizer(self.tokenizer_name)
            except ImportError:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.run_inference(step=0)  # Changed to just pass step
    
    def post_step(self):  # No trainer parameter here either
        """Run inference periodically during training."""
        if self.trainer.global_step % self.interval == 0 and self.trainer.global_step > 0:
            self.run_inference(step=self.trainer.global_step)
    
    def run_inference(self, step: int):  # Updated to only take step
        """
        Run text generation with the current model.
        
        Args:
            step: Current training step
        """
        # Only run on rank 0 if in distributed mode
        if is_distributed() and get_rank() != 0:
            return
            
        # Get the model from train_module
        model = self.train_module.model
        
        try:
            # Only sync once at the beginning for tokenization
            with torch.no_grad():
                model.eval()
                
                # Create tensor on CPU first then transfer to GPU (reduces sync points)
                tokens = self.tokenizer.encode(self.prompt)
                seq_len = self.train_module.max_sequence_length
                if len(tokens) > seq_len:
                    tokens = tokens[:seq_len]
                    
                input_ids = torch.tensor([tokens], device="cpu").to(model.device, non_blocking=True)
                
                # Pre-allocate tensor for generated tokens to avoid repeated creation
                generated = tokens.copy()
                max_length = len(tokens) + self.max_new_tokens
                
                # Run initial prediction
                outputs = model(input_ids)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                # Main generation loop
                for _ in range(self.max_new_tokens):
                    # Extract next token logits
                    next_token_logits = logits[0, -1, :].clone() / self.temperature
                    
                    # Create a mask for token 0 instead of setting to -inf
                    next_token_logits[0] = torch.finfo(next_token_logits.dtype).min
                    
                    # Sample next token (this still requires synchronization for .item())
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    
                    # Check for EOS
                    if next_token == self.tokenizer.eos_token_id:
                        break
                        
                    # Add token to generated sequence
                    generated.append(next_token)
                    
                    # Manage sequence length
                    if len(generated) > seq_len:
                        generated = generated[-seq_len:]
                    
                    # Update input_ids efficiently (avoid creating new tensors repeatedly)
                    input_ids = torch.tensor([generated], device="cpu").to(model.device, non_blocking=True)
                    
                    # Get next prediction
                    outputs = model(input_ids)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                
                # Only one sync point at the end for generation result
                generated_text = self.tokenizer.decode(generated, skip_special_tokens=True)
                
                model.train()
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            # Log the generation
            logger.info(f"\nStep {step}, Generated text: {generated_text}")
            
            # Log to file if specified
            if self.log_to_file:
                with open(self.log_to_file, 'a') as f:
                    f.write(f"Step {step} | {generated_text}\n\n")
            
            # Log to wandb if available
            if WANDB_AVAILABLE and wandb.run is not None:
                try:
                    # Create a proper WandB Table for better visualization
                    table = wandb.Table(columns=["step", "prompt", "generated_text"])
                    table.add_data(step, self.prompt, generated_text)
                    
                    wandb.log({
                        "generation_samples": table,
                        "generated_text": generated_text,  # Also log as string for compatibility
                        "step": step
                    })
                except Exception as e:
                    logger.warning(f"Error logging to WandB: {e}")
                    # Fallback to simple logging
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
        """Initialize the checkpoint callback."""
        self.save_dir = save_dir
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.keep_last_k = keep_last_k
        self.saved_checkpoints = []
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def post_step(self):  # Remove the trainer parameter
        """Save checkpoints periodically during training."""
        step = self.trainer.global_step
        if step % self.interval == 0 and step > 0:
            self.save_checkpoint(step)
    
    def save_checkpoint(self, step: int):  # Updated to only take step
        """
        Save a checkpoint of the model and optimizer.
        
        Args:
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
            'model_state_dict': self.trainer.train_module.model.state_dict(),
        }
        
        # Add loss if available
        try:
            if hasattr(self.trainer, 'train_state') and hasattr(self.trainer.train_state, 'loss'):
                checkpoint['loss'] = self.trainer.train_state.loss
        except Exception as e:
            logger.warning(f"Could not save loss in checkpoint: {e}")
        
        # Add optimizer state if requested
        if self.save_optimizer and hasattr(self.trainer, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.trainer.optimizer.state_dict()
        
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
    
    def post_train(self):  # Remove the trainer parameter
        """Save final checkpoint after training."""
        step = self.trainer.global_step
        
        # Save with special name to indicate it's the final checkpoint
        if is_distributed() and get_rank() != 0:
            return
            
        final_checkpoint_path = os.path.join(self.save_dir, "final_model.pt")
        
        # Create checkpoint dictionary
        checkpoint = {
            'step': step,
            'model_state_dict': self.trainer.train_module.model.state_dict(),
        }
        
        # Add loss if available
        try:
            if hasattr(self.trainer, 'train_state') and hasattr(self.trainer.train_state, 'loss'):
                checkpoint['loss'] = self.trainer.train_state.loss
        except Exception as e:
            logger.warning(f"Could not save loss in final checkpoint: {e}")
        
        # Add optimizer state if requested
        if self.save_optimizer and hasattr(self.trainer, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.trainer.optimizer.state_dict()
        
        # Save checkpoint
        try:
            torch.save(checkpoint, final_checkpoint_path)
            logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving final checkpoint: {e}")

class WandBCallback(Callback):
    """
    Enhanced Weights & Biases integration for OLMo training.
    """
    
    def __init__(
        self,
        project_name: str = "olmo_training",
        config: Optional[Dict[str, Any]] = None,
        log_every: int = 1
    ):
        """Initialize the WandB callback."""
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for WandBCallback")
        
        self.project_name = project_name
        self.config = config or {}
        self.initialized = False
        self.log_every = log_every
        self.last_logged_step = -1
        
    def pre_train(self):
        """Initialize WandB run before training."""
        if is_distributed() and get_rank() != 0:
            return
            
        # Clean up existing run if needed
        if wandb.run is not None:
            wandb.finish()
            
        # Initialize a fresh run with specific settings
        run = wandb.init(
            project=self.project_name,
            config=self.config,
            reinit=True,
            settings=wandb.Settings(start_method="thread")
        )
        
        self.initialized = True
        
        # Log model details
        if hasattr(self.trainer, 'train_module') and hasattr(self.trainer.train_module, 'model'):
            model = self.trainer.train_module.model
            params_info = {
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
            wandb.config.update(params_info)
            logger.info(f"Initialized WandB run: {run.name} with {params_info['num_parameters']} parameters")
    
    def post_step(self):
        """Log metrics after each training step."""
        if not self.initialized or (is_distributed() and get_rank() != 0):
            return
            
        current_step = self.trainer.global_step
        
        # Skip if we've already logged this step or if it's not time to log yet
        if current_step <= self.last_logged_step or current_step % self.log_every != 0:
            return
            
        # Remember this step
        self.last_logged_step = current_step
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Method 1: Get metrics from console logger format (most reliable)
        # This is what appears in the console output
        try:
            if hasattr(self.trainer, 'stats') and self.trainer.stats and current_step in self.trainer.stats:
                step_metrics = self.trainer.stats[current_step]
                for name, value in step_metrics.items():
                    # Convert to Python scalar if it's a tensor
                    if isinstance(value, torch.Tensor):
                        try:
                            metrics[name] = value.item()
                        except:
                            pass
                    elif isinstance(value, (int, float)):
                        metrics[name] = value
                    
                    # Special case: Store original loss in a standard format for charts  
                    if name == "train/CE loss" and "loss" not in metrics:
                        metrics["loss"] = metrics[name]
        except Exception as e:
            logger.warning(f"Error extracting metrics from trainer.stats: {e}")
        
        # Method 2: Get loss directly from train module
        try:
            # Sometimes the loss is stored directly on the train module
            if hasattr(self.trainer.train_module, 'loss'):
                loss = self.trainer.train_module.loss
                if isinstance(loss, torch.Tensor):
                    metrics["loss"] = loss.item()
                else:
                    metrics["loss"] = float(loss)
                
                # Calculate perplexity if we have CE loss
                if metrics["loss"] > 0:
                    metrics["perplexity"] = float(math.exp(metrics["loss"]))
        except Exception as e:
            logger.debug(f"Could not extract loss from train_module: {e}")
        
        # Method 3: Extract learning rate
        try:
            if hasattr(self.trainer.train_module, 'optimizer'):
                optimizer = self.trainer.train_module.optimizer
                if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                    metrics["learning_rate"] = optimizer.param_groups[0]['lr']
        except Exception as e:
            logger.debug(f"Could not extract learning rate: {e}")
        
        # Add step information
        metrics["step"] = current_step
        metrics["epoch"] = getattr(self.trainer, 'epoch', 0)
        
        # Log metrics
        if metrics:
            # Log with explicit step to ensure proper ordering
            wandb.log(metrics, step=current_step)
            
            # Show what we're logging in debug mode
            logger.debug(f"Logged {len(metrics)} metrics to WandB at step {current_step}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"WandB metrics: {metrics}")
        else:
            logger.warning(f"No metrics found to log to WandB at step {current_step}")
    
    def post_train(self):
        """Clean up WandB run after training."""
        if not self.initialized or (is_distributed() and get_rank() != 0):
            return
        
        # Make sure final metrics are logged
        self.post_step()
        
        # Add summary metrics
        try:
            if hasattr(self.trainer, 'stats'):
                last_step = max(self.trainer.stats.keys()) if self.trainer.stats else None
                if last_step is not None:
                    for name, value in self.trainer.stats[last_step].items():
                        if isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                            if isinstance(value, torch.Tensor):
                                value = value.item()
                            wandb.run.summary[name] = value
        except Exception as e:
            logger.warning(f"Error logging summary metrics: {e}")
        
        # Finish the run
        wandb.finish()