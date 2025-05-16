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
    Weights & Biases integration for experiment tracking.
    """
    
    def __init__(
        self,
        project_name: str = "olmo_training",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the WandB callback.
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
    
    def pre_train(self):
        """Initialize WandB run before training."""
        # Only initialize on main process
        if is_distributed() and get_rank() != 0:
            return
            
        # Initialize wandb
        if not self.initialized:
            wandb.init(project=self.project_name, config=self.config)
            self.initialized = True
            
            # Log model details if available
            if hasattr(self.trainer.train_module, 'model'):
                model = self.trainer.train_module.model
                wandb.config.update({
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
                })
    
    def post_step(self):
        """Log metrics after each step."""
        if not self.initialized or (is_distributed() and get_rank() != 0):
            return
            
        # Extract and log metrics
        metrics = {}
        
        # FIXED: Direct approach to get loss from train_module
        try:
            # The most reliable way to get loss from OLMo's trainer
            train_module = self.trainer.train_module
            if hasattr(train_module, 'loss'):
                loss = train_module.loss
                # Ensure loss is a Python primitive, not a tensor
                if isinstance(loss, torch.Tensor):
                    metrics["loss"] = loss.item()
                else:
                    metrics["loss"] = float(loss)
                
                # Also calculate and log perplexity
                if metrics["loss"] > 0:
                    metrics["perplexity"] = torch.exp(torch.tensor(metrics["loss"])).item()
        except Exception as e:
            logger.warning(f"Could not extract loss from train_module: {e}")
            
            # Fallbacks - try other known locations
            try:
                if hasattr(self.trainer, 'train_state') and hasattr(self.trainer.train_state, 'loss'):
                    loss = self.trainer.train_state.loss
                    if isinstance(loss, torch.Tensor):
                        metrics["loss"] = loss.item()
                    else:
                        metrics["loss"] = float(loss)
            except Exception:
                pass
            
            try:
                # Check metrics dict directly
                if hasattr(self.trainer, 'metrics') and isinstance(self.trainer.metrics, dict):
                    for key, value in self.trainer.metrics.items():
                        if "loss" in key.lower():
                            if isinstance(value, torch.Tensor):
                                metrics[key] = value.item()
                            else:
                                metrics[key] = float(value)
            except Exception:
                pass
            
        # Log learning rate
        try:
            optimizer = getattr(self.trainer.train_module, 'optimizer', None)
            if optimizer and hasattr(optimizer, 'param_groups'):
                metrics["learning_rate"] = optimizer.param_groups[0]['lr']
        except Exception as e:
            logger.warning(f"Could not extract learning rate: {e}")
        
        # Log step
        metrics["step"] = self.trainer.global_step
        
        # Add any other metrics from trainer stats if available 
        try:
            if hasattr(self.trainer, 'stats') and isinstance(self.trainer.stats, dict):
                for k, v in self.trainer.stats.items():
                    if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1):
                        if isinstance(v, torch.Tensor):
                            metrics[k] = v.item()
                        else:
                            metrics[k] = v
        except Exception as e:
            logger.warning(f"Could not extract additional metrics from trainer.stats: {e}")
        
        # Debug log what we're sending to wandb
        if metrics:
            logger.debug(f"Logging metrics to wandb: {metrics}")
            wandb.log(metrics)
        else:
            logger.warning("No metrics found to log to wandb")
    
    def post_train(self):
        """Finalize WandB run after training."""
        if not self.initialized or (is_distributed() and get_rank() != 0):
            return
        
        wandb.finish()