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
                    # Log as HTML for better formatting in the UI
                    wandb.log({
                        "inference/generated_text": wandb.Html(f"<p><b>Prompt:</b> {self.prompt}</p><p>{generated_text}</p>"),
                        "inference/num_tokens": len(generated),
                        "inference/new_tokens": len(generated) - len(tokens)
                    }, step=step)
                except Exception as e:
                    logger.warning(f"Error logging to WandB: {e}")
                
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
        config: Optional[Dict[str, Any]] = None,
        create_run: bool = True
    ):
        """
        Initialize the WandB callback.
        
        Args:
            project_name: WandB project name
            config: Configuration to log
            create_run: Whether to create a new run or reuse existing one
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for WandBCallback")
        
        self.project_name = project_name
        self.config = config or {}
        self.initialized = False
        self.last_step = -1
        self.create_run = create_run
        self.metrics_buffer = {}
    
    def pre_train(self):
        """Initialize WandB run before training."""
        if is_distributed() and get_rank() != 0:
            return
        
        # Check if wandb is already initialized
        if wandb.run is not None:
            logger.info("WandB run already exists, reusing it")
            self.initialized = True
            return
            
        # Only create a new run if requested
        if self.create_run:
            # Start a fresh wandb run
            run = wandb.init(
                project=self.project_name, 
                config=self.config,
                name=f"olmo-train-{time.strftime('%Y%m%d-%H%M%S')}",
                resume="allow"
            )
            
            self.initialized = True
            logger.info(f"Initialized new WandB run: {run.name}")
        
        # Log model details
        if self.initialized and hasattr(self.trainer, 'train_module') and hasattr(self.trainer.train_module, 'model'):
            model = self.trainer.train_module.model
            params_info = {
                "model/num_parameters": sum(p.numel() for p in model.parameters()),
                "model/trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            wandb.log(params_info, step=0)
    
    def post_step(self):
        """Log metrics after each step."""
        if not self.initialized or (is_distributed() and get_rank() != 0):
            return
            
        # Get current step and ensure monotonic increases
        step = self.trainer.global_step
        if step <= self.last_step:
            return
        
        self.last_step = step
        
        # Initialize metrics dictionary
        metrics = {
            "step": step,  # Always include step number
            "epoch": getattr(self.trainer, 'epoch', 0)  # Include epoch if available
        }
        
        # Method 1: Try to extract metrics from console_logger callback
        # This is more reliable than trying to access trainer.stats directly
        console_logger = None
        for callback_name, callback in self.trainer._callbacks.items():
            if 'console_logger' in callback_name:
                console_logger = callback
                break
                
        if console_logger is not None and hasattr(console_logger, 'metrics'):
            for name, value in console_logger.metrics.items():
                if isinstance(value, torch.Tensor):
                    try:
                        metrics[name] = value.item()
                    except:
                        pass
                elif isinstance(value, (int, float)):
                    metrics[name] = value
        
        # Method 2: Directly get loss from train_module
        try:
            train_module = self.trainer.train_module
            if hasattr(train_module, 'loss'):
                loss = train_module.loss
                if isinstance(loss, torch.Tensor):
                    metrics["train/loss"] = loss.item()
                else:
                    metrics["train/loss"] = float(loss)
                
                # Calculate perplexity
                if metrics["train/loss"] > 0:
                    metrics["train/perplexity"] = math.exp(metrics["train/loss"])
        except Exception as e:
            pass
        
        # Method 3: Look for metrics in an official way
        try:
            for metric_key, metric_value in self.trainer.get_metrics().items():
                if isinstance(metric_value, torch.Tensor):
                    metrics[metric_key] = metric_value.item()
                else:
                    metrics[metric_key] = metric_value
        except:
            pass
            
        # Method 4: Check if there are metrics in train_state
        try:
            if hasattr(self.trainer, 'train_state'):
                for key in dir(self.trainer.train_state):
                    if key.startswith('_'):
                        continue
                    val = getattr(self.trainer.train_state, key)
                    if isinstance(val, (int, float)):
                        metrics[f"train_state/{key}"] = val
                    elif isinstance(val, torch.Tensor) and val.numel() == 1:
                        metrics[f"train_state/{key}"] = val.item()
        except:
            pass
            
        # Get learning rate
        try:
            optim = getattr(self.trainer.train_module, 'optimizer', None)
            if optim and hasattr(optim, 'param_groups') and len(optim.param_groups) > 0:
                metrics["train/learning_rate"] = optim.param_groups[0]['lr']
        except Exception as e:
            pass
        
        # Only log if we have metrics beyond step/epoch
        if len(metrics) > 2:
            wandb.log(metrics, step=step)
    
    def post_train(self):
        """Finalize WandB run after training."""
        if not self.initialized or (is_distributed() and get_rank() != 0):
            return
            
        # Only finish the run if we created it
        if self.create_run and wandb.run is not None:
            wandb.finish()