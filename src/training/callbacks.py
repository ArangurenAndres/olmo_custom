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
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the WandB callback.
        """
        super().__init__() # Ensure base class is initialized if it has an __init__
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for WandBCallback. Please install with `pip install wandb`.")

        self.project_name = project_name
        self.config = config or {}
        self.wandb_initialized_by_this_callback = False # Flag to ensure init happens once per instance effectively
        self.run = None # To store the wandb run object

    def pre_train(self):
        """Initialize WandB run before training."""
        if is_distributed() and get_rank() != 0:
            return

        if self.wandb_initialized_by_this_callback:
            logger.warning(
                "WandBCallback.pre_train called again on rank 0 after initial setup. Ignoring."
            )
            return

        # Clean up any existing run only if it wasn't started by this callback instance
        # or if we want to enforce a fresh run every time pre_train is effectively called.
        # The original logic was to finish any active run.
        if wandb.run is not None:
            logger.warning(
                f"An existing wandb run (ID: {wandb.run.id}, Name: {wandb.run.name}) was active. "
                "The WandBCallback will finish it and start a new one as per its pre_train logic."
            )
            wandb.finish(quiet=True) # quiet=True to reduce log noise if this is common

        # Start a new wandb run
        self.run = wandb.init(project=self.project_name, config=self.config)
        if self.run is None: # wandb.init should return run object, or raise error. Robustness check.
             logger.error("wandb.init() failed to return a run object.")
             return # Cannot proceed

        self.wandb_initialized_by_this_callback = True
        logger.info(f"WandB run (ID: {self.run.id}, Name: {self.run.name}) initialized by WandBCallback.")


        # Log model details
        # Ensure self.trainer is set by the Trainer before pre_train is called.
        if hasattr(self.trainer, 'train_module') and hasattr(self.trainer.train_module, 'model'):
            model = self.trainer.train_module.model
            try:
                params_info = {
                    "model/num_parameters": sum(p.numel() for p in model.parameters()),
                    "model/trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
                }
                wandb.log(params_info)
            except Exception as e:
                logger.error(f"Failed to log model parameters: {e}")
        else:
            logger.warning("Could not log model parameters: trainer.train_module.model not found.")


    def post_step(self):
        """Log metrics after each step."""
        if not self.wandb_initialized_by_this_callback or (is_distributed() and get_rank() != 0) or wandb.run is None:
            return

        # Get current step
        # Ensure self.trainer is set and has global_step
        if not hasattr(self.trainer, 'global_step'):
            logger.warning("trainer.global_step not found. Cannot log metrics for step.")
            return
        step = self.trainer.global_step

        metrics = {} # Initialize metrics dictionary for this step

        # Get loss directly from trainer
        if hasattr(self.trainer, 'loss') and self.trainer.loss is not None:
            # self.trainer.loss should already be a scalar float
            current_loss = float(self.trainer.loss)
            metrics["train/loss"] = current_loss # Using a prefix like 'train/' is good practice
            if current_loss > 0:
                try:
                    metrics["train/perplexity"] = math.exp(current_loss)
                except OverflowError:
                    metrics["train/perplexity"] = float('inf')
                    logger.warning(f"OverflowError calculating perplexity for loss {current_loss} at step {step}.")
                except Exception as e: # General exception for math.exp
                    logger.warning(f"Error calculating perplexity for loss {current_loss} at step {step}: {e}")
        else:
            logger.debug(f"trainer.loss not available at step {step}.")


        # Get learning rate directly from trainer's optimizer
        if hasattr(self.trainer, 'optimizer') and \
           hasattr(self.trainer.optimizer, 'param_groups') and \
           self.trainer.optimizer.param_groups:
            metrics["train/learning_rate"] = self.trainer.optimizer.param_groups[0]['lr']
        else:
            logger.debug(f"trainer.optimizer.param_groups not available for LR at step {step}.")

        # Log other stats from self.trainer.stats (if any)
        # These might include validation metrics if the trainer populates them here,
        # or other custom training stats.
        if hasattr(self.trainer, 'stats') and isinstance(self.trainer.stats, dict) and step in self.trainer.stats:
            for name, value in self.trainer.stats[step].items():
                # Avoid overwriting loss/lr if already sourced more directly, unless names are different
                metric_name = f"trainer_stats/{name}" # Prefix to avoid collision
                if isinstance(value, torch.Tensor):
                    try:
                        metrics[metric_name] = value.item()
                    except ValueError: # if tensor is not a scalar
                        logger.debug(f"Cannot convert tensor '{metric_name}' to scalar at step {step}.")
                    except Exception as e:
                        logger.warning(f"Error itemizing tensor '{metric_name}' at step {step}: {e}")
                elif isinstance(value, (int, float)):
                    metrics[metric_name] = value
                # else: log a warning or skip if type is not plottable

        # Log additional metrics from self.trainer.metrics (a general dict for metrics)
        if hasattr(self.trainer, 'metrics') and isinstance(self.trainer.metrics, dict):
            for name, value in self.trainer.metrics.items():
                metric_name = f"trainer_metrics/{name}" # Prefix
                if isinstance(value, torch.Tensor):
                    try:
                        metrics[metric_name] = value.item()
                    except ValueError:
                        logger.debug(f"Cannot convert tensor '{metric_name}' from trainer.metrics to scalar at step {step}.")
                    except Exception as e:
                        logger.warning(f"Error itemizing tensor '{metric_name}' from trainer.metrics at step {step}: {e}")
                elif isinstance(value, (int, float)):
                    metrics[metric_name] = value

        # Log to wandb if there are any metrics collected
        if metrics: # Check if metrics dict is not empty
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.error(f"Failed to log metrics to wandb at step {step}: {e}. Metrics: {metrics}")
        else:
            logger.debug(f"No metrics collected to log at step {step}.")


    def post_train(self):
        """Finalize WandB run after training."""
        if not self.wandb_initialized_by_this_callback or (is_distributed() and get_rank() != 0) or wandb.run is None:
            return

        logger.info(f"Finishing WandB run (ID: {wandb.run.id}, Name: {wandb.run.name}) in post_train.")
        wandb.finish()
        self.wandb_initialized_by_this_callback = False # Reset for potential re-use if trainer is run multiple times in one script
        self.run = None

# class WandBCallback(Callback):
#     """
#     Weights & Biases integration for experiment tracking.
#     """
    
#     def __init__(
#         self,
#         project_name: str = "olmo_training",
#         config: Optional[Dict[str, Any]] = None
#     ):
#         """
#         Initialize the WandB callback.
#         """
#         if not WANDB_AVAILABLE:
#             raise ImportError("wandb is required for WandBCallback")
        
#         self.project_name = project_name
#         self.config = config or {}
#         self.initialized = False
#         self.last_step = -1
    
#     def pre_train(self):
#         """Initialize WandB run before training."""
#         if is_distributed() and get_rank() != 0:
#             return
        
#         # Properly handle existing runs - prevent duplicate initialization
#         if wandb.run is not None:
#             logger.info("WandB run already exists, reusing it")
#             self.initialized = True
#             return
        
#         # Start a clean new wandb run with explicit configuration
#         run = wandb.init(
#             project=self.project_name, 
#             config=self.config,
#             reinit=True,
#             name=f"olmo-train-{time.strftime('%Y%m%d-%H%M%S')}",
#             resume="never"
#         )
        
#         self.initialized = True
#         logger.info(f"Initialized new WandB run: {run.name}")
        
#         # Log model details immediately
#         if hasattr(self.trainer, 'train_module') and hasattr(self.trainer.train_module, 'model'):
#             model = self.trainer.train_module.model
#             params_info = {
#                 "model/num_parameters": sum(p.numel() for p in model.parameters()),
#                 "model/trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
#             }
#             wandb.log(params_info, step=0)
    
#     def post_step(self):
#         """Log metrics after each step."""
#         if not self.initialized or (is_distributed() and get_rank() != 0):
#             return
            
#         # Get current step and ensure monotonic increases
#         step = self.trainer.global_step
#         if step <= self.last_step:
#             return
        
#         self.last_step = step
        
#         # Initialize metrics dictionary
#         metrics = {}
        
#         # Get metrics from OLMo console logger output 
#         # This is the most reliable source for metrics
#         if hasattr(self.trainer, 'stats') and isinstance(self.trainer.stats, dict):
#             try:
#                 # Get the most recent stats
#                 if step in self.trainer.stats:
#                     for name, value in self.trainer.stats[step].items():
#                         # Convert tensor to scalar if needed
#                         if isinstance(value, torch.Tensor):
#                             try:
#                                 metrics[name] = value.item()
#                             except:
#                                 pass
#                         elif isinstance(value, (int, float)):
#                             metrics[name] = value
#             except Exception as e:
#                 logger.debug(f"Could not extract stats from trainer: {e}")
        
#         # Directly get loss from train_module (fallback)
#         if "train/CE loss" not in metrics and hasattr(self.trainer, 'train_module'):
#             try:
#                 train_module = self.trainer.train_module
#                 if hasattr(train_module, 'loss'):
#                     loss = train_module.loss
#                     if isinstance(loss, torch.Tensor):
#                         metrics["loss"] = loss.item()
#                     else:
#                         metrics["loss"] = float(loss)
                        
#                     # Calculate perplexity if we have loss
#                     if metrics["loss"] > 0:
#                         metrics["perplexity"] = math.exp(metrics["loss"])
#             except Exception as e:
#                 logger.debug(f"Could not extract loss from train_module: {e}")
        
#         # Get learning rate
#         try:
#             if hasattr(self.trainer.train_module, 'optimizer'):
#                 optimizer = self.trainer.train_module.optimizer
#                 if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
#                     metrics["learning_rate"] = optimizer.param_groups[0]['lr']
#         except Exception as e:
#             logger.debug(f"Could not extract learning rate: {e}")
        
#         # Log everything with explicit step to ensure proper order
#         if metrics:
#             wandb.log(metrics, step=step)
    
#     def post_train(self):
#         """Finalize WandB run after training."""
#         if not self.initialized or (is_distributed() and get_rank() != 0):
#             return
        
#         # Ensure no additional runs are created
#         if wandb.run is not None:
#             wandb.finish()