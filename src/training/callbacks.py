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
    Weights & Biases integration for OLMo training.
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
            create_run: Whether to create a new run or use existing
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for WandBCallback")
        
        self.project_name = project_name
        self.config = config or {}
        self.initialized = False
        self.create_run = create_run
        self.last_logged_step = -1
        self.metrics_buffer = {}  # Buffer to store metrics by step
        self.max_buffer_size = 1000  # Maximum buffer size to prevent memory issues
    
    def pre_train(self):
        """Initialize WandB run before training."""
        if is_distributed() and get_rank() != 0:
            return
            
        # Handle existing wandb run
        if wandb.run is not None:
            logger.info(f"WandB already initialized, using existing run: {wandb.run.name}")
            self.initialized = True
            return
            
        if self.create_run:
            # Initialize a fresh run with unique name
            run_name = f"olmo-train-{time.strftime('%Y%m%d-%H%M%S')}"
            run = wandb.init(
                project=self.project_name,
                name=run_name,
                config=self.config,
                reinit=False
            )
            logger.info(f"Initialized new WandB run: {run.name}")
            self.initialized = True
            
            # Log model details if available
            if hasattr(self.trainer.train_module, 'model'):
                model = self.trainer.train_module.model
                wandb.config.update({
                    "model/num_parameters": sum(p.numel() for p in model.parameters()),
                    "model/trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
                })
    
    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """
        Log metrics received from trainer.
        This is the correct hook to use in OLMo, not post_step!
        """
        if not self.initialized or (is_distributed() and get_rank() != 0):
            return
        
        # Convert any non-serializable values (like tensors) to Python scalars
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                try:
                    clean_metrics[key] = value.item()
                except:
                    pass  # Skip metrics that can't be converted
            elif isinstance(value, (int, float)):
                clean_metrics[key] = value
        
        if not clean_metrics:
            return  # No metrics to log
        
        # Add metrics to buffer with their corresponding step
        if step not in self.metrics_buffer:
            self.metrics_buffer[step] = {}
        
        # Update or add new metrics
        self.metrics_buffer[step].update(clean_metrics)
        
        # Log all buffered metrics in step order
        self._flush_metrics_buffer()
    
    def _flush_metrics_buffer(self):
        """Log all buffered metrics in the correct step order."""
        # Get all steps in order
        steps = sorted(self.metrics_buffer.keys())
        
        # Log each step in order if it's greater than the last logged step
        for step in steps:
            if step > self.last_logged_step:
                wandb.log(self.metrics_buffer[step], step=step)
                self.last_logged_step = step
                del self.metrics_buffer[step]
        
        # Clean buffer if it gets too large (keep only most recent steps)
        if len(self.metrics_buffer) > self.max_buffer_size:
            logger.warning(f"WandB metrics buffer exceeds {self.max_buffer_size} items, clearing oldest entries")
            steps_to_keep = sorted(self.metrics_buffer.keys())[-self.max_buffer_size:]
            self.metrics_buffer = {step: self.metrics_buffer[step] for step in steps_to_keep}
    
    def post_step(self):
        """Additional logging at each step if needed."""
        if not self.initialized or (is_distributed() and get_rank() != 0):
            return
        
        step = self.trainer.global_step
        
        # Get learning rate which might not be in the metrics
        try:
            if hasattr(self.trainer.train_module, 'optimizer'):
                optim = self.trainer.train_module.optimizer
                if hasattr(optim, 'param_groups') and len(optim.param_groups) > 0:
                    # Add to buffer instead of logging directly
                    lr = optim.param_groups[0]['lr']
                    if step not in self.metrics_buffer:
                        self.metrics_buffer[step] = {}
                    self.metrics_buffer[step]["learning_rate"] = lr
                    
                    # Then flush buffer
                    self._flush_metrics_buffer()
        except Exception as e:
            logger.debug(f"Could not log learning rate: {e}")
    
    def post_train(self):
        """Finalize WandB run after training."""
        if not self.initialized or (is_distributed() and get_rank() != 0):
            return
        
        # Final flush of any remaining buffered metrics
        self._flush_metrics_buffer()
        
        # Only finish if we created the run
        if self.create_run and wandb.run is not None:
            wandb.finish()



# class WandBCallback(Callback):
#     """
#     Weights & Biases integration for OLMo training.
#     """
    
#     def __init__(
#         self,
#         project_name: str = "olmo_training",
#         config: Optional[Dict[str, Any]] = None,
#         create_run: bool = True
#     ):
#         """
#         Initialize the WandB callback.
        
#         Args:
#             project_name: WandB project name
#             config: Configuration to log
#             create_run: Whether to create a new run or use existing
#         """
#         if not WANDB_AVAILABLE:
#             raise ImportError("wandb is required for WandBCallback")
        
#         self.project_name = project_name
#         self.config = config or {}
#         self.initialized = False
#         self.create_run = create_run
#         self.last_logged_step = -1
    
#     def pre_train(self):
#         """Initialize WandB run before training."""
#         if is_distributed() and get_rank() != 0:
#             return
            
#         # Handle existing wandb run
#         if wandb.run is not None:
#             logger.info(f"WandB already initialized, using existing run: {wandb.run.name}")
#             self.initialized = True
#             return
            
#         if self.create_run:
#             # Initialize a fresh run with unique name
#             run_name = f"olmo-train-{time.strftime('%Y%m%d-%H%M%S')}"
#             run = wandb.init(
#                 project=self.project_name,
#                 name=run_name,
#                 config=self.config,
#                 reinit=False
#             )
#             logger.info(f"Initialized new WandB run: {run.name}")
#             self.initialized = True
            
#             # Log model details if available
#             if hasattr(self.trainer.train_module, 'model'):
#                 model = self.trainer.train_module.model
#                 wandb.config.update({
#                     "model/num_parameters": sum(p.numel() for p in model.parameters()),
#                     "model/trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
#                 })
    
#     def log_metrics(self, step: int, metrics: Dict[str, Any]):
#         """
#         Log metrics received from trainer.
#         This is the correct hook to use in OLMo, not post_step!
#         """
#         if not self.initialized or (is_distributed() and get_rank() != 0):
#             return
            
#         # Only log if step is greater than last logged step
#         if step <= self.last_logged_step:
#             return
            
#         self.last_logged_step = step
        
#         # Convert any non-serializable values (like tensors) to Python scalars
#         clean_metrics = {}
#         for key, value in metrics.items():
#             if isinstance(value, torch.Tensor):
#                 try:
#                     clean_metrics[key] = value.item()
#                 except:
#                     pass  # Skip metrics that can't be converted
#             elif isinstance(value, (int, float)):
#                 clean_metrics[key] = value
                
#         # Ensure we have data worth logging
#         if clean_metrics:
#             # Make sure we log the step explicitly
#             wandb.log(clean_metrics, step=step)
    
#     # We can still use post_step to handle additional custom logging if needed
#     def post_step(self):
#         """Additional logging at each step if needed."""
#         if not self.initialized or (is_distributed() and get_rank() != 0):
#             return
            
#         # Get learning rate which might not be in the metrics
#         try:
#             if hasattr(self.trainer.train_module, 'optimizer'):
#                 optim = self.trainer.train_module.optimizer
#                 if hasattr(optim, 'param_groups') and len(optim.param_groups) > 0:
#                     wandb.log({
#                         "learning_rate": optim.param_groups[0]['lr']
#                     }, step=self.trainer.global_step)
#         except Exception as e:
#             logger.debug(f"Could not log learning rate: {e}")
    
#     def post_train(self):
#         """Finalize WandB run after training."""
#         if not self.initialized or (is_distributed() and get_rank() != 0):
#             return
            
#         # Only finish if we created the run
#         if self.create_run and wandb.run is not None:
#             wandb.finish()

