"""
Inference callback for generating text during training.
"""

import torch
import wandb
from transformers import AutoTokenizer
from olmo_core.train.callbacks import Callback


class InferenceCallback(Callback):
    """
    Callback for running inference during training to monitor progress.
    """
    
    def __init__(self, model, tokenizer_config, inference_interval, inference_prompt, log_file=None):
        """
        Initialize the inference callback.
        
        Args:
            model: The OLMo model
            tokenizer_config: Tokenizer configuration
            inference_interval: Interval for running inference
            inference_prompt: Prompt for inference
            log_file: Path to log file
        """
        self.model = model
        self.tokenizer_config = tokenizer_config
        self.inference_interval = inference_interval
        self.inference_prompt = inference_prompt
        self.inference_tokenizer = None
        self.log_file = log_file
        
    def log_message(self, message):
        """
        Log a message to both console and file.
        
        Args:
            message: Message to log
        """
        print(message, flush=True)
        # Only write to file if the log file path is defined
        if self.log_file:
            # Open the file with UTF-8 encoding
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{message}\n")
        
    def pre_train(self):
        """
        Called before training loop starts.
        """
        # Load tokenizer
        self.inference_tokenizer = AutoTokenizer.from_pretrained(
            "allenai/gpt-neox-olmo-dolma-v1_5",
            cache_dir="cache/tokenizers"
        )
        
        # Run initial inference
        self.log_message("Running inference with untrained model...")
        self.run_inference(0)
    
    def post_step(self):
        """
        Called after each training step.
        """
        if self.trainer.global_step % self.inference_interval == 0 and self.trainer.global_step > 0:
            self.log_message(f"Triggering inference at step {self.trainer.global_step}")
            self.run_inference(self.trainer.global_step)
            
    def post_train(self):
        """
        Called at the end of training.
        """
        # Run final inference
        self.log_message("Running final inference with trained model...")
        self.run_inference(self.trainer.global_step)
        
    def run_inference(self, step):
        """
        Generate text with the model for monitoring training progress.
        
        Args:
            step: Current training step
        """
        self.log_message(f"Running inference at step {step}...")
    
        # Save original model state
        was_training = self.model.training
        self.model.eval()  # Set to evaluation mode
    
        # Tokenize the prompt
        tokens = self.inference_tokenizer.encode(self.inference_prompt)
    
        # Ensure no token ID 0 in prompt
        tokens = [t for t in tokens if t != 0]
    
        token_tensor = torch.tensor([tokens], device=self.model.device)
    
        with torch.no_grad():
            # Initial forward pass
            logits = self.model(token_tensor)  # OLMo returns logits directly
        
            # Setup generation
            generated_tokens = tokens.copy()
            max_new_tokens = 50
            temperature = 0.8  # Add some randomness
            
            for _ in range(max_new_tokens):
                # Get predictions for next token
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Create a mask to filter out token ID 0
                mask = torch.ones_like(next_token_logits, dtype=torch.bool)
                mask[0] = False  # Mask out token ID 0
                
                # Apply the mask (set masked logits to negative infinity)
                masked_logits = next_token_logits.clone()
                masked_logits[~mask] = -float('inf')
                
                # Apply softmax to masked logits
                probs = torch.nn.functional.softmax(masked_logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Stop if EOS token
                if next_token == self.tokenizer_config.eos_token_id:
                    break
                
                # Add token and continue
                generated_tokens.append(next_token)
                next_input = torch.tensor([generated_tokens], device=self.model.device)
                logits = self.model(next_input)
    
        # Decode the generated text
        generated_text = self.inference_tokenizer.decode(generated_tokens)
    
        # Log output
        message = f"INFERENCE at Step {step}: \"{generated_text}\""
        self.log_message(message)
    
        # Log to W&B
        if wandb.run is not None:
            wandb.log({
                "inference/generated_text": wandb.Html(f"<p>{generated_text}</p>"),
                "inference/generated_length": len(generated_tokens) - len(tokens)
            }, step=step)
    
        # Restore original model state
        if was_training:
            self.model.train()
    
        return generated_text 
