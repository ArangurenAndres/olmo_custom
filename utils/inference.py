import torch
from transformers import AutoTokenizer
from olmo_core.train.callbacks import Callback
import wandb

class InferenceCallback(Callback):
    def __init__(self, model, tokenizer_config, prompts, interval, inference_mode="all", skip_pre_train=False):
        self.model = model
        self.tokenizer_config = tokenizer_config
        self.prompts = prompts if isinstance(prompts, list) else [prompts]
        self.interval = int(interval)
        self.inference_mode = inference_mode
        self.skip_pre_train = skip_pre_train
        
        # Initialize tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")

    def pre_train(self):
        if not self.skip_pre_train:
            self.run_inference(0)
        else:
            print("Skipping pre_train inference in distributed mode")

    def post_step(self):
        if self.trainer.global_step % self.interval == 0 and self.trainer.global_step > 0:
            self.run_inference(self.trainer.global_step)

    def run_inference(self, step):
        """Run inference with proper FSDP handling"""
        try:
            # Set model to eval mode
            self.model.eval()
            
            # Select prompt based on inference mode
            if self.inference_mode == "cycle":
                prompt = self.prompts[step % len(self.prompts)]
            elif self.inference_mode == "random":
                import random
                prompt = random.choice(self.prompts)
            else:  # "all" mode
                prompt = self.prompts[0]  # Use first prompt for simplicity in distributed mode
            
            # Tokenize input (filter out token 0)
            tokens = [t for t in self.tokenizer.encode(prompt) if t != 0]
            input_tensor = torch.tensor([tokens], device=self.model.device)
            generated = tokens.copy()

            print(f"[Step {step}] Starting inference with prompt: {prompt[:50]}...")

            with torch.no_grad():
                # Initial forward pass
                logits = self.model(input_tensor)
                
                # Generate tokens one by one
                for i in range(50):  # Limit generation length
                    # Get logits for next token
                    next_token_logits = logits[0, -1, :] / 0.8  # Temperature scaling
                    next_token_logits[0] = -float("inf")  # Avoid token 0
                    
                    # Sample next token
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    
                    # Check for EOS token
                    if hasattr(self.tokenizer_config, 'eos_token_id') and next_token == self.tokenizer_config.eos_token_id:
                        break
                    
                    # Add token and continue
                    generated.append(next_token)
                    
                    # Update input tensor for next iteration
                    input_tensor = torch.tensor([generated], device=self.model.device)
                    logits = self.model(input_tensor)

            # Decode and print result
            try:
                decoded = self.tokenizer.decode(generated)
                print(f"[Step {step}] Generated: {decoded}")
            except Exception as decode_error:
                print(f"[Step {step}] Decoding error: {decode_error}")
                print(f"[Step {step}] Generated {len(generated)} tokens")
            
        except Exception as e:
            print(f"[Step {step}] Inference error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always return model to train mode
            self.model.train()

    def _run_single_inference(self, prompt, step, prompt_idx):
        tokens = [t for t in self.tokenizer.encode(prompt) if t != 0]
        input_tensor = torch.tensor([tokens], device=self.model.device)
        generated = tokens.copy()

        with torch.no_grad():
            logits = self.model(input_tensor)
            for _ in range(50):
                next_token_logits = logits[0, -1, :]
                next_token_logits = next_token_logits / 0.8
                if next_token_logits.size(0) > 0:
                    next_token_logits[0] = -float("inf")

                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                token = torch.multinomial(probs, 1).item()
                if token == self.tokenizer_config.eos_token_id:
                    break
                generated.append(token)
                input_tensor = torch.tensor([generated], device=self.model.device)
                logits = self.model(input_tensor)

        decoded_text = self.tokenizer.decode(generated)
        print(f"[Step {step}] Prompt {prompt_idx + 1}: {prompt[:50]}...")
        print(f"[Step {step}] Generated: {decoded_text}")

        if wandb.run is not None:
            # Extract only the newly generated part for specific logging
            # 'tokens' contains the tokenized prompt
            # 'generated' contains the tokenized prompt + tokenized generation
            original_prompt_token_count = len(tokens)
            newly_generated_tokens_only = generated[original_prompt_token_count:]
            generated_text_only_for_log = self.tokenizer.decode(newly_generated_tokens_only)

            wandb.log({
                f"inference/prompt_{prompt_idx + 1}/prompt_text": prompt,
                f"inference/prompt_{prompt_idx + 1}/generated_text": wandb.Html(f"<p><strong>Prompt:</strong> {prompt}</p><p><strong>Generated:</strong> {generated_text_only_for_log}</p>")
            }, step=step)
