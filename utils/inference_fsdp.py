# utils/inference.py
import torch
from transformers import AutoTokenizer
from olmo_core.train.callbacks import Callback


class InferenceCallback(Callback):
    def __init__(self, model, tokenizer_config, prompt, interval):
        self.model = model
        self.tokenizer_config = tokenizer_config
        self.prompt = prompt
        self.interval = interval
        # It's good practice to initialize the tokenizer once
        if not hasattr(self, "tokenizer"): # Avoid re-initializing if callback is reused
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")

    def pre_train_loop(self, trainer):
        if not hasattr(self, "trainer"):
            return

        # All ranks set model to eval mode
        self.model.eval()

        if self.trainer.dist.get_rank() == 0:
            print(f"Initial inference check on rank {self.trainer.dist.get_rank()}:")
        
        # All ranks must participate in run_inference if it contains FSDP collective operations
        self.run_inference(0) 
        
        # All ranks set model back to train mode
        self.model.train()

        if self.trainer.dist.get_world_size() > 1:
            self.trainer.dist.barrier()

    def post_train_step(self, trainer, step_output):
        if not hasattr(self, "trainer"):
            return
            
        if self.trainer.global_step > 0 and self.trainer.global_step % self.interval == 0:
            # All ranks set model to eval mode
            self.model.eval()
            
            if self.trainer.dist.get_rank() == 0:
                 print(f"Running inference at global step {self.trainer.global_step} on rank {self.trainer.dist.get_rank()}:")
            
            # All ranks must participate in run_inference
            self.run_inference(self.trainer.global_step)

            # All ranks set model back to train mode
            self.model.train()

            if self.trainer.dist.get_world_size() > 1:
                self.trainer.dist.barrier()

    def run_inference(self, step):
        # Tokenization can be done by all ranks, it's typically lightweight.
        tokens = [t for t in self.tokenizer.encode(self.prompt) if t != 0]
        # Ensure model.device is valid for all ranks; FSDP handles placement.
        # If model is on CUDA, device should be the local CUDA device.
        device = self.model.device if hasattr(self.model, 'device') else self.trainer.device
        input_tensor = torch.tensor([tokens], device=device)
        generated_tokens = tokens.copy()

        with torch.no_grad():
            # This model call is a collective operation if model is FSDP wrapped.
            # All ranks MUST execute this.
            output = self.model(input_ids=input_tensor)
            logits = output.logits

            # The generation loop itself can run on all ranks.
            # The operations are mostly local until the next model call.
            for _ in range(50): # Max generated tokens
                next_token_logits = logits[0, -1, :]
                next_token_logits = next_token_logits / 0.8 # temperature
                if next_token_logits.size(0) > 0 : # Avoid error on empty logits
                     next_token_logits[0] = -float("inf") # Suppress padding token if it's at index 0
                
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                if next_token_id == self.tokenizer_config.eos_token_id:
                    break
                
                generated_tokens.append(next_token_id)
                # Re-create input_tensor for the next step of generation
                input_tensor = torch.tensor([generated_tokens], device=device)
                # This model call is also a collective operation.
                output = self.model(input_ids=input_tensor)
                logits = output.logits

        # Decoding can be done by all ranks.
        decoded_text = self.tokenizer.decode(generated_tokens)

        # Only rank 0 prints the generated text to avoid clutter.
        if hasattr(self, "trainer") and self.trainer.dist.get_rank() == 0:
            print(f"[Rank {self.trainer.dist.get_rank()} / Step {step}] Generated: {decoded_text}")
        
        self.model.train()