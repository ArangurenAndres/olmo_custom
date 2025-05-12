import torch
from transformers import AutoTokenizer
from olmo_core.train.callbacks import Callback
from torch.nn.parallel import DistributedDataParallel as DDP # Import DDP to check instance type

# It's good practice to have a utility function for checking the main process
# This might be defined in your train.py or a common utils file.
# For this example, I'll redefine it here for clarity.
def is_main_process(global_rank: int) -> bool:
    return global_rank == 0

class InferenceCallback(Callback):
    def __init__(self, model, tokenizer_config, prompt: str, interval: int,
                 global_rank: int, local_rank: int, is_distributed: bool):
        super().__init__() # Call superclass __init__ if it has one
        self.prompt = prompt
        self.interval = interval
        self.global_rank = global_rank
        self.is_distributed = is_distributed
        self.local_rank = local_rank # Useful for setting device on main process

        self.model_to_infer = None
        self.tokenizer = None
        self.device_for_inference = None
        self.effective_eos_token_id = None

        if is_main_process(self.global_rank):
            # Unwrap the model if it's DDP-wrapped
            if self.is_distributed and isinstance(model, DDP):
                self.model_to_infer = model.module
            else:
                self.model_to_infer = model

            # Determine and set the device for inference operations on the main process
            if torch.cuda.is_available():
                # In a distributed setup, local_rank determines the GPU for the main process.
                # In a non-distributed CUDA setup, it's typically GPU 0.
                cuda_device_idx = self.local_rank if self.is_distributed else 0
                self.device_for_inference = torch.device(f"cuda:{cuda_device_idx}")
            else:
                self.device_for_inference = torch.device("cpu")
            
            # Ensure the inference model is on the correct device.
            # This might be redundant if the model (rank 0's replica) is already on this device,
            # but it's a safeguard.
            self.model_to_infer.to(self.device_for_inference)

            print(f"[Rank {self.global_rank}] InferenceCallback: Initializing tokenizer on device {self.device_for_inference}.")
            # It's generally fine to load the tokenizer this way.
            # Hugging Face handles caching, so it shouldn't cause issues if other ranks
            # were to (mistakenly) call it, though our logic prevents that.
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")

            # Determine the EOS token ID to use for generation.
            # Prioritize the EOS token ID from the training tokenizer_config if available.
            if hasattr(tokenizer_config, 'eos_token_id') and tokenizer_config.eos_token_id is not None:
                self.effective_eos_token_id = tokenizer_config.eos_token_id
            elif self.tokenizer.eos_token_id is not None:
                self.effective_eos_token_id = self.tokenizer.eos_token_id
            else:
                # Fallback: if no EOS token is found, generation might not stop correctly.
                # Using pad_token_id as a last resort if available, or logging a warning.
                self.effective_eos_token_id = self.tokenizer.pad_token_id
                if self.effective_eos_token_id is None:
                    print(f"[Rank {self.global_rank}] Warning: InferenceCallback: EOS token ID not found in provided "
                          f"tokenizer_config or the loaded 'allenai/gpt-neox-olmo-dolma-v1_5' tokenizer. "
                          f"Generation might not stop as expected.")
                else:
                    print(f"[Rank {self.global_rank}] Warning: InferenceCallback: EOS token ID not found. Using PAD token ID "
                          f"({self.effective_eos_token_id}) as EOS. This might not be optimal.")
            
            if self.tokenizer.pad_token_id is None and self.effective_eos_token_id is not None:
                 # If the tokenizer doesn't have a pad token, but we have an EOS,
                 # some models/generation pipelines might use EOS as PAD.
                 # For AutoTokenizer, it's usually set if available.
                 # If batch generation were used, padding would be more critical.
                 # For single-prompt generation here, it's less of an issue.
                 pass


    def pre_train(self):
        if is_main_process(self.global_rank) and self.model_to_infer is not None:
            self.run_inference(step=0)

    def post_step(self):
        if is_main_process(self.global_rank) and self.model_to_infer is not None:
            if self.trainer.global_step > 0 and self.trainer.global_step % self.interval == 0:
                self.run_inference(self.trainer.global_step)

    def run_inference(self, step: int):
        # This method should only be callable if self.model_to_infer and self.tokenizer are initialized,
        # which only happens on the main process.
        if not is_main_process(self.global_rank) or self.model_to_infer is None or self.tokenizer is None:
            return

        self.model_to_infer.eval() # Set the inference model to evaluation mode

        # Encode the prompt, ensuring tokens are on the correct device for inference
        # The filter for t != 0 seems specific; usually, special tokens are handled by the tokenizer.
        # Assuming this is intentional for your "allenai/gpt-neox-olmo-dolma-v1_5" tokenizer or prompt.
        tokens = [t for t in self.tokenizer.encode(self.prompt) if t != 0] # Or handle special tokens as needed
        if not tokens:
            print(f"[Step {step} / Rank {self.global_rank}] Generated: Error - Prompt encoded to empty token list.")
            self.model_to_infer.train() # Return model to training mode
            return
            
        input_tensor = torch.tensor([tokens], device=self.device_for_inference)
        generated_token_ids = tokens[:] # Use a copy

        max_new_tokens = 50 # Max new tokens to generate

        print(f"[Step {step} / Rank {self.global_rank}] Generating from prompt: \"{self.prompt}\"")

        with torch.no_grad():
            for i in range(max_new_tokens):
                # Get logits from the model
                outputs = self.model_to_infer(input_tensor)
                # Logits are usually the first element of the output tuple from Hugging Face models
                next_token_logits = outputs.logits[0, -1, :] if hasattr(outputs, 'logits') else outputs[0, -1, :]


                # Apply temperature if needed (0.8 in your original code)
                # Note: Dividing logits by temperature. Temperature < 1 makes it sharper, > 1 softer.
                # Your original code had `logits[0] = -float("inf")` which seems like it was meant to
                # mask a specific token (perhaps BOS if your vocab has it at index 0 and it's not filtered).
                # This is unusual; usually, one would use `logits_processor` or directly manipulate vocab_size dimension.
                # For now, replicating the temperature scaling. The masking of token 0 is potentially problematic.
                next_token_logits = next_token_logits / 0.8
                
                # If you intend to forbid a specific token (e.g., token ID 0 if it's special and unwanted):
                # next_token_logits[0] = -float("inf") # Re-check if this specific manipulation is intended and correct

                # Get probabilities
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

                # Sample the next token
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                # Stop if EOS token is generated
                if self.effective_eos_token_id is not None and next_token_id == self.effective_eos_token_id:
                    break
                
                generated_token_ids.append(next_token_id)

                # Update input_tensor for the next iteration
                input_tensor = torch.tensor([generated_token_ids], device=self.device_for_inference)
                
                # Break if generated sequence is too long (input_tensor includes prompt)
                if input_tensor.shape[1] >= self.tokenizer.model_max_length -1 : # -1 for safety
                     print(f"[Step {step} / Rank {self.global_rank}] Warning: Max length ({self.tokenizer.model_max_length}) reached during generation.")
                     break


        decoded_text = self.tokenizer.decode(generated_token_ids)
        print(f"[Step {step} / Rank {self.global_rank}] Generated: {decoded_text}")

        self.model_to_infer.train() # Return the inference model to training mode