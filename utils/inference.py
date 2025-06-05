import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from olmo_core.train.callbacks import Callback
import wandb
from olmo_core.distributed.utils import is_distributed, get_rank
import time
import random

class InferenceCallback(Callback):
    def __init__(self, model, tokenizer_config, prompts, interval, max_new_tokens=50, inference_mode="all", skip_pre_train=False):
        self.model = model
        self.tokenizer_config = tokenizer_config
        self.prompts = prompts if isinstance(prompts, list) else [prompts]
        self.interval = int(interval)
        self.max_new_tokens = max_new_tokens
        self.inference_mode = inference_mode
        self.skip_pre_train = skip_pre_train

        if not is_distributed() or get_rank() == 0:
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")
        else:
            self.tokenizer = None

    def pre_train(self):
        if not self.skip_pre_train:
            if not is_distributed() or get_rank() == 0:
                print("Running pre_train inference...")
                self.run_inference(0, self.max_new_tokens)
        else:
            if not is_distributed() or get_rank() == 0:
                print("Skipping pre_train inference.")

    def post_step(self):
        if self.trainer.global_step > 0 and self.trainer.global_step % self.interval == 0:
            self.run_inference(self.trainer.global_step, self.max_new_tokens)

    def run_inference(self, step, max_new_tokens):
        """
        Production-ready, FSDP-safe inference callback for the OLMo model.
        This version minimizes GPU/CPU synchronization warnings.
        """
        rank = get_rank() if is_distributed() else 0
        actual_model = self.trainer.train_module.model
        is_fsdp = hasattr(actual_model, '_fsdp_enabled') or 'FSDP' in str(type(actual_model))

        # All ranks must enter the eval context
        actual_model.eval()

        # All ranks must participate in the callback to avoid deadlocks
        try:
            input_tensor = None
            tokens = []
            prompt_for_logging = ""

            # Step 1: Rank 0 prepares data and determines its size
            if rank == 0:
                if self.tokenizer is None:
                    print("Tokenizer not available on rank 0. Skipping inference.")
                    if is_distributed():
                         # Signal an error with a -1 size
                        dist.broadcast(torch.tensor([-1], device=self.trainer.device), src=0)
                    return
                # Select a prompt and tokenize it
                if self.inference_mode == "cycle":
                    prompt = self.prompts[step % len(self.prompts)]
                elif self.inference_mode == "random":
                    prompt = random.choice(self.prompts)
                else:
                    prompt = self.prompts[0]
                prompt_for_logging = prompt
                tokens = [t for t in self.tokenizer.encode(prompt) if t != 0]

            # Step 2: Synchronize the tensor across all GPUs
            if is_distributed():
                # Broadcast the size of the token list
                size_tensor = torch.tensor([len(tokens) if rank == 0 else 0], dtype=torch.long, device=self.trainer.device)
                dist.broadcast(size_tensor, src=0)
                synced_size = size_tensor.item()

                if synced_size == -1: return # Exit if rank 0 had an error

                # Create tensors on their respective devices with the correct size
                if rank == 0:
                    input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.trainer.device)
                else:
                    input_tensor = torch.empty((1, synced_size), dtype=torch.long, device=self.trainer.device)
                
                # Broadcast the actual tensor data from rank 0 to all other ranks
                dist.broadcast(input_tensor, src=0)
            else: # Non-distributed case
                 if self.tokenizer:
                    input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.trainer.device)

            if input_tensor is None: return
            
            generated_tokens_list = tokens
            
            for _ in range(max_new_tokens):
                # Step 3: Perform the forward pass
                with torch.no_grad():
                    if is_fsdp:
                        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                        with FSDP.summon_full_params(actual_model, recurse=True, offload_to_cpu=False):
                            logits = actual_model(input_tensor)
                    else:
                        logits = actual_model(input_tensor)

                # Step 4: Process and log output only on rank 0
                if rank == 0:
                    # All calculations are kept on the GPU until the very end
                    next_token_logits = logits[0, -1, :] / 0.8
                    next_token_logits[0] = -float("inf") # Suppress special tokens
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1) # Get tensor on GPU
                    
                    generated_tokens_list.extend(next_token.tolist())
                    
                    # Update input_tensor for the next iteration
                    input_tensor = torch.tensor([generated_tokens_list], dtype=torch.long, device=self.trainer.device)

            if rank == 0:
                print(f"Inference at step {step} successful.")
                decoded = self.tokenizer.decode(generated_tokens_list)
                print(f"[Step {step}] Generated: {decoded}")

                if wandb.run is not None:
                    wandb.log({f"inference/step_{step}/full_text": decoded}, step=step)

        finally:
            # Ensure all ranks return the model to training mode and synchronize
            actual_model.train()
            if is_distributed():
                dist.barrier()

# import torch
# import torch.distributed as dist
# from transformers import AutoTokenizer
# from olmo_core.train.callbacks import Callback
# import wandb
# from olmo_core.distributed.utils import is_distributed, get_rank
# import time
# import random

# class InferenceCallback(Callback):
#     def __init__(self, model, tokenizer_config, prompts, interval, inference_mode="all", skip_pre_train=False):
#         self.model = model
#         self.tokenizer_config = tokenizer_config
#         self.prompts = prompts if isinstance(prompts, list) else [prompts]
#         self.interval = int(interval)
#         self.inference_mode = inference_mode
#         self.skip_pre_train = skip_pre_train

#         if not is_distributed() or get_rank() == 0:
#             self.tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")
#         else:
#             self.tokenizer = None

#     def pre_train(self):
#         if not self.skip_pre_train:
#             if not is_distributed() or get_rank() == 0:
#                 print("Running pre_train inference...")
#                 self.run_inference(0)
#         else:
#             if not is_distributed() or get_rank() == 0:
#                 print("Skipping pre_train inference.")

#     def post_step(self):
#         if self.trainer.global_step > 0 and self.trainer.global_step % self.interval == 0:
#             self.run_inference(self.trainer.global_step)

#     def run_inference(self, step):
#         """
#         Production-ready, FSDP-safe inference callback for the OLMo model.
#         This version minimizes GPU/CPU synchronization warnings.
#         """
#         rank = get_rank() if is_distributed() else 0
#         actual_model = self.trainer.train_module.model
#         is_fsdp = hasattr(actual_model, '_fsdp_enabled') or 'FSDP' in str(type(actual_model))

#         # All ranks must enter the eval context
#         actual_model.eval()

#         # All ranks must participate in the callback to avoid deadlocks
#         try:
#             input_tensor = None
#             tokens = []
#             prompt_for_logging = ""

#             # Step 1: Rank 0 prepares data and determines its size
#             if rank == 0:
#                 if self.tokenizer is None:
#                     print("Tokenizer not available on rank 0. Skipping inference.")
#                     if is_distributed():
#                          # Signal an error with a -1 size
#                         dist.broadcast(torch.tensor([-1], device=self.trainer.device), src=0)
#                     return
#                 # Select a prompt and tokenize it
#                 if self.inference_mode == "cycle":
#                     prompt = self.prompts[step % len(self.prompts)]
#                 elif self.inference_mode == "random":
#                     prompt = random.choice(self.prompts)
#                 else:
#                     prompt = self.prompts[0]
#                 prompt_for_logging = prompt
#                 tokens = [t for t in self.tokenizer.encode(prompt) if t != 0]

#             # Step 2: Synchronize the tensor across all GPUs
#             if is_distributed():
#                 # Broadcast the size of the token list
#                 size_tensor = torch.tensor([len(tokens) if rank == 0 else 0], dtype=torch.long, device=self.trainer.device)
#                 dist.broadcast(size_tensor, src=0)
#                 synced_size = size_tensor.item()

#                 if synced_size == -1: return # Exit if rank 0 had an error

#                 # Create tensors on their respective devices with the correct size
#                 if rank == 0:
#                     input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.trainer.device)
#                 else:
#                     input_tensor = torch.empty((1, synced_size), dtype=torch.long, device=self.trainer.device)
                
#                 # Broadcast the actual tensor data from rank 0 to all other ranks
#                 dist.broadcast(input_tensor, src=0)
#             else: # Non-distributed case
#                  if self.tokenizer:
#                     input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.trainer.device)

#             if input_tensor is None: return

#             # Step 3: Perform the forward pass
#             with torch.no_grad():
#                 if is_fsdp:
#                     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
#                     with FSDP.summon_full_params(actual_model, recurse=True, offload_to_cpu=False):
#                         logits = actual_model(input_tensor)
#                 else:
#                     logits = actual_model(input_tensor)

#             # Step 4: Process and log output only on rank 0
#             if rank == 0:
#                 print(f"Inference at step {step} successful.")
#                 # All calculations are kept on the GPU until the very end
#                 next_token_logits = logits[0, -1, :] / 0.8
#                 next_token_logits[0] = -float("inf") # Suppress special tokens
#                 probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
#                 next_token = torch.multinomial(probs, num_samples=1) # Get tensor on GPU
                
#                 generated_tokens = tokens + next_token.tolist() # .tolist() is a clean way to get data
#                 decoded = self.tokenizer.decode(generated_tokens)
#                 print(f"[Step {step}] Generated: {decoded}")

#                 if wandb.run is not None:
#                     wandb.log({f"inference/step_{step}/full_text": decoded}, step=step)

#         finally:
#             # Ensure all ranks return the model to training mode and synchronize
#             actual_model.train()
#             if is_distributed():
#                 dist.barrier()
