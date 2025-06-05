import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from olmo_core.train.callbacks import Callback
import wandb
from olmo_core.distributed.utils import is_distributed, get_rank
import time
import random

class InferenceCallback(Callback):
    def __init__(self, model, tokenizer_config, prompts, interval, inference_mode="all", skip_pre_train=False):
        self.model = model
        self.tokenizer_config = tokenizer_config
        self.prompts = prompts if isinstance(prompts, list) else [prompts]
        self.interval = int(interval)
        self.inference_mode = inference_mode
        self.skip_pre_train = skip_pre_train

        if not is_distributed() or get_rank() == 0:
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")
        else:
            self.tokenizer = None
        print(f"InferenceCallback initialized with interval: {self.interval}, skip_pre_train: {self.skip_pre_train}")

    def pre_train(self):
        if not self.skip_pre_train:
            self.run_inference(0)
        else:
            print("Skipping pre_train inference in distributed mode")

    def post_step(self):
        if self.trainer.global_step > 0 and self.trainer.global_step % self.interval == 0:
            rank = get_rank() if is_distributed() else 0
            if rank == 0:
                print(f"post_step: Running inference at step {self.trainer.global_step}")
            self.run_inference(self.trainer.global_step)

    def run_inference(self, step):
        """Final, robust FSDP-safe inference for the OLMo model."""
        rank = get_rank() if is_distributed() else 0
        actual_model = self.trainer.train_module.model
        is_fsdp = hasattr(actual_model, '_fsdp_enabled') or 'FSDP' in str(type(actual_model))
        actual_model.eval()

        if is_distributed():
            dist.barrier()

        try:
            input_tensor = None
            tokens = []
            prompt_for_logging = ""

            if rank == 0:
                if self.tokenizer is None:
                    if is_distributed():
                        # Signal an error to other ranks
                        dist.broadcast(torch.tensor([-1], device=self.trainer.device), src=0)
                    return
                # Tokenize and prepare data
                if self.inference_mode == "cycle":
                    prompt = self.prompts[step % len(self.prompts)]
                elif self.inference_mode == "random":
                    prompt = random.choice(self.prompts)
                else:
                    prompt = self.prompts[0]
                prompt_for_logging = prompt
                tokens = [t for t in self.tokenizer.encode(prompt) if t != 0]

            if is_distributed():
                # Step 1: Broadcast tensor size
                size_tensor = torch.tensor([len(tokens) if rank == 0 else 0], dtype=torch.long, device=self.trainer.device)
                dist.broadcast(size_tensor, src=0)
                synced_size = size_tensor.item()
                if synced_size == -1: return

                # Step 2: Create tensors and broadcast data
                if rank == 0:
                    input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.trainer.device)
                else:
                    input_tensor = torch.empty((1, synced_size), dtype=torch.long, device=self.trainer.device)
                dist.broadcast(input_tensor, src=0)
            else: # Non-distributed case
                if self.tokenizer:
                    input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.trainer.device)

            if input_tensor is None: return

            # Step 3: FSDP-aware forward pass with correct output handling
            with torch.no_grad():
                if is_fsdp:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    # Use summon_full_params to gather the model for inference
                    with FSDP.summon_full_params(actual_model, recurse=True):
                        # THE FIX: The model returns a raw tensor, not an object with .logits
                        logits = actual_model(input_tensor)
                else:
                    # Non-FSDP forward pass
                    logits = actual_model(input_tensor)

            # Step 4: Rank 0 processes and logs the output
            if rank == 0:
                print(f"[Rank {rank}, Step {step}] Processing output.")
                next_token_logits = logits[0, -1, :] / 0.8
                next_token_logits[0] = -float("inf")
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated_tokens = tokens + [next_token]
                decoded = self.tokenizer.decode(generated_tokens)
                print(f"[Step {step}] Generated: {decoded}")

                if wandb.run is not None:
                    wandb.log({f"inference/step_{step}/full_text": decoded}, step=step)

        except Exception as e:
            if rank == 0:
                print(f"[Rank {rank}, Step {step}] CRITICAL: Inference error on Rank 0: {e}")
                import traceback
                traceback.print_exc()
        finally:
            # All ranks must switch back to train mode and synchronize.
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
#         print(f"InferenceCallback initialized with interval: {self.interval}, skip_pre_train: {self.skip_pre_train}")

#     def pre_train(self):
#         if not self.skip_pre_train:
#             self.run_inference(0)
#         else:
#             print("Skipping pre_train inference in distributed mode")

#     def post_step(self):
#         if self.trainer.global_step > 0 and self.trainer.global_step % self.interval == 0:
#             rank = get_rank() if is_distributed() else 0
#             if rank == 0:
#                 print(f"post_step: Running inference at step {self.trainer.global_step}")
#             self.run_inference(self.trainer.global_step)

#     def run_inference(self, step):
#         """FSDP-safe inference with robust, all-GPU synchronization."""
#         rank = get_rank() if is_distributed() else 0
#         print(f"[Rank {rank}, Step {step}] Entering run_inference.")
#         start_time = time.time()

#         actual_model = self.trainer.train_module.model
#         is_fsdp = hasattr(actual_model, '_fsdp_enabled') or 'FSDP' in str(type(actual_model))
#         actual_model.eval()

#         if is_distributed():
#             dist.barrier()

#         try:
#             input_tensor = None
#             prompt_for_logging = ""
#             tokens_for_logging = []

#             # --- Robustly broadcast tensor size using broadcast_object_list ---
#             if is_distributed():
#                 size_list = [0]
#                 if rank == 0:
#                     # Tokenize on rank 0 only
#                     if self.tokenizer is None:
#                         # Error case
#                         size_list = [-1]
#                     else:
#                         if self.inference_mode == "cycle":
#                             prompt = self.prompts[step % len(self.prompts)]
#                         elif self.inference_mode == "random":
#                             prompt = random.choice(self.prompts)
#                         else:
#                             prompt = self.prompts[0]
#                         prompt_for_logging = prompt
#                         tokens = [t for t in self.tokenizer.encode(prompt) if t != 0]
#                         tokens_for_logging = tokens
#                         size_list = [len(tokens)]
                
#                 print(f"[Rank {rank}, Step {step}] Pre-broadcast size_list. Local value: {size_list}")
#                 dist.broadcast_object_list([size_list], src=0)
#                 print(f"[Rank {rank}, Step {step}] Post-broadcast size_list. Synced value: {size_list}")
                
#                 synced_size = size_list[0]
#                 if synced_size == -1:
#                     print(f"[Rank {rank}, Step {step}] Received error signal. Aborting.")
#                     return

#                 # --- Create and broadcast the main tensor, all on GPU ---
#                 if rank == 0:
#                     # Create tensor directly on the correct GPU device
#                     input_tensor = torch.tensor([tokens_for_logging], dtype=torch.long, device=self.trainer.device)
#                 else:
#                     # Create placeholder on the correct GPU device
#                     input_tensor = torch.empty((1, synced_size), dtype=torch.long, device=self.trainer.device)

#                 print(f"[Rank {rank}, Step {step}] Tensor created on device {input_tensor.device}. Shape: {input_tensor.shape}")
#                 # This broadcast should now work as shapes and devices are correct.
#                 dist.broadcast(input_tensor, src=0)
#                 print(f"[Rank {rank}, Step {step}] Post-broadcast input_tensor.")

#             with torch.no_grad():
#                 if is_fsdp:
#                     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
#                     with FSDP.summon_full_params(actual_model, recurse=True):
#                         logits = actual_model(input_tensor).logits
#                 else:
#                      # Non-distributed or non-FSDP case
#                     if rank == 0:
#                         input_tensor = torch.tensor([tokens_for_logging], dtype=torch.long, device=self.trainer.device)
#                         logits = actual_model(input_tensor).logits

#             if rank == 0:
#                 print(f"[Rank {rank}, Step {step}] Processing output on rank 0.")
#                 next_token_logits = logits[0, -1, :] / 0.8
#                 next_token_logits[0] = -float("inf")
#                 probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
#                 next_token = torch.multinomial(probs, 1).item()
#                 generated = tokens_for_logging + [next_token]
#                 decoded = self.tokenizer.decode(generated)
#                 print(f"[Step {step}] Generated: {decoded}")

#                 if wandb.run is not None:
#                     wandb.log({
#                         f"inference/step_{step}/prompt": prompt_for_logging,
#                         f"inference/step_{step}/full_text": decoded,
#                     }, step=step)
#                 print(f"[Rank {rank}, Step {step}] ========== INFERENCE COMPLETE ON RANK 0 ==========")

#         except Exception as e:
#             if rank == 0:
#                 print(f"[Rank {rank}, Step {step}] CRITICAL: Inference error on Rank 0: {e}")
#                 import traceback
#                 traceback.print_exc()
#         finally:
#             print(f"[Rank {rank}, Step {step}] Setting model back to train mode.")
#             actual_model.train()
#             if is_distributed():
#                 dist.barrier()
#                 print(f"[Rank {rank}, Step {step}] Post-final barrier. Exiting run_inference.")

