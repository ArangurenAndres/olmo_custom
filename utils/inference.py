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
        """FSDP-safe inference with robust, all-GPU synchronization."""
        rank = get_rank() if is_distributed() else 0
        print(f"[Rank {rank}, Step {step}] Entering run_inference.")
        start_time = time.time()

        actual_model = self.trainer.train_module.model
        is_fsdp = hasattr(actual_model, '_fsdp_enabled') or 'FSDP' in str(type(actual_model))
        actual_model.eval()

        if is_distributed():
            dist.barrier()

        try:
            input_tensor = None
            prompt_for_logging = ""
            tokens_for_logging = []

            # --- Robustly broadcast tensor size using broadcast_object_list ---
            if is_distributed():
                size_list = [0]
                if rank == 0:
                    # Tokenize on rank 0 only
                    if self.tokenizer is None:
                        # Error case
                        size_list = [-1]
                    else:
                        if self.inference_mode == "cycle":
                            prompt = self.prompts[step % len(self.prompts)]
                        elif self.inference_mode == "random":
                            prompt = random.choice(self.prompts)
                        else:
                            prompt = self.prompts[0]
                        prompt_for_logging = prompt
                        tokens = [t for t in self.tokenizer.encode(prompt) if t != 0]
                        tokens_for_logging = tokens
                        size_list = [len(tokens)]
                
                print(f"[Rank {rank}, Step {step}] Pre-broadcast size_list. Local value: {size_list}")
                dist.broadcast_object_list([size_list], src=0)
                print(f"[Rank {rank}, Step {step}] Post-broadcast size_list. Synced value: {size_list}")
                
                synced_size = size_list[0]
                if synced_size == -1:
                    print(f"[Rank {rank}, Step {step}] Received error signal. Aborting.")
                    return

                # --- Create and broadcast the main tensor, all on GPU ---
                if rank == 0:
                    # Create tensor directly on the correct GPU device
                    input_tensor = torch.tensor([tokens_for_logging], dtype=torch.long, device=self.trainer.device)
                else:
                    # Create placeholder on the correct GPU device
                    input_tensor = torch.empty((1, synced_size), dtype=torch.long, device=self.trainer.device)

                print(f"[Rank {rank}, Step {step}] Tensor created on device {input_tensor.device}. Shape: {input_tensor.shape}")
                # This broadcast should now work as shapes and devices are correct.
                dist.broadcast(input_tensor, src=0)
                print(f"[Rank {rank}, Step {step}] Post-broadcast input_tensor.")

            with torch.no_grad():
                if is_fsdp:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    with FSDP.summon_full_params(actual_model, recurse=True):
                        logits = actual_model(input_tensor).logits
                else:
                     # Non-distributed or non-FSDP case
                    if rank == 0:
                        input_tensor = torch.tensor([tokens_for_logging], dtype=torch.long, device=self.trainer.device)
                        logits = actual_model(input_tensor).logits

            if rank == 0:
                print(f"[Rank {rank}, Step {step}] Processing output on rank 0.")
                next_token_logits = logits[0, -1, :] / 0.8
                next_token_logits[0] = -float("inf")
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated = tokens_for_logging + [next_token]
                decoded = self.tokenizer.decode(generated)
                print(f"[Step {step}] Generated: {decoded}")

                if wandb.run is not None:
                    wandb.log({
                        f"inference/step_{step}/prompt": prompt_for_logging,
                        f"inference/step_{step}/full_text": decoded,
                    }, step=step)
                print(f"[Rank {rank}, Step {step}] ========== INFERENCE COMPLETE ON RANK 0 ==========")

        except Exception as e:
            if rank == 0:
                print(f"[Rank {rank}, Step {step}] CRITICAL: Inference error on Rank 0: {e}")
                import traceback
                traceback.print_exc()
        finally:
            print(f"[Rank {rank}, Step {step}] Setting model back to train mode.")
            actual_model.train()
            if is_distributed():
                dist.barrier()
                print(f"[Rank {rank}, Step {step}] Post-final barrier. Exiting run_inference.")

    # def run_inference(self, step):
    #     """Run inference with proper FSDP handling and detailed logging"""
    #     start_time = time.time()
        
    #     try:
    #         # Only run on rank 0 in distributed training
    #         if is_distributed() and get_rank() != 0:
    #             print(f"[Step {step}] Skipping inference on rank {get_rank()}")
    #             return
                
    #         print(f"[Step {step}] ========== STARTING INFERENCE ==========")
    #         print(f"[Step {step}] Time: {time.time() - start_time:.3f}s")
            
    #         # Check model state
    #         print(f"[Step {step}] Model training mode: {self.model.training}")
    #         print(f"[Step {step}] Model device: {next(self.model.parameters()).device}")
            
    #         # Set model to eval mode
    #         print(f"[Step {step}] Setting model to eval mode...")
    #         self.model.eval()
    #         print(f"[Step {step}] Model eval mode set. Time: {time.time() - start_time:.3f}s")
            
    #         # Select prompt based on inference mode
    #         if self.inference_mode == "cycle":
    #             prompt = self.prompts[step % len(self.prompts)]
    #         elif self.inference_mode == "random":
    #             import random
    #             prompt = random.choice(self.prompts)
    #         else:  # "all" mode
    #             prompt = self.prompts[0]  # Use first prompt for simplicity
            
    #         print(f"[Step {step}] Selected prompt: {prompt[:50]}...")
    #         print(f"[Step {step}] Prompt selection done. Time: {time.time() - start_time:.3f}s")
            
    #         # Tokenize input (filter out token 0)
    #         print(f"[Step {step}] Starting tokenization...")
    #         tokens = [t for t in self.tokenizer.encode(prompt) if t != 0]
    #         print(f"[Step {step}] Tokenized {len(tokens)} tokens. Time: {time.time() - start_time:.3f}s")
            
    #         # Create tensor on the correct device
    #         print(f"[Step {step}] Creating input tensor...")
    #         device = next(self.model.parameters()).device
    #         print(f"[Step {step}] Device obtained: {device}")
            
    #         # Use a more FSDP-friendly approach for tensor creation
    #         input_tensor = torch.tensor([tokens], dtype=torch.long)
    #         input_tensor = input_tensor.to(device)
    #         print(f"[Step {step}] Input tensor created and moved to device. Time: {time.time() - start_time:.3f}s")
            
    #         generated = tokens.copy()

    #         print(f"[Step {step}] Starting generation loop...")

    #         with torch.no_grad():
    #             # Generate tokens one by one with proper error handling
    #             for i in range(10):  # Reduce to 10 tokens for debugging
    #                 loop_start = time.time()
    #                 print(f"[Step {step}] Generation token {i}/10...")
                    
    #                 try:
    #                     # Forward pass - this is likely where it hangs
    #                     print(f"[Step {step}] Token {i}: Starting forward pass...")
    #                     forward_start = time.time()
                        
    #                     logits = self.model(input_tensor)
                        
    #                     forward_time = time.time() - forward_start
    #                     print(f"[Step {step}] Token {i}: Forward pass completed in {forward_time:.3f}s")
    #                     print(f"[Step {step}] Token {i}: Logits shape: {logits.shape}")
                        
    #                     # Get logits for next token
    #                     print(f"[Step {step}] Token {i}: Processing logits...")
    #                     next_token_logits = logits[0, -1, :] / 0.8  # Temperature scaling
    #                     next_token_logits[0] = -float("inf")  # Avoid token 0
                        
    #                     # Sample next token
    #                     print(f"[Step {step}] Token {i}: Sampling next token...")
    #                     probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    #                     next_token = torch.multinomial(probs, 1).item()
    #                     print(f"[Step {step}] Token {i}: Sampled token {next_token}")
                        
    #                     # Check for EOS token or special stopping conditions
    #                     if hasattr(self.tokenizer_config, 'eos_token_id') and next_token == self.tokenizer_config.eos_token_id:
    #                         print(f"[Step {step}] Token {i}: EOS token encountered, stopping")
    #                         break
    #                     if next_token == 0:  # Skip token 0
    #                         print(f"[Step {step}] Token {i}: Token 0 encountered, skipping")
    #                         continue
                            
    #                     # Add token and update input
    #                     generated.append(next_token)
    #                     print(f"[Step {step}] Token {i}: Added token, total length now {len(generated)}")
                        
    #                     # Create new input tensor efficiently
    #                     print(f"[Step {step}] Token {i}: Creating new input tensor...")
    #                     input_tensor = torch.tensor([generated], dtype=torch.long).to(device)
                        
    #                     loop_time = time.time() - loop_start
    #                     total_time = time.time() - start_time
    #                     print(f"[Step {step}] Token {i}: Loop completed in {loop_time:.3f}s, total: {total_time:.3f}s")
                        
    #                 except Exception as gen_error:
    #                     print(f"[Step {step}] Generation error at token {i}: {gen_error}")
    #                     import traceback
    #                     traceback.print_exc()
    #                     break

    #         print(f"[Step {step}] Generation loop completed. Time: {time.time() - start_time:.3f}s")

    #         # Decode and print result
    #         try:
    #             print(f"[Step {step}] Starting decoding...")
    #             decoded = self.tokenizer.decode(generated)
    #             print(f"[Step {step}] Generated: {decoded}")
                
    #             # Log to wandb if available
    #             if wandb.run is not None:
    #                 print(f"[Step {step}] Logging to wandb...")
    #                 original_prompt_token_count = len(tokens)
    #                 newly_generated_tokens = generated[original_prompt_token_count:]
    #                 generated_text_only = self.tokenizer.decode(newly_generated_tokens)
                    
    #                 wandb.log({
    #                     f"inference/step_{step}/prompt": prompt,
    #                     f"inference/step_{step}/generated": generated_text_only,
    #                     f"inference/step_{step}/full_text": decoded,
    #                     f"inference/step_{step}/generation_time": time.time() - start_time
    #                 }, step=step)
    #                 print(f"[Step {step}] Wandb logging completed")
                    
    #         except Exception as decode_error:
    #             print(f"[Step {step}] Decoding error: {decode_error}")
    #             print(f"[Step {step}] Generated {len(generated)} tokens")
            
    #     except Exception as e:
    #         print(f"[Step {step}] CRITICAL: Inference error: {e}")
    #         import traceback
    #         traceback.print_exc()
    #     finally:
    #         # Always return model to train mode
    #         print(f"[Step {step}] Setting model back to train mode...")
    #         self.model.train()
    #         total_time = time.time() - start_time
    #         print(f"[Step {step}] ========== INFERENCE COMPLETE in {total_time:.3f}s ==========")