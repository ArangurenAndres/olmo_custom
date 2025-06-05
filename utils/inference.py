import torch
from transformers import AutoTokenizer
from olmo_core.train.callbacks import Callback
import wandb
from olmo_core.distributed.utils import is_distributed, get_rank
import time

class InferenceCallback(Callback):
    def __init__(self, model, tokenizer_config, prompts, interval, inference_mode="all", skip_pre_train=False):
        self.model = model
        self.tokenizer_config = tokenizer_config
        self.prompts = prompts if isinstance(prompts, list) else [prompts]
        self.interval = int(interval)
        self.inference_mode = inference_mode
        self.skip_pre_train = skip_pre_train
        
        # Initialize tokenizer only on rank 0
        if not is_distributed() or get_rank() == 0:
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")
        else:
            self.tokenizer = None
            
        print(f"InferenceCallback initialized with interval: {self.interval}, skip_pre_train: {self.skip_pre_train}")

    def pre_train(self):
        if not self.skip_pre_train:
            print("Running pre_train inference...")
            self.run_inference(0)
        else:
            print("Skipping pre_train inference in distributed mode")

    def post_step(self):
        # All ranks call this, but only rank 0 executes inference
        should_run = (not is_distributed() or get_rank() == 0) and \
                    self.trainer.global_step % self.interval == 0 and \
                    self.trainer.global_step > 0
        
        if should_run:
            print(f"post_step: Running inference at step {self.trainer.global_step}")
            self.run_inference(self.trainer.global_step)

    def run_inference(self, step):
        """FSDP-safe inference implementation with detailed logging"""
        start_time = time.time()
        
        try:
            # Only execute on rank 0, but method exists on all ranks
            if is_distributed() and get_rank() != 0:
                print(f"[Step {step}] Skipping inference on rank {get_rank()}")
                return
                
            if self.tokenizer is None:
                print(f"[Step {step}] No tokenizer available")
                return
                
            print(f"[Step {step}] ========== STARTING FSDP-SAFE INFERENCE ==========")
            print(f"[Step {step}] Time: {time.time() - start_time:.3f}s")
            
            # Get the actual model from the train_module (this will be FSDP-wrapped)
            actual_model = self.trainer.train_module.model
            print(f"[Step {step}] Using model from train_module: {type(actual_model)}")
            
            # Check if model is FSDP wrapped - check for OLMo Core's FSDP wrapper
            is_fsdp = hasattr(actual_model, '_fsdp_enabled') or 'FSDP' in str(type(actual_model))
            print(f"[Step {step}] Model is FSDP wrapped: {is_fsdp}")
            
            # Check model state
            print(f"[Step {step}] Model training mode: {actual_model.training}")
            
            # Set model to eval mode
            print(f"[Step {step}] Setting model to eval mode...")
            actual_model.eval()
            print(f"[Step {step}] Model eval mode set. Time: {time.time() - start_time:.3f}s")
            
            # Select prompt based on inference mode
            if self.inference_mode == "cycle":
                prompt = self.prompts[step % len(self.prompts)]
            elif self.inference_mode == "random":
                import random
                prompt = random.choice(self.prompts)
            else:  # "all" mode
                prompt = self.prompts[0]
            
            print(f"[Step {step}] Selected prompt: {prompt[:50]}...")
            print(f"[Step {step}] Prompt selection done. Time: {time.time() - start_time:.3f}s")
            
            # Tokenize input (filter out token 0)
            print(f"[Step {step}] Starting tokenization...")
            tokens = [t for t in self.tokenizer.encode(prompt) if t != 0]
            print(f"[Step {step}] Tokenized {len(tokens)} tokens. Time: {time.time() - start_time:.3f}s")
            
            # Create tensor on CPU first to avoid synchronization warning
            print(f"[Step {step}] Creating input tensor on CPU...")
            input_tensor = torch.tensor([tokens], dtype=torch.long)
            print(f"[Step {step}] Input tensor created on CPU. Time: {time.time() - start_time:.3f}s")
            
            with torch.no_grad():
                if is_fsdp:
                    print(f"[Step {step}] Using FSDP summon_full_params for inference...")
                    # For FSDP inference, we need to use summon_full_params to avoid deadlocks
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    
                    # Use summon_full_params context to ensure all parameters are available
                    with FSDP.summon_full_params(actual_model, recurse=True):
                        print(f"[Step {step}] FSDP parameters summoned. Time: {time.time() - start_time:.3f}s")
                        
                        # Move tensor to device after summon_full_params
                        device = next(actual_model.parameters()).device
                        print(f"[Step {step}] Model device: {device}")
                        input_tensor = input_tensor.to(device, non_blocking=True)
                        print(f"[Step {step}] Input tensor moved to device. Time: {time.time() - start_time:.3f}s")
                        
                        # Add CUDA synchronization for safety
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                            print(f"[Step {step}] CUDA synchronized. Time: {time.time() - start_time:.3f}s")
                        
                        # Forward pass within FSDP context
                        print(f"[Step {step}] Starting FSDP forward pass...")
                        forward_start = time.time()
                        
                        try:
                            logits = actual_model(input_tensor)
                            forward_time = time.time() - forward_start
                            print(f"[Step {step}] Forward pass completed in {forward_time:.3f}s")
                            
                            # Process output
                            next_token_logits = logits[0, -1, :] / 0.8
                            next_token_logits[0] = -float("inf")
                            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                            next_token = torch.multinomial(probs, 1).item()
                            generated = tokens + [next_token]
                            
                        except Exception as forward_error:
                            print(f"[Step {step}] FSDP forward pass error: {forward_error}")
                            import traceback
                            traceback.print_exc()
                            return
                else:
                    print(f"[Step {step}] Using regular inference for non-FSDP model...")
                    # Get device and move tensor
                    device = next(actual_model.parameters()).device
                    print(f"[Step {step}] Model device: {device}")
                    input_tensor = input_tensor.to(device, non_blocking=True)
                    print(f"[Step {step}] Input tensor moved to device. Time: {time.time() - start_time:.3f}s")
                    
                    # Add CUDA synchronization for safety
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                        print(f"[Step {step}] CUDA synchronized. Time: {time.time() - start_time:.3f}s")
                    
                    # Direct forward pass for non-FSDP case
                    print(f"[Step {step}] Starting direct forward pass...")
                    forward_start = time.time()
                    
                    try:
                        logits = actual_model(input_tensor)
                        forward_time = time.time() - forward_start
                        print(f"[Step {step}] Forward pass completed in {forward_time:.3f}s")
                        
                        # Process output
                        next_token_logits = logits[0, -1, :] / 0.8
                        next_token_logits[0] = -float("inf")
                        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                        generated = tokens + [next_token]
                        
                    except Exception as forward_error:
                        print(f"[Step {step}] Forward pass error: {forward_error}")
                        import traceback
                        traceback.print_exc()
                        return
                
                print(f"[Step {step}] Generation completed. Time: {time.time() - start_time:.3f}s")
                
                # Decode and print result
                try:
                    print(f"[Step {step}] Starting decoding...")
                    decoded = self.tokenizer.decode(generated)
                    print(f"[Step {step}] Generated: {decoded}")
                    
                    # Log to wandb if available
                    if wandb.run is not None:
                        print(f"[Step {step}] Logging to wandb...")
                        original_prompt_token_count = len(tokens)
                        newly_generated_tokens = generated[original_prompt_token_count:]
                        generated_text_only = self.tokenizer.decode(newly_generated_tokens)
                        
                        wandb.log({
                            f"inference/step_{step}/prompt": prompt,
                            f"inference/step_{step}/generated": generated_text_only,
                            f"inference/step_{step}/full_text": decoded,
                            f"inference/step_{step}/generation_time": time.time() - start_time
                        }, step=step)
                        print(f"[Step {step}] Wandb logging completed")
                        
                except Exception as decode_error:
                    print(f"[Step {step}] Decoding error: {decode_error}")
                    print(f"[Step {step}] Generated {len(generated)} tokens")
                            
        except Exception as e:
            print(f"[Step {step}] CRITICAL: Inference error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always return model to train mode
            print(f"[Step {step}] Setting model back to train mode...")
            try:
                actual_model = self.trainer.train_module.model
                actual_model.train()
            except:
                print(f"[Step {step}] Could not access model to set train mode")
            total_time = time.time() - start_time
            print(f"[Step {step}] ========== INFERENCE COMPLETE in {total_time:.3f}s ==========")

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