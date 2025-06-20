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

    def _run_single_inference(self, prompt, step, prompt_idx, max_new_tokens):
        """
        Run inference for a single prompt. This logic is factored out so that we can
        easily iterate over *all* prompts when ``self.inference_mode == "all"``.

        Parameters
        ----------
        prompt : str
            The text prompt to feed into the model.
        step : int
            The current global training step (used for logging).
        prompt_idx : int
            Index of the prompt within ``self.prompts`` (used for nice logging
            and WandB key grouping).
        max_new_tokens : int
            Number of new tokens to generate.
        """

        rank = get_rank() if is_distributed() else 0

        # Retrieve the wrapped model that is being trained
        actual_model = self.trainer.train_module.model
        is_fsdp = hasattr(actual_model, '_fsdp_enabled') or 'FSDP' in str(type(actual_model))

        # All ranks enter eval so that layers such as dropout are disabled
        actual_model.eval()

        try:
            input_tensor = None
            tokens = []

            # ------------------------------------------------------------------
            # 1. Rank-0 prepares the prompt and broadcasts it to the other ranks
            # ------------------------------------------------------------------
            if rank == 0:
                if self.tokenizer is None:
                    print("Tokenizer not available on rank 0. Skipping inference.")
                    if is_distributed():
                        # Signal an error with a negative size so other ranks can bail
                        dist.broadcast(torch.tensor([-1], device=self.trainer.device), src=0)
                    return

                # Encode the prompt.  We filter out the BOS (token id 0) that
                # the NeoX tokenizer may prepend so that we do not double count
                # it when generating.
                tokens = [t for t in self.tokenizer.encode(prompt) if t != 0]

            if is_distributed():
                # Broadcast the length so that all ranks can size their tensor
                size_tensor = torch.tensor([len(tokens) if rank == 0 else 0], dtype=torch.long, device=self.trainer.device)
                dist.broadcast(size_tensor, src=0)
                synced_size = size_tensor.item()

                # Bail out early if rank-0 signalled an error
                if synced_size == -1:
                    return

                # Allocate and broadcast the actual prompt tokens
                if rank == 0:
                    input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.trainer.device)
                else:
                    input_tensor = torch.empty((1, synced_size), dtype=torch.long, device=self.trainer.device)
                dist.broadcast(input_tensor, src=0)
            else:
                # Single-GPU / CPU case
                if self.tokenizer:
                    input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.trainer.device)

            # Sanity check
            if input_tensor is None:
                return

            # We will grow this list autoregressively
            generated_tokens = tokens.copy()

            # ------------------------------------------------------------------
            # 2. Autoregressive generation loop (done on every rank for safety)
            # ------------------------------------------------------------------
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    if is_fsdp:
                        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                        with FSDP.summon_full_params(actual_model, recurse=True, offload_to_cpu=False):
                            logits = actual_model(input_tensor)
                    else:
                        logits = actual_model(input_tensor)

                if rank == 0:
                    next_token_logits = logits[0, -1, :] / 0.8  # temperature = 0.8
                    next_token_logits[0] = -float("inf")  # suppress <|endoftext|> / special token
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Append and build the new input tensor for the next step
                    generated_tokens.extend(next_token.tolist())
                    input_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=self.trainer.device)

            # ------------------------------------------------------------------
            # 3. Rank-0 pretty printing and WandB logging
            # ------------------------------------------------------------------
            if rank == 0:
                print(f"Inference for prompt {prompt_idx + 1} at step {step} successful.")
                decoded_full = self.tokenizer.decode(generated_tokens)
                print(f"[Step {step}] Prompt {prompt_idx + 1} => {decoded_full}")

                if wandb.run is not None:
                    prompt_token_count = len(tokens)
                    generated_only = generated_tokens[prompt_token_count:]
                    generated_text_only = self.tokenizer.decode(generated_only)

                    wandb.log(
                        {
                            f"inference/prompt_{prompt_idx + 1}/prompt_text": prompt,
                            f"inference/prompt_{prompt_idx + 1}/generated_text": wandb.Html(
                                f"<p><strong>Prompt:</strong> {prompt}</p><p><strong>Generated:</strong> {generated_text_only}</p>"
                            ),
                        },
                        step=step,
                    )

        finally:
            # Return the model to training mode and synchronise all ranks
            actual_model.train()
            if is_distributed():
                dist.barrier()

    def run_inference(self, step, max_new_tokens):
        # Handle the different inference modes and delegate to the helper above.
        if self.inference_mode == "all":
            # Run inference on *all* prompts sequentially.
            for i, prompt in enumerate(self.prompts):
                self._run_single_inference(prompt, step, i, max_new_tokens)
        else:
            # Determine which prompt index to use given the chosen mode.
            if self.inference_mode == "cycle":
                prompt_idx = step % len(self.prompts)
            elif self.inference_mode == "random":
                prompt_idx = random.randrange(len(self.prompts))
            else:  # default behaviour is to use the first prompt
                prompt_idx = 0

            prompt = self.prompts[prompt_idx]
            self._run_single_inference(prompt, step, prompt_idx, max_new_tokens)
