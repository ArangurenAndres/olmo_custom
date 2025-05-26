import torch
from transformers import AutoTokenizer
from olmo_core.train.callbacks import Callback
import wandb

class InferenceCallback(Callback):
    def __init__(self, model, tokenizer_config, prompts, interval, inference_mode="cycle"):
        self.model = model
        self.tokenizer_config = tokenizer_config
        self.prompts = prompts if isinstance(prompts, list) else [prompts]  # Support both single prompt and list
        self.interval = interval
        self.inference_mode = inference_mode  # "cycle", "all", or "random"
        self.current_prompt_idx = 0  # For cycling through prompts
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")

    def pre_train(self):
        self.run_inference(0)

    def post_step(self):
        if self.trainer.global_step % self.interval == 0:
            self.run_inference(self.trainer.global_step)

    def run_inference(self, step):
        self.model.eval()
        
        if self.inference_mode == "all":
            # Run inference on all prompts
            for i, prompt in enumerate(self.prompts):
                self._run_single_inference(prompt, step, i)
        elif self.inference_mode == "random":
            # Pick a random prompt
            import random
            prompt_idx = random.randint(0, len(self.prompts) - 1)
            prompt = self.prompts[prompt_idx]
            self._run_single_inference(prompt, step, prompt_idx)

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
