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
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")

    def pre_train_loop(self, trainer): # Or pre_train
        #highlight-start
        if hasattr(self, "trainer") and self.trainer.dist.get_rank() == 0: # MODIFIED: get_rank()
            print(f"Initial inference check on rank {self.trainer.dist.get_rank()}:") # MODIFIED: get_rank()
            self.run_inference(0)
        if hasattr(self, "trainer") and self.trainer.dist.get_world_size() > 1:
            self.trainer.dist.barrier()
        #highlight-end

    def post_train_step(self, trainer, step_output): # Or post_step
        if self.trainer.global_step % self.interval == 0:
            self.model.eval()
            #highlight-start
            if self.trainer.dist.get_rank() == 0: # MODIFIED: get_rank()
                self.run_inference(self.trainer.global_step)

            if self.trainer.dist.get_world_size() > 1:
                self.trainer.dist.barrier()
            #highlight-end
            self.model.train()

    def run_inference(self, step):
        tokens = [t for t in self.tokenizer.encode(self.prompt) if t != 0]
        input_tensor = torch.tensor([tokens], device=self.model.device)
        generated_tokens = tokens.copy()

        with torch.no_grad():
            output = self.model(input_ids=input_tensor)
            logits = output.logits

            for _ in range(50):
                next_token_logits = logits[0, -1, :]
                next_token_logits = next_token_logits / 0.8
                if next_token_logits.size(0) > 0:
                    next_token_logits[0] = -float("inf")
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                if next_token_id == self.tokenizer_config.eos_token_id:
                    break
                generated_tokens.append(next_token_id)
                input_tensor = torch.tensor([generated_tokens], device=self.model.device)
                output = self.model(input_ids=input_tensor)
                logits = output.logits

        decoded_text = self.tokenizer.decode(generated_tokens)
        #highlight-start
        current_rank = self.trainer.dist.get_rank() if hasattr(self, "trainer") else "N/A" # MODIFIED: get_rank()
        print(f"[Rank {current_rank} / Step {step}] Generated: {decoded_text}")
        #highlight-end