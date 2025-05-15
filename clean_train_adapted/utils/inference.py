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

    def pre_train(self):
        self.run_inference(0)

    def post_step(self):
        if self.trainer.global_step % self.interval == 0:
            self.run_inference(self.trainer.global_step)

    def run_inference(self, step):
        self.model.eval()
        tokens = [t for t in self.tokenizer.encode(self.prompt) if t != 0]
        input_tensor = torch.tensor([tokens], device=self.model.device)
        generated = tokens.copy()

        with torch.no_grad():
            logits = self.model(input_tensor)
            for _ in range(50):
                logits = logits[0, -1, :] / 0.8
                logits[0] = -float("inf")
                probs = torch.nn.functional.softmax(logits, dim=-1)
                token = torch.multinomial(probs, 1).item()
                if token == self.tokenizer_config.eos_token_id:
                    break
                generated.append(token)
                input_tensor = torch.tensor([generated], device=self.model.device)
                logits = self.model(input_tensor)

        decoded = self.tokenizer.decode(generated)
        print(f"[Step {step}] Generated: {decoded}")
        self.model.train()
