import os
import pprint
import datetime
import torch
import yaml
import wandb  
import shutil
from olmo_core.train import TrainerConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from olmo_core.utils import seed_all

from utils.dataloader import prepare_data
from utils.model import build_model
from utils.inference import InferenceCallback
from olmo_core.train.callbacks import Callback


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class WandbLossCallback(Callback):
    """
    Logs loss to Weights & Biases after every step.
    """
    priority = 10

    def post_step(self):
        if self.trainer and hasattr(self.trainer.train_module, "loss"):
            loss = self.trainer.train_module.loss
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            wandb.log({"loss": loss, "step": self.trainer.global_step})


def run(config):
    seed_all(42)

    save_dir = os.path.join(config["data_dir"], "checkpoints")
    work_dir = os.path.join(config["data_dir"], "trainer_work_dir")

    if os.path.exists(save_dir):
        print(f"Deleting old checkpoint directory: {save_dir}")
    shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)


    device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device.index if device.index is not None else 0)
        print(f"Running on CUDA device: {torch.cuda.current_device()} ({device})")
    else:
        print(f"Running on {device}")

    # ======= Initialize wandb run =======
    wandb.init(project="olmo_training", config=config)

    # ======= Initialize wandb run =======
    wandb.init(project="olmo_training", config=config)

    # ======= Print the full config and device =======
    print("\n========== Training Configuration ==========")
    pprint.pprint(config)
    print(f"Device: {device}")
    print("=============================================\n")

    total_tokens = config["steps"] * config["batch_size"] * config["sequence_length"]
    total_sequences = int(total_tokens * 1.1 / config["sequence_length"])

    dataloader, tokenizer_config = prepare_data(
        data_dir=config["data_dir"],
        total_sequences=total_sequences,
        sequence_length=config["sequence_length"],
        use_small_dataset=config.get("use_small_dataset", True)
    )

    model, train_module = build_model(
        vocab_size=tokenizer_config.padded_vocab_size(),
        device=device,
        sequence_length=config["sequence_length"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=config["betas"]
    )

    save_dir = os.path.join(config["data_dir"], "checkpoints")
    work_dir = os.path.join(config["data_dir"], "trainer_work_dir")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # ======= Callbacks =======
    inference_cb = InferenceCallback(
        model=model,
        tokenizer_config=tokenizer_config,
        prompt=config["inference_prompt"],
        interval=config["inference_interval"]
    )
    wandb_cb = WandbLossCallback()

    trainer_config = TrainerConfig(
        save_folder=save_dir,
        save_overwrite=True,
        work_dir=work_dir,
        metrics_collect_interval=1,
        cancel_check_interval=5,
        max_duration=Duration.steps(config["steps"]),
        device=str(device),
    ).with_callback("inference", inference_cb
    ).with_callback("wandb_loss", wandb_cb)

    trainer = trainer_config.build(train_module=train_module, data_loader=dataloader)

    print(f"Training for {config['steps']} steps on device: {device}\n")
    trainer.fit()
    print("\n Training complete")

    # ===== Save final model checkpoint =====
    final_checkpoint_path = os.path.join(save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"\n Final model saved at: {final_checkpoint_path}")

    # ===== Finish wandb run =====
    wandb.finish()


if __name__ == "__main__":
    config = load_config()
    run(config)