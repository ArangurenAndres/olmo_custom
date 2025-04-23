import os
import torch
import yaml
import mlflow

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


class MLflowLossCallback(Callback):
    priority = 10  # Required for OLMo callback sorting

    def post_step(self):
        if self.trainer and hasattr(self.trainer.train_module, "loss"):
            loss = self.trainer.train_module.loss
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            mlflow.log_metric("loss", loss, step=self.trainer.global_step)


def run(config):
    seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    total_tokens = config["steps"] * config["batch_size"] * config["sequence_length"]
    total_sequences = int(total_tokens * 1.1 / config["sequence_length"])

    dataloader, tokenizer_config = prepare_data(config["data_dir"], total_sequences)

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

    # Callbacks
    inference_cb = InferenceCallback(
        model=model,
        tokenizer_config=tokenizer_config,
        prompt=config["inference_prompt"],
        interval=config["inference_interval"]
    )
    mlflow_cb = MLflowLossCallback()

    trainer_config = TrainerConfig(
        save_folder=save_dir,
        save_overwrite=True,
        work_dir=work_dir,
        metrics_collect_interval=1,
        cancel_check_interval=5,
        max_duration=Duration.steps(config["steps"]),
        device=str(device),
    ).with_callback("inference", inference_cb
    ).with_callback("mlflow_loss", mlflow_cb)

    trainer = trainer_config.build(train_module=train_module, data_loader=dataloader)

    # 🚀 Start MLflow run
    with mlflow.start_run(run_name="olmo_local_run"):
        mlflow.log_params({
            "steps": config["steps"],
            "batch_size": config["batch_size"],
            "sequence_length": config["sequence_length"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "betas": config["betas"],
            "inference_prompt": config["inference_prompt"]
        })

        print(f" Training for {config['steps']} steps on device: {device}")
        trainer.fit()
        print(" Training complete")


if __name__ == "__main__":
    config = load_config()
    run(config)
