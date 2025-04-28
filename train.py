import os
import pprint
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

    # Determine the base device type
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    if device.type == "cuda":
        torch.cuda.set_device(0)      
        print(f"Running on CUDA device: {torch.cuda.current_device()} (set explicitly to index 0)") # Optional: confirmation
    else:
        print("Running on CPU")

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
    with mlflow.start_run(run_name="olmo_local_run"):
        # Log all config parameters automatically
        mlflow.log_params(config)
        mlflow.log_param("device", str(device))  # Also log the device (cuda/cpu)

        print(f"Training for {config['steps']} steps on device: {device}\n")
        trainer.fit()
        print("\n✅ Training complete")


if __name__ == "__main__":
    config = load_config()
    run(config)
