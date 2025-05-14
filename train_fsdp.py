import os
import pprint
import torch
import yaml
import mlflow
import argparse

from olmo_core.distributed import utils # User's correct import
from olmo_core.train.config import TrainerConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from olmo_core.utils import seed_all
# from olmo_core.utils import log_extra_hyperparameters # User commented this out

from utils.dataloader_fsdp import prepare_data
from utils.model_fsdp import build_model
from utils.inference import InferenceCallback
from olmo_core.train.callbacks import Callback
import torch.distributed as dist


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class MLflowLossCallback(Callback):
    priority = 10

    def post_train_step(self, trainer, step_output):
        #highlight-start
        if hasattr(self, "trainer") and self.trainer.dist.get_rank() == 0: # MODIFIED: get_rank()
        #highlight-end
            loss_val = None
            if step_output is not None and "loss" in step_output:
                loss_val = step_output["loss"]
            elif hasattr(self.trainer.train_module, "last_loss"):
                loss_val = self.trainer.train_module.last_loss
            
            if loss_val is not None:
                if isinstance(loss_val, torch.Tensor):
                    loss_val = loss_val.item()
                mlflow.log_metric("loss", loss_val, step=self.trainer.global_step)


def run(config, args):
    # Initialize distributed world using the function from olmo_core.distributed.utils
    # This relies on torchrun setting RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    #highlight-start
    utils.init_distributed(backend="nccl") # Specify backend, e.g., "nccl" for GPUs

    # Get rank information using the correct functions
    global_rank = utils.get_rank()         # MODIFIED: This is the main fix for the TypeError
    local_rank = utils.get_local_rank()
    world_size = utils.get_world_size()
    #highlight-end

    device_id = local_rank

    seed_all(config.get("seed", 42) + global_rank)

    if torch.cuda.is_available() and world_size > 0:
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        if global_rank == 0:
             print(f"Rank {global_rank}/{world_size} (local_rank {local_rank} on this node): Running on CUDA device: {device}")
    else:
        if global_rank == 0:
            print("Running on CPU (FSDP not applicable or CUDA not available/specified for distributed training)")
        device = torch.device("cpu")

    if global_rank == 0:
        print("\n========== Training Configuration ==========")
        pprint.pprint(config)
        print(f"World size: {world_size}")
        print(f"Effective device for rank 0: {device}")
        print("=============================================\n")

    global_batch_size = config["global_batch_size"]
    sequence_length = config["sequence_length"]
    num_gradient_steps = args.steps if args.steps is not None else config["steps"]
    total_sequences_to_materialize = config.get(
        "total_sequences_to_prepare",
        int(num_gradient_steps * global_batch_size * 1.1)
    )

    dataloader, tokenizer_config = prepare_data(
        data_dir=config["data_dir"],
        total_sequences_to_prepare=total_sequences_to_materialize,
        sequence_length=sequence_length,
        actual_global_batch_size=global_batch_size,
        use_small_dataset=config.get("use_small_dataset", True)
    )

    model_dtype_str = config.get("model_dtype", "bf16" if device.type == "cuda" else "fp32")

    model, train_module = build_model(
        vocab_size=tokenizer_config.padded_vocab_size(),
        device=device,
        sequence_length=sequence_length,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=config["betas"],
        # n_kv_heads=config.get("n_kv_heads"),
        # model_dtype_str=model_dtype_str,
        # fnn_scalars=config.get("fnn_scalars"), # Ensure build_model handles these
        # qkv_scalars=config.get("qkv_scalars")  # Ensure build_model handles these
    )

    save_dir = os.path.join(config["data_dir"], "checkpoints")
    work_dir = os.path.join(config["data_dir"], "trainer_work_dir")
    
    if global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)
    
    if world_size > 1:
        #highlight-start
        utils.barrier() # MODIFIED: Use utils.barrier
        #highlight-end

    inference_prompt = args.prompt if args.prompt is not None else config["inference_prompt"]
    inference_cb = InferenceCallback(
        model=model,
        tokenizer_config=tokenizer_config,
        prompt=inference_prompt,
        interval=config["inference_interval"]
    )
    mlflow_cb = MLflowLossCallback()

    trainer_config_params = {
        "save_folder": save_dir,
        "save_overwrite": True,
        "work_dir": work_dir,
        # "metrics_collect_interval": 1,
        "cancel_check_interval": 5,
        "max_duration": Duration.steps(num_gradient_steps),
        # "precision": model_dtype_str,
        # "global_train_batch_size": global_batch_size,
    }
    # if "gradient_accumulation_steps" in config:
    #     trainer_config_params["gradient_accumulation_steps"] = config["gradient_accumulation_steps"]

    trainer_config = TrainerConfig(**trainer_config_params
    ).with_callback("inference", inference_cb
    ).with_callback("mlflow_loss", mlflow_cb)

    trainer = trainer_config.build(train_module=train_module, data_loader=dataloader)
    
    active_mlflow_run = None  # Variable to store the active run object for rank 0
    if global_rank == 0:
        print(f"Starting training for {num_gradient_steps} global steps on world_size {world_size}...")
        mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", None))
        mlflow.set_experiment(config.get("mlflow_experiment_name", "OLMo_FSDP_Custom_Model"))

        # Start MLflow run only on rank 0
        active_mlflow_run = mlflow.start_run(run_name=config.get("mlflow_run_name", "olmo_fsdp_run"))
        if active_mlflow_run:
            mlflow.log_params(config)
            # log_extra_hyperparameters was commented out by you
            mlflow.log_param("world_size", world_size)
            mlflow.log_param("effective_num_steps", num_gradient_steps)
            mlflow.log_param("effective_inference_prompt", inference_prompt)
        else:
            # It's good practice to handle the case where MLflow might fail to start a run
            print(f"Warning: MLflow run could not be started on rank {global_rank}.")

    # trainer.fit() MUST be called by ALL ranks
    trainer.fit()

    if global_rank == 0:
        print(f"\n✅ Training complete on rank {global_rank}.")
        if active_mlflow_run:  # Only end the run if it was successfully started by rank 0
            mlflow.end_run()
            print(f"MLflow run ended on rank {global_rank}.")
        else:
            print(f"No active MLflow run to end on rank {global_rank}.")

    if world_size > 1:
        utils.barrier() # Ensure all ranks complete before exiting run function potentially

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--steps", type=int, help="Number of training steps, overrides config if set")
    parser.add_argument("--prompt", type=str, help="Inference prompt, overrides config if set")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        run(config, args)
    finally:
        if dist.is_initialized(): # Check if distributed process group is initialized
            dist.destroy_process_group()
            print("Successfully destroyed process group.")