import os
import pprint
import torch
import yaml
import mlflow # type: ignore

# Imports for distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP # Still needed for type checking
from torch.utils.data import Dataset 

from olmo_core.train import TrainerConfig # type: ignore
from olmo_core.train.common import Duration # type: ignore
from olmo_core.train.trainer import Trainer # type: ignore
from olmo_core.utils import seed_all # type: ignore
from olmo_core.data import NumpyDataLoaderConfig 

# Assuming model_ddp.py is in utils directory and contains get_model_and_train_module_config
from DL_project.olmo_custom.backup.model_ddp import get_model_and_train_module_config 
from DL_project.olmo_custom.backup.dataloader_ddp import prepare_data
from DL_project.olmo_custom.backup.inference_ddp import InferenceCallback 
from olmo_core.train.callbacks import Callback # type: ignore


def load_config(path="config.yaml"):
    """Loads a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def initialize_distributed_env():
    """Initializes the distributed environment if relevant environment variables are set."""
    if "LOCAL_RANK" in os.environ and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500") 
        init_method = f"tcp://{master_addr}:{master_port}"

        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=world_size,
            rank=global_rank
        )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_distributed = True
        if global_rank == 0:
            print(f"Initialized distributed training. World size: {world_size}, Backend: NCCL, Init: {init_method}")
        print(f"  Process Info: Global rank: {global_rank}, Local rank: {local_rank}, Device: {device}")
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
        if device.type == "cuda":
             try:
                torch.cuda.set_device(0)
                device = torch.device("cuda:0")
                print(f"Running on single CUDA device: {torch.cuda.current_device()} (explicitly set to index 0)")
             except Exception as e:
                print(f"Could not set CUDA device 0: {e}. PyTorch will use default CUDA device if available.")
        else:
            print("Running on CPU")
        is_distributed = False
    return is_distributed, global_rank, local_rank, world_size, device

def is_main_process(global_rank: int) -> bool:
    """Checks if the current process is the main process (rank 0)."""
    return global_rank == 0

class MLflowLossCallback(Callback):
    """Callback to log training loss to MLflow."""
    priority = 10

    def __init__(self, global_rank: int, is_distributed: bool):
        super().__init__()
        self.global_rank = global_rank
        self.is_distributed = is_distributed

    def post_step(self):
        """Called after each training step."""
        if is_main_process(self.global_rank):
            if self.trainer and hasattr(self.trainer.train_module, "loss"):
                loss_tensor = self.trainer.train_module.loss
                if isinstance(loss_tensor, torch.Tensor):
                    loss_val = loss_tensor.item()
                    mlflow.log_metric("loss", loss_val, step=self.trainer.global_step)


def run(config: dict):
    """Main function to set up and run the training process."""
    is_distributed, global_rank, local_rank, world_size, device = initialize_distributed_env()

    seed = config.get("seed", 42)
    seed_all(seed + global_rank) 

    if is_main_process(global_rank):
        print("\n========== Training Configuration ==========")
        pprint.pprint(config)
        print(f"Running on device: {device} (master process view)")
        print(f"Distributed training enabled: {is_distributed}, World size: {world_size}")
        print(f"Base seed: {seed}, Rank {global_rank} seed: {seed + global_rank}")
        print("=============================================\n")

    global_batch_size_in_sequences = config["batch_size"] * world_size
    global_batch_size_in_tokens = global_batch_size_in_sequences * config["sequence_length"]
    total_tokens_for_dataset_estimation = config["steps"] * global_batch_size_in_sequences * config["sequence_length"]
    total_sequences_for_dataset = int(total_tokens_for_dataset_estimation * config.get("dataset_oversample_factor", 1.1) / config["sequence_length"])

    if is_main_process(global_rank):
        print(f"Per-rank sequences: {config['batch_size']}")
        print(f"Global batch size in sequences: {global_batch_size_in_sequences}")
        print(f"Global batch size in tokens (for DataLoader): {global_batch_size_in_tokens}")
        print(f"Estimated total sequences for dataset preparation: {total_sequences_for_dataset}")

    dataset, tokenizer_config = prepare_data(
        data_dir=config["data_dir"],
        total_sequences=total_sequences_for_dataset,
        sequence_length=config["sequence_length"],
        global_rank=global_rank,
        world_size=world_size,
        use_small_dataset=config.get("use_small_dataset", True),
        tokenizer_name=config.get("tokenizer_name", "allenai/gpt-neox-olmo-dolma-v1_5")
    )

    if not isinstance(dataset, Dataset): # type: ignore
        raise TypeError(
            f"prepare_data must return a torch.utils.data.Dataset instance. Got {type(dataset)}"
        )

    loader_cfg = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size_in_tokens, 
        seed=seed,
        num_workers=config.get("num_workers", 0),
        work_dir=config.get("dataloader_work_dir"), 
        num_threads=config.get("dataloader_num_threads"), 
        prefetch_factor=config.get("dataloader_prefetch_factor"), 
        target_device_type=config.get("dataloader_target_device_type")
    )
    dataloader = loader_cfg.build(dataset)

    if is_main_process(global_rank):
        print(f"DataLoader built using OLMo's NumpyDataLoaderConfig. Type: {type(dataloader)}")
        if hasattr(dataloader, 'dp_world_size') and hasattr(dataloader, 'dp_rank'):
            print(f"  DataLoader dp_world_size: {dataloader.dp_world_size}, dp_rank: {dataloader.dp_rank}") # type: ignore
            if is_distributed and dataloader.dp_world_size != world_size: # type: ignore
                 print(f"  Warning: DataLoader dp_world_size ({dataloader.dp_world_size}) " # type: ignore
                       f"does not match torch.distributed world_size ({world_size}).")
        else:
            print("  Warning: Built DataLoader does NOT have dp_world_size/dp_rank attributes. This may cause issues with OLMo Trainer.")

    
    # --- Model Building ---
    # Get the raw model and train module config.
    # Pass is_distributed flag so model_ddp.py can set dp_config for OLMo.
    raw_model, train_module_config = get_model_and_train_module_config(
        vocab_size=tokenizer_config.padded_vocab_size(),
        device=device, 
        sequence_length=config["sequence_length"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=config["betas"],
        batch_size_per_rank=config["batch_size"], 
        is_distributed=is_distributed, # MODIFIED: Pass the flag here
        n_kv_heads_gqa=config.get("n_kv_heads")
    )

    # DO NOT manually wrap with DDP. OLMo will handle it via dp_config passed to TransformerTrainModuleConfig.
    model_to_pass_to_train_module = raw_model # MODIFIED: Always pass the raw model
    if is_main_process(global_rank):
        print(f"Building TransformerTrainModule with raw model (type: {type(model_to_pass_to_train_module)}) on device {device}.")
        if is_distributed and train_module_config.dp_config is not None: # Check if dp_config was set
            print(f"  OLMo's TransformerTrainModule is configured for DDP: {train_module_config.dp_config.name}")
        elif is_distributed:
            print("  Warning: OLMo's TransformerTrainModule is NOT configured for DDP (dp_config is None). This is likely an error.")

    
    train_module = train_module_config.build(model=model_to_pass_to_train_module, device=device)
    
    if is_main_process(global_rank):
        # After train_module is built, its internal model should be DDP wrapped by OLMo's parallelize_model
        # if dp_config was correctly set and processed.
        print(f"TransformerTrainModule built. Model type in train_module: {type(train_module.model)}")
        if is_distributed:
            if isinstance(train_module.model, DDP):
                print("  Model in TrainModule is DDP wrapped by OLMo (as expected).")
            else:
                print(f"  Warning: Model in TrainModule is NOT DDP wrapped by OLMo. Type: {type(train_module.model)}. "
                      "Gradient synchronization might not occur if Trainer doesn't also wrap it.")


    save_dir = os.path.join(config["data_dir"], "checkpoints")
    work_dir = os.path.join(config["data_dir"], "trainer_work_dir")
    if is_main_process(global_rank):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)
    
    if is_distributed:
        dist.barrier()

    trainer_config_obj = TrainerConfig(
        save_folder=save_dir,
        save_overwrite=config.get("save_overwrite", True), 
        work_dir=work_dir,
        metrics_collect_interval=config.get("metrics_collect_interval", 1),
        cancel_check_interval=config.get("cancel_check_interval", 50),
        max_duration=Duration.steps(config["steps"]),
        device=str(device) 
    )

    if is_main_process(global_rank):
        inference_cb_instance = InferenceCallback(
            model=train_module.model, # InferenceCallback handles .module if model is DDP
            tokenizer_config=tokenizer_config,
            prompt=config["inference_prompt"],
            interval=config["inference_interval"],
            global_rank=global_rank,
            local_rank=local_rank,
            is_distributed=is_distributed
        )
        trainer_config_obj = trainer_config_obj.with_callback("inference", inference_cb_instance)
        
        mlflow_cb_instance = MLflowLossCallback(global_rank=global_rank, is_distributed=is_distributed)
        trainer_config_obj = trainer_config_obj.with_callback("mlflow_loss", mlflow_cb_instance)
    
    trainer = trainer_config_obj.build(
        train_module=train_module,
        data_loader=dataloader
    )
    if is_main_process(global_rank):
        print("OLMo Trainer built successfully.")
        if is_distributed:
            # After Trainer is built, check if train_module.model is DDP wrapped by the Trainer
            # This is the ultimate check for DDP status.
            if isinstance(trainer.train_module.model, DDP):
                print(f"  Model in Trainer's final train_module is DDP wrapped. Type: {type(trainer.train_module.model)}")
            else:
                print(f"  Warning: Model in Trainer's final train_module is NOT DDP wrapped. Type: {type(trainer.train_module.model)}. "
                      "Gradient synchronization will NOT occur.")


    active_mlflow_run = None
    if is_main_process(global_rank):
        print(f"Starting MLflow run (on rank {global_rank})...")
        if config.get("clear_stale_checkpoints_on_start", True) and os.path.exists(save_dir): 
            if any(os.scandir(save_dir)): 
                print(f"Clearing stale checkpoints from {save_dir} as per configuration...")
                import shutil
                for item in os.listdir(save_dir):
                    item_path = os.path.join(save_dir, item)
                    if os.path.isdir(item_path): 
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            else:
                print(f"Checkpoint directory {save_dir} is empty or does not exist. No stale checkpoints to clear.")

        active_mlflow_run = mlflow.start_run(run_name=config.get("mlflow_run_name", "olmo_run"))
        mlflow.log_params(config)
        mlflow.log_param("device_type_used", device.type)
        mlflow.log_param("is_distributed_run", is_distributed)
        if is_distributed:
            mlflow.log_param("world_size_ddp", world_size)

    try:
        if is_main_process(global_rank):
            print(f"Starting training for {config['steps']} global steps on device: {device} (master view)\n")
        
        trainer.fit()

        if is_main_process(global_rank):
            print("\n✅ Training complete (master process)")
        else:
            print(f"\n✅ Training complete (rank {global_rank})")

    except Exception as e:
        if is_main_process(global_rank):
            print(f"\n❌ Training failed (master process): {e}")
        else:
            print(f"\n❌ Training failed (rank {global_rank}): {e}")
        raise
    finally:
        if is_main_process(global_rank) and active_mlflow_run:
            mlflow.end_run()
            print("MLflow run ended.")
        if is_distributed:
            dist.destroy_process_group()
            print(f"Rank {global_rank}: Destroyed process group.")


if __name__ == "__main__":
    training_config = load_config()
    # Forcing checkpoint clearing during these debugging iterations.
    training_config["clear_stale_checkpoints_on_start"] = True 
    run(training_config)
