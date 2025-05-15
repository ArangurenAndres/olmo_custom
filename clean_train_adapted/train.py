"""
Main training script for OLMo models.

This script coordinates the training process by importing and using modules for:
1. Configuration parsing
2. Dataset loading from pre-prepared data
3. Model building
4. Training execution

Total tokens trained = batch_size * steps * sequence_length 
- batch size is dependent on the GPU memory (higher always better)
- steps is dependent on the number of tokens to train
- sequence length is fixed at 1024

"""

import os
import torch
import wandb
import pprint
import time
import yaml
import shutil

from utils.model import build_model
from data_utils.load_dataset import load_prepared_dataset
from olmo_core.data import NumpyDataLoaderConfig
from utils.token_0_handling import apply_special_token_handling
from olmo_core.train import TrainerConfig
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from olmo_core.utils import seed_all
from utils.dataloader import prepare_data
from utils.inference import InferenceCallback
from olmo_core.train.callbacks import Callback, WandBCallback
from olmo_core.data import TokenizerConfig
from utils.setup_env_variables import setup_environment



config_file_name = "config.yaml"

def load_config(path=config_file_name):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Parse command line arguments
    config = load_config()

    seed_all(config["seed"])

    # setup environment variables
    setup_environment()

    # SETUP DATA DIRECTORIES and checkpoints dir, work dir
    timestamp = time.strftime('%m-%d_%H-%M')
    run_dir = os.path.join(config["data_dir"], f"run_{timestamp}")
    save_dir = os.path.join(run_dir, "checkpoints")
    work_dir = os.path.join(run_dir, "trainer_work_dir")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True) 
    os.makedirs(work_dir, exist_ok=True)
    

    # Copy config file to checkpoints folder 
    try:
        shutil.copy2(os.path.join(os.path.dirname(__file__), config_file_name), os.path.join(save_dir, config_file_name))
    except Exception as e:
        print(f"Error copying config file: {e}")
    
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)      
        # Disable synchronization debug mode - usually appears when inference is called - relates to efficiency
        torch.cuda.set_sync_debug_mode(False) 
        print(f"Running on CUDA device: {torch.cuda.current_device()} (set explicitly to index 0)") # Optional: confirmation
    else:
        print("Running on CPU")
    
    
    # ======= Print the full config and device =======
    print("\n========== Training Configuration ==========")
    pprint.pprint(config)
    print(f"Device: {device}")
    print("=============================================\n")

    # Load pre-prepared dataset + tokenizer config 
    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    dataset = load_prepared_dataset(config)



    # Configure data loader
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=config["batch_size"] * config["sequence_length"],
        seed=42,
        num_workers=8,
    )

    # Build data loader
    data_loader = data_loader_config.build(dataset)
    print("Data loader built successfully")
    
    
    model, train_module = build_model(
        vocab_size=tokenizer_config.padded_vocab_size(),
        device=device,
        config=config
    )

    
    # IP ADRESS PROBLEM SOLVED with setting the token ID 0 to zeros and logit bias to large negative
    apply_special_token_handling(model)
    

    
    # Initialize wandb and callback
    #wandb.init(
       # project=config["wandb_project"],
      #  name=f"{config['wandb_name']}-{timestamp}",
     #   config=config
    #)

    wandb_cb = WandBCallback(
        project=config["wandb_project"],
        name=config["wandb_name"],
        entity=None,
        enabled=True,
        cancel_check_interval=10,
        config=config
    )
    inference_cb = InferenceCallback(
        model=model,
        tokenizer_config=tokenizer_config,
        prompt=config["inference_prompt"],
        interval=config["inference_interval"]
    )

    # Configure checkpointer with the desired save interval
    checkpointer_cfg = CheckpointerConfig(
        save_interval=Duration.steps(config["save_interval_steps"])
    )

    trainer_config = TrainerConfig(
        save_folder=save_dir,
        save_overwrite=True,
        work_dir=work_dir,
        checkpointer=checkpointer_cfg,
        metrics_collect_interval=1,
        cancel_check_interval=5,
        max_duration=Duration.steps(config["steps"]),
        device=str(device),
    ).with_callback("wandb", wandb_cb
    ).with_callback("inference", inference_cb)
    

    trainer = trainer_config.build(train_module=train_module, data_loader=data_loader)
   
    print(f"Training for {config['steps']} steps on device: {device}\n")
    trainer.fit()
    print("\n✅ Training complete")
    
    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main() 