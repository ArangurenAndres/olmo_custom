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

from utils.model import build_model, build_train_module_with_fsdp
from datetime import timedelta
from data_utils.load_dataset import load_prepared_dataset
from olmo_core.data import NumpyDataLoaderConfig
from utils.token_0_handling import apply_special_token_handling
from olmo_core.train import TrainerConfig
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.trainer import Trainer
from olmo_core.utils import seed_all
from utils.dataloader import prepare_data, create_distributed_dataloader
from utils.inference import InferenceCallback
from olmo_core.train.callbacks import Callback, WandBCallback, DownstreamEvaluatorCallbackConfig
from olmo_core.data import TokenizerConfig
from utils.setup_env_variables import setup_environment

from olmo_core.train.callbacks.evaluator_callback import LMEvaluatorCallbackConfig
from olmo_core.data import NumpyDatasetConfig, NumpyDatasetType
from utils.load_config import load_config

from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.distributed.utils import is_distributed, get_rank, get_world_size
import torch.distributed as dist

def main():
    prepare_training_environment(
        seed=42,
        backend="cpu:gloo,cuda:nccl",
        timeout=timedelta(minutes=30)
    )
    try:
        config = load_config()
        seed_all(config["seed"])
        setup_environment()

        # SETUP DATA DIRECTORIES and checkpoints dir, work dir
        timestamp = time.strftime('%m-%d_%H-%M')
        run_dir = os.path.join(config["data_dir"], "checkpoints", f"run_{timestamp}")
        # save_dir = os.path.join("/scratch-shared/tmp.GcVy0pChFL", f"checkpoints_{timestamp}")
        save_dir = os.path.join("train_run", f"checkpoints_{timestamp}")
        work_dir = os.path.join(run_dir, "trainer_work_dir")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True) 
        os.makedirs(work_dir, exist_ok=True)

        loaded_config_name = config.get('_args_config_name', 'config') 
        
        # Set device - Remove explicit device setting for distributed training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            # Don't set specific device in distributed training
            if not is_distributed():
                torch.cuda.set_device(0)
            torch.cuda.set_sync_debug_mode(0) 
            current_device = torch.cuda.current_device() if torch.cuda.is_available() else "N/A"
            print(f"Running on CUDA device: {current_device}")
        else:
            print("Running on CPU")
        
        print("\n========== Training Configuration ==========")
        pprint.pprint(config)
        print(f"Device: {device}")
        if is_distributed():
            print(f"Distributed training: Rank {get_rank()}/{get_world_size()}")
        print("=============================================\n")

        # Load tokenizer and dataset
        tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
        dataset = load_prepared_dataset(config)

        # Build distributed dataloader
        dataloader = create_distributed_dataloader(dataset, config)
        print("Distributed data loader built successfully")
        
        # Build model with distributed awareness
        model, _ = build_model(
            vocab_size=tokenizer_config.padded_vocab_size(),
            device=device,
            config=config
        )
        
        # Apply special token handling
        apply_special_token_handling(model)
        
        # Build train module with FSDP
        train_module = build_train_module_with_fsdp(model, config)
        
        # Initialize wandb only on rank 0
        if not is_distributed() or get_rank() == 0:
            wandb.init(
                project=config["wandb_project"],
                name=f"{config['wandb_name']}-{timestamp}",
                config=config
            )

        # Add distributed barrier AFTER all initialization
        if is_distributed():
            print(f"Rank {get_rank()}: Finished initialization, synchronizing...")
            dist.barrier()
            print(f"Rank {get_rank()}: All ranks synchronized after initialization")

        # Get inference prompts and mode from config
        inference_prompts = config.get("inference_prompts", 
                                     [config.get("inference_prompt", "Hello world")])
        inference_mode = config.get("inference_mode", "all")

        downstream_eval_tasks = [
            "arc_challenge", "arc_easy", "boolq", "commonsense_qa",
            "hellaswag", "openbook_qa", "piqa", "social_iqa", "sciq",
            "mmlu_stem", "basic_arithmetic", "gsm8k_gold_bpb_5shot"
        ]

        # Create callbacks AFTER variable definitions but BEFORE next barrier
        if not is_distributed() or get_rank() == 0:
            print(f"Rank {get_rank()}: Creating callbacks...")
            
            wandb_cb = WandBCallback(
                project=config["wandb_project"],
                name=f"{config['wandb_name']}-{timestamp}",
                entity=None,
                enabled=True,
                cancel_check_interval=10,
                config=config
            )
            print(f"Rank {get_rank()}: Succesfully created WandB Callback")
            inference_cb = InferenceCallback(
                model=model,
                tokenizer_config=tokenizer_config,
                prompts=inference_prompts,
                interval=config["steps"]/config["inference_times"],
                inference_mode=inference_mode,
                # skip_pre_train=is_distributed()
            )
            print(f"Rank {get_rank()}: Succesfully created Inference Callback")
            downstream_eval_cb_config = DownstreamEvaluatorCallbackConfig(
                tasks=downstream_eval_tasks,
                tokenizer=tokenizer_config,
                eval_interval=config["steps"]/config["evaluation_times"],
                eval_on_startup=False,
                log_interval=5,
                enabled=True
            )
            print(f"Rank {get_rank()}: Succesfully created Eval Callback")
            # lm_eval_callback_config = None  # Temporarily disabled
            
            print(f"Rank {get_rank()}: Callbacks created successfully")
        else:
            print(f"Rank {get_rank()}: Skipping callback creation (non-zero rank)")
            wandb_cb = None
            inference_cb = None
            downstream_eval_cb_config = None
            lm_eval_callback_config = None



        # Evaluation tasks (only on rank 0)


        # # Create inference callback (only on rank 0) - DISABLE pre_train inference for distributed
        # if not is_distributed() or get_rank() == 0:
        #     inference_cb = InferenceCallback(
        #         model=model,
        #         tokenizer_config=tokenizer_config,
        #         prompts=inference_prompts,
        #         interval=config["steps"]/config["inference_times"],
        #         inference_mode=inference_mode,
        #         skip_pre_train=is_distributed()  # Skip pre_train inference in distributed mode
        #     )
        # else:
        #     inference_cb = None

        # # Temporarily disable downstream evaluation to isolate the issue
        # if not is_distributed() or get_rank() == 0:
        #     # Disable evaluation during debugging
            
        #     # Original evaluation config (commented out for debugging)
        #     downstream_eval_cb_config = DownstreamEvaluatorCallbackConfig(
        #         tasks=downstream_eval_tasks,
        #         tokenizer=tokenizer_config,
        #         eval_interval=config["steps"]/config["evaluation_times"],
        #         eval_on_startup=False,  # Disable eval_on_startup to avoid blocking
        #         log_interval=5,
        #         enabled=True
        #     )
        # else:
        #     downstream_eval_cb_config = None
        #     lm_eval_callback_config = None
    
        # Add barrier AFTER callback creation to synchronize all ranks
        if is_distributed():
            print(f"Rank {get_rank()}: About to synchronize after callback creation")
            dist.barrier()
            print(f"Rank {get_rank()}: All ranks synchronized after callback creation")

        # Configure trainer
        trainer_config = TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            work_dir=work_dir,
            load_strategy=LoadStrategy.never,
            metrics_collect_interval=1,
            cancel_check_interval=5,
            max_duration=Duration.steps(config["steps"]),
            device=str(device) if not is_distributed() else None,  # Let FSDP handle device
        )

        # Add callbacks only on rank 0
        if wandb_cb:
            trainer_config = trainer_config.with_callback("wandb", wandb_cb)
            print(f"Rank {get_rank()}: Succesfully added WandB Callback")
        if inference_cb:
            trainer_config = trainer_config.with_callback("inference", inference_cb)
            print(f"Rank {get_rank()}: Succesfully added Inference Callback")
        if downstream_eval_cb_config:
            trainer_config = trainer_config.with_callback("downstream_eval", downstream_eval_cb_config)
            print(f"Rank {get_rank()}: Succesfully added Downstream Eval Callback")
        if lm_eval_callback_config:
            trainer_config = trainer_config.with_callback("lm_evaluator", lm_eval_callback_config)
            print(f"Rank {get_rank()}: Succesfully added LM Eval Callback")
        
        # Enhanced logging before trainer build
        if is_distributed():
            print(f"Rank {get_rank()}: About to synchronize before trainer build")
            dist.barrier()
            print(f"Rank {get_rank()}: Synchronized, building trainer")
        else:
            print("Single process: building trainer")
            
        # Build trainer
        trainer = trainer_config.build(train_module=train_module, data_loader=dataloader)
    
        print(f"Training for {config['steps']} steps on device: {device}\n")
        
        # Enhanced logging before training starts
        if is_distributed():
            print(f"Rank {get_rank()}: About to start training with {len(trainer.callbacks)} callbacks")
            dist.barrier()
            print(f"Rank {get_rank()}: All ranks ready, starting training")
        else:
            print("Single process: starting training")

        # Start training with error handling
        try:
            trainer.fit()
            print("\n✅ Training complete")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            if is_distributed():
                print(f"Rank {get_rank()}: Training error occurred")
            raise
        
        # Close wandb only on rank 0
        if not is_distributed() or get_rank() == 0:
            wandb.finish()

    finally:
        teardown_training_environment()
        print("Training environment torn down successfully")

if __name__ == "__main__":
    main() 