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
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.trainer import Trainer
from olmo_core.utils import seed_all
from utils.dataloader import prepare_data
from utils.inference import InferenceCallback
from olmo_core.train.callbacks import Callback, WandBCallback, DownstreamEvaluatorCallbackConfig
from olmo_core.data import TokenizerConfig
from utils.setup_env_variables import setup_environment

from olmo_core.train.callbacks.evaluator_callback import LMEvaluatorCallbackConfig
from olmo_core.data import NumpyDatasetConfig, NumpyDatasetType
from utils.load_config import load_config


def main():
    # Parse command line arguments (this will now be handled by load_config)
    config = load_config()

    seed_all(config["seed"])

    # setup environment variables
    setup_environment()

    # SETUP DATA DIRECTORIES and checkpoints dir, work dir
    timestamp = time.strftime('%m-%d_%H-%M')
    run_dir = os.path.join(config["data_dir"], "checkpoints", f"run_{timestamp}")
    save_dir = os.path.join("/scratch-shared/tmp.GcVy0pChFL", "checkpoints")
    work_dir = os.path.join(run_dir, "trainer_work_dir")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True) 
    os.makedirs(work_dir, exist_ok=True)
    

    # Copy config file to checkpoints folder 
    # This needs to be adjusted as we don't have a single config_file_name variable directly
    # We need to reconstruct the path of the loaded config or decide not to copy if it's complex.
    # For now, let's try to reconstruct it. This assumes the structure from load_config.
    loaded_config_name = config.get('_args_config_name', 'config') # We need to inject this into config if we want to know it here
                                                                   # Or, load_config could return it alongside the config dict.
                                                                   # Simpler for now: assume default or don't copy, or make load_config return path.
                                                                   # Let's assume we can derive it from the --config arg if that was passed.
                                                                   # The load_config now doesn't give us the name used. Let's skip copy for now, or refine later.
    # try:
    #     # This part is tricky because load_config() as written doesn't return the path used.
    #     # We'll need to modify load_config to return the path, or make an assumption.
    #     # For now, I'll comment out the copy, as it's not straightforward to get the source path here.
    #     # shutil.copy2(os.path.join(os.path.dirname(__file__), config_file_name), os.path.join(save_dir, config_file_name))
    #     print("Skipping config copy to save_dir as source path is not directly available from new loader.")
    # except Exception as e:
    #     print(f"Error copying config file: {e}")
    
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)      
        # Disable synchronization debug mode - usually appears when inference is called - relates to efficiency
        torch.cuda.set_sync_debug_mode(0) 
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
    wandb.init(
        project=config["wandb_project"],
        name=f"{config['wandb_name']}-{timestamp}",
        config=config
    )

    # TO DO: put all callbacks in utils/callbacks.py
    wandb_cb = WandBCallback(
        project=config["wandb_project"],
        name=f"{config['wandb_name']}-{timestamp}",
        entity=None,
        enabled=True,
        cancel_check_interval=10,
        config=config
    )
    inference_cb = InferenceCallback(
        model=model,
        tokenizer_config=tokenizer_config,
        prompt=config["inference_prompt"],
        interval=config["steps"]/config["inference_times"]
    )

    # Evaluation tasks CallBack
    # TODO: Consider moving task list, eval_interval, and eval_duration to config.yaml
    #["arc_challenge", "arc_easy", "boolq", "copa", "headqa_en", "hellaswag", "logiqa", "mathqa", "mrpc",
    #"openbookqa", "piqa", "qnli", "qqp", "rte", "sciq", "sst", "wic", "winogrande", "wnli", "wsc"];

    #downstream_eval_tasks = ["arc_challenge", "arc_easy", "boolq", "commonsense_qa",  #"hellaswag", "mmlu_stem", "openbook_qa", "piqa", "social_iqa"]

    downstream_eval_tasks = [
        # General Reasoning / QA / Commonsense (Good for most models)
        "arc_challenge",
        "arc_easy",
        "boolq",
        "commonsense_qa",
        "hellaswag",
        "openbook_qa",
        "piqa",
        "social_iqa",
        "sciq",
        # Math Specific
        "mmlu_stem",                     # Broad STEM, includes math
        # "mmlu_stem_mc_5shot",          # MMLU STEM, multiple choice, 5-shot
        # "mmlu_stem_val_mc_5shot",      # MMLU STEM, validation, multiple choice, 5-shot
        "basic_arithmetic",
         "gsm8k_gold_bpb_5shot",        # Grade School Math
        # Minerva Math (more advanced math)
      #   "minerva_math_prealgebra_gold_bpb_0shot",
      #   "minerva_math_precalculus_gold_bpb_0shot",
        # Code Specific
       # "codex_humaneval_gold_bpb_0shot",
       # "codex_mbpp_gold_bpb_0shot"
    ]
    #"mmlu_mc_std_elementary_mathematics", "mmlu_mc_std_high_school_mathematics" ]
    # "qqp", "qnli"]  "m2d2_wikipedia_unsplit", "wikitext_103"] 
    #"mrpc", "matqa", "sciq",  ,
    #"mmlu_std_high_school_mathematics",

    # each task has usually 1000 validation examples
    downstream_eval_cb_config = DownstreamEvaluatorCallbackConfig(
        tasks=downstream_eval_tasks,
        tokenizer=tokenizer_config,
        eval_interval=config["steps"]/config["evaluation_times"],
        eval_on_startup=True, # run evaluation at the beginning as baseline
        log_interval=5,      # log progress every 20 eval batches
        enabled=True
    )

    # ===== LM Evaluator Callback Config =====
    lm_eval_dataset_config = NumpyDatasetConfig(
        paths=[config["data_dir"] + "/c4_validation.npy"],
        tokenizer=tokenizer_config,
        sequence_length=config["sequence_length"], # Assuming sequence_length is in your config
        name=NumpyDatasetType.padded_fsl,
        work_dir=work_dir,
        metadata=[{"label": "c4_validation_custom"}]
    )

    lm_eval_callback_config = LMEvaluatorCallbackConfig(
        eval_dataset=lm_eval_dataset_config,
        eval_interval=config["steps"] / config.get("evaluation_times", 1), # Default to 1 if not set
        eval_on_startup=True,
        log_interval=5,
        enabled=True
    )
    # =======================================

    trainer_config = TrainerConfig(
        save_folder=save_dir,
        save_overwrite=True,
        work_dir=work_dir,
        load_strategy=LoadStrategy.never,  # ADDED: Ensure training from scratch
        metrics_collect_interval=1,
        cancel_check_interval=5,
        max_duration=Duration.steps(config["steps"]),
        device=str(device),
    ).with_callback("wandb", wandb_cb
    ).with_callback("inference", inference_cb
    ).with_callback("downstream_eval", downstream_eval_cb_config
    )

    if config.get("data_preparation", {}).get("validation", False):
        trainer_config = trainer_config.with_callback("lm_evaluator", lm_eval_callback_config)
    

    trainer = trainer_config.build(train_module=train_module, data_loader=data_loader)
   
    print(f"Training for {config['steps']} steps on device: {device}\n")
    trainer.fit()
    print("\n✅ Training complete")
    
    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main() 