"""
Dataset loading module for pre-tokenized data.
"""

import os
import json
import numpy as np
from olmo_core.data import NumpyDatasetConfig, NumpyDatasetType
from data_utils.tokenizer import get_tokenizer_config
from utils.environment import setup_environment
def load_prepared_dataset(args, data_dir, device):
    """
    Load pre-prepared dataset for OLMo training.

    Args:
        args: Command line arguments (should include batch_size, steps,
              sequence_length).
        data_dir: Base directory containing the 'wiki_data' folder.
        device: Training device (currently unused in this function but kept for signature consistency).

    Returns:
        tuple: (tokenizer_config, dataset)
    """
    setup_environment()

    # --- 1. Calculate Requirements & Print Info ---
    sequence_length = args.sequence_length
    global_batch_size = args.batch_size * sequence_length
    total_tokens_needed = args.steps * global_batch_size
    # Add margin for safety
    total_tokens_with_margin = int(total_tokens_needed * 1.05)
    sequences_needed = (total_tokens_with_margin + sequence_length - 1) // sequence_length

    print("--- Dataset Loading Configuration ---")
    print(f"- Batch size: {args.batch_size} sequences")
    print(f"- Training steps: {args.steps}")
    print(f"- Sequence length: {sequence_length}")
    print(f"- Base tokens needed for training steps: {total_tokens_needed:,}")
    print(f"- Estimated sequences needed (w/ 5% margin): {sequences_needed:,}")

    # --- 2. Get Tokenizer ---
    tokenizer_config = get_tokenizer_config()

    
    # --- 3. Validate Data Path ---
    final_data_path = os.path.join(data_dir, "wiki_tokens.npy")
    if not os.path.exists(final_data_path):
        raise FileNotFoundError(
            f"Base tokenized data not found at {final_data_path}. "
            "Run data preparation first."
        )
        

    # --- 4. Configure and Build Dataset ---
    dataset_config = NumpyDatasetConfig(
        tokenizer=tokenizer_config,
        name=NumpyDatasetType.fsl, 
        paths=[final_data_path],
        sequence_length=sequence_length,
        #work_dir=os.path.join(data_dir, "dataset_work_dir")
    )

    dataset = dataset_config.build()
    
    # --- 5. Check Sufficiency ---
    if len(dataset) < sequences_needed:
        print(f"Error: The loaded dataset ({len(dataset):,} sequences) "
              f"is smaller than the estimated requirement ({sequences_needed:,} sequences).")
        print("Please prepare more data or reduce training steps/batch size.")
        raise ValueError("Insufficient data for training requirements")

    print("--- Dataset Loading Complete ---")
    return tokenizer_config, dataset
