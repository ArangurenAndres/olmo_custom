import os
import numpy as np
from datasets import load_dataset # type: ignore
from tqdm import tqdm # type: ignore
from transformers import AutoTokenizer # type: ignore
from olmo_core.data import (
    TokenizerConfig, NumpyDatasetConfig, NumpyDatasetType
)
import torch.distributed as dist # For DDP barrier and rank/world_size

def prepare_data(
    data_dir: str,
    total_sequences: int,       # This should be the total number of sequences for the entire dataset
    sequence_length: int,
    global_rank: int,           # Current global rank of the process
    world_size: int,            # Total number of processes
    use_small_dataset: bool = True,
    tokenizer_name: str = "allenai/gpt-neox-olmo-dolma-v1_5" # Make tokenizer configurable
):
    """
    Prepares the dataset for training.
    In a DDP setup, data download and tokenization are done only by the main process (rank 0).
    All processes will load the tokenized data.

    Args:
        data_dir: Directory to store and load tokenized data.
        total_sequences: The total number of sequences the final dataset should contain.
                         The tokenization process will aim to generate at least this many.
        sequence_length: The length of each sequence.
        global_rank: The global rank of the current process.
        world_size: The total number of distributed processes.
        use_small_dataset: Whether to use a small subset of Wikipedia for faster testing.
        tokenizer_name: The name of the Hugging Face tokenizer to use.

    Returns:
        dataset: An olmo_core.data.NumpyDataset (or compatible torch.utils.data.Dataset).
        tokenizer_config: The OLMo TokenizerConfig.
    """
    if global_rank == 0:
        print(f"[Rank {global_rank}] Preparing data...")
        os.makedirs(data_dir, exist_ok=True)
    
    # Define the path for the tokenized file
    # Using a more descriptive name based on dataset and sequence length could be beneficial
    # if you plan to experiment with different settings.
    token_file_basename = "wiki_tokens"
    if use_small_dataset:
        token_file_basename += "_small"
    token_file_basename += f"_seqlen{sequence_length}.npy"
    token_file = os.path.join(data_dir, token_file_basename)

    # --- Tokenizer Setup ---
    # This can be done by all processes, as Hugging Face caches tokenizers.
    # Or, you could also restrict this to rank 0 and broadcast tokenizer_config if it were complex.
    # For TokenizerConfig.gpt_neox_olmo_dolma_v1_5(), it's simple enough.
    try:
        tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
        tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")
    except Exception as e:
        # Fallback if from_pretrained on TokenizerConfig fails or for custom paths
        print(f"[Rank {global_rank}] Warning: Could not load TokenizerConfig directly for '{tokenizer_name}'. "
              f"Using default OLMo GPT-NeoX settings. Error: {e}")
        tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5() # Default
        tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5") # Ensure this matches

    # --- Data Tokenization and Saving (Rank 0 Only) ---
    if global_rank == 0:
        if not os.path.exists(token_file):
            print(f"[Rank {global_rank}] Token file '{token_file}' not found. Starting tokenization.")
            # Load the Wikipedia dataset
            # Using a specific version for reproducibility.
            # Consider adding try-except for dataset loading if network issues are common.
            dataset_name = "wikipedia"
            dataset_config_name = "20220301.en"
            print(f"[Rank {global_rank}] Loading '{dataset_name}' dataset ('{dataset_config_name}')...")
            wiki_dataset = load_dataset(dataset_name, dataset_config_name, split="train")
            
            all_tokens = []

            num_articles_to_process = 1000 if use_small_dataset else len(wiki_dataset)
            if use_small_dataset:
                print(f"[Rank {global_rank}] Using a small subset of Wikipedia ({num_articles_to_process} articles) for testing.")
                # Shuffle and select to get a more diverse small set if desired, though range is simpler.
                processed_dataset = wiki_dataset.select(range(num_articles_to_process))
            else:
                print(f"[Rank {global_rank}] Using full Wikipedia dataset ({num_articles_to_process} articles) for training.")
                processed_dataset = wiki_dataset

            # Calculate the target number of raw tokens needed
            target_raw_tokens = total_sequences * sequence_length
            
            print(f"[Rank {global_rank}] Tokenizing articles. Target raw tokens: {target_raw_tokens}, Target sequences: {total_sequences}")
            for article in tqdm(processed_dataset, desc=f"[Rank {global_rank}] Tokenizing", total=num_articles_to_process):
                if article["text"] is None or not article["text"].strip():
                    continue # Skip empty articles
                
                # Encode text into tokens
                # The filter `t != 0` is kept from your original script.
                # Token ID 0 can sometimes be <unk>, <pad>, or even <bos>/<eos> depending on the tokenizer.
                # Ensure this filtering is intended for your specific tokenizer and use case.
                # For many tokenizers (like GPT-NeoX), 0 might be a valid token or a special one.
                # If 0 is, for example, the EOS token from the tokenizer, filtering it out here
                # might be problematic unless handled carefully later.
                tokens = tokenizer.encode(article["text"])
                tokens = [t for t in tokens if t != 0] # Original filter
                all_tokens.extend(tokens)

                # Stop if we have enough tokens for the desired number of sequences
                if len(all_tokens) >= target_raw_tokens:
                    print(f"[Rank {global_rank}] Reached target token count ({len(all_tokens)} >= {target_raw_tokens}).")
                    break
            
            if len(all_tokens) < target_raw_tokens:
                print(f"[Rank {global_rank}] Warning: Collected {len(all_tokens)} tokens, "
                      f"which is less than the target {target_raw_tokens} tokens "
                      f"for {total_sequences} sequences of length {sequence_length}.")
                # Adjust total_sequences if not enough tokens were collected
                actual_sequences = len(all_tokens) // sequence_length
                print(f"[Rank {global_rank}] Will create {actual_sequences} sequences instead.")
            else:
                actual_sequences = total_sequences

            # Trim to the exact number of tokens needed for the sequences and reshape
            final_tokens_count = actual_sequences * sequence_length
            tokens_to_save = all_tokens[:final_tokens_count]
            
            if not tokens_to_save:
                raise ValueError(f"[Rank {global_rank}] No tokens were collected or processed. Cannot create an empty dataset.")

            # Reshape into (num_sequences, sequence_length)
            reshaped_tokens = np.array(tokens_to_save, dtype=np.int32).reshape(-1, sequence_length)
            
            print(f"[Rank {global_rank}] Saving {reshaped_tokens.shape[0]} sequences to '{token_file}'.")
            np.save(token_file, reshaped_tokens)
            print(f"[Rank {global_rank}] Tokenization and saving complete.")
        else:
            print(f"[Rank {global_rank}] Token file '{token_file}' already exists. Skipping tokenization.")

    # --- Synchronization Point ---
    # All processes wait here until rank 0 has finished creating the token_file.
    if world_size > 1:
        print(f"[Rank {global_rank}] Waiting for data preparation barrier...")
        dist.barrier()
        print(f"[Rank {global_rank}] Passed data preparation barrier.")

    # --- Load Dataset (All Processes) ---
    # Now all processes can safely access token_file.
    # Ensure the file exists before proceeding, especially for worker ranks.
    if not os.path.exists(token_file):
        raise FileNotFoundError(
            f"[Rank {global_rank}] Error: Token file '{token_file}' not found after barrier. "
            "This should not happen if rank 0 completed successfully."
        )
    
    # Configure and build the OLMo NumpyDataset
    # This dataset object should be compatible with torch.utils.data.Dataset
    dataset_work_dir = os.path.join(data_dir, f"dataset_work_rank{global_rank}") # Rank-specific work_dir if needed
    if global_rank == 0: # Create the base data_dir if it wasn't by tokenization (e.g. if token_file existed)
        os.makedirs(data_dir, exist_ok=True)
    # os.makedirs(dataset_work_dir, exist_ok=True) # NumpyDatasetConfig might handle its own work_dir creation

    print(f"[Rank {global_rank}] Building NumpyDatasetConfig with token_file: '{token_file}'")
    dataset_config = NumpyDatasetConfig(
        tokenizer=tokenizer_config,
        name=NumpyDatasetType.fsl,  # Few-Shot Learning type, assumes standard token layout
        paths=[token_file],         # Path to the .npy file
        sequence_length=sequence_length,
        # work_dir could be shared or rank-specific depending on NumpyDatasetConfig's needs
        # If it writes temporary files that could conflict, a rank-specific dir is safer.
        # For read-only operations from a single .npy, a shared dir or even no explicit work_dir might be fine.
        work_dir=os.path.join(data_dir, "dataset_work_shared") # Example: shared work_dir
    )
    
    print(f"[Rank {global_rank}] Building dataset...")
    try:
        dataset = dataset_config.build()
        print(f"[Rank {global_rank}] Dataset built successfully. Type: {type(dataset)}")
        # You might want to log len(dataset) here if it's available and meaningful
        # For DDP, each rank gets the full dataset object, and DistributedSampler handles sharding.
    except Exception as e:
        print(f"[Rank {global_rank}] Error building dataset from NumpyDatasetConfig: {e}")
        raise

    # The DataLoader is now created in train.py using this dataset and DistributedSampler.
    # So, we remove the NumpyDataLoaderConfig and loader building from here.

    return dataset, tokenizer_config