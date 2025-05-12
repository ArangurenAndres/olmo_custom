import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from olmo_core.data import (
    TokenizerConfig, NumpyDatasetConfig, NumpyDataLoaderConfig, NumpyDatasetType
)

#highlight-start
# It's good practice to pass the actual desired global batch size (number of sequences)
# as an argument to this function, rather than deriving it from sequence_length.
def prepare_data(data_dir, total_sequences_to_prepare, sequence_length, actual_global_batch_size, use_small_dataset=True):
#highlight-end
    os.makedirs(data_dir, exist_ok=True)
    token_file = os.path.join(data_dir, "wiki_tokens.npy")

    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    # The tokenizer is used here for preparing data.
    # Ensure the same tokenizer or its configuration is consistently used elsewhere if needed.
    tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")

    if not os.path.exists(token_file):
        dataset = load_dataset("wikipedia", "20220301.en", split="train")
        all_tokens = []

        if use_small_dataset:
            dataset = dataset.select(range(1000))
            print("Using a small subset of Wikipedia (1000 articles) for testing.")
        else:
            print("Using full Wikipedia dataset for training.")

        # The total_sequences_to_prepare determines the size of the .npy file
        # It should be large enough for your training run.
        for article in tqdm(dataset, desc="Tokenizing"):
            tokens = tokenizer.encode(article["text"])
            tokens = [t for t in tokens if t != 0] # Remove padding or special tokens if not desired
            all_tokens.extend(tokens)
            # Ensure enough tokens are collected for the specified number of sequences
            if len(all_tokens) >= total_sequences_to_prepare * sequence_length:
                break
        
        # Truncate to the exact number of tokens needed for total_sequences_to_prepare
        tokens_to_save = all_tokens[:total_sequences_to_prepare * sequence_length]
        # Reshape into (total_sequences_to_prepare, sequence_length)
        reshaped_tokens = np.array(tokens_to_save, dtype=np.int32).reshape(-1, sequence_length)
        np.save(token_file, reshaped_tokens)
        print(f"Saved {reshaped_tokens.shape[0]} sequences of length {sequence_length} to {token_file}")


    dataset_config = NumpyDatasetConfig(
        tokenizer=tokenizer_config, # This is mainly for metadata if data is pre-tokenized
        name=NumpyDatasetType.fsl,
        paths=[token_file],
        sequence_length=sequence_length,
        work_dir=os.path.join(data_dir, "dataset_work")
    )
    dataset = dataset_config.build()

    loader_config = NumpyDataLoaderConfig(
        #highlight-start
        # This is a CRITICAL parameter.
        # It should be the total number of sequences you want to process
        # in one gradient accumulation step, across ALL GPUs.
        # For example, if you want each of 4 GPUs to process 8 sequences,
        # and you have no gradient accumulation, global_batch_size would be 32.
        # It should NOT be `sequence_length`.
        global_batch_size=actual_global_batch_size,
        #highlight-end
        seed=42,  # Seed for shuffling (if shuffle=True, which is default)
        num_workers=0, # Adjust based on your system for optimal data loading speed
        shuffle=True, # Default is True, good for training
        # drop_last=True # Default is True for training, False for eval.
                         # Ensures all batches have the same size, which can be important for FSDP.
    )
    loader = loader_config.build(dataset)

    return loader, tokenizer_config