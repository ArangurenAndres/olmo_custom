import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from olmo_core.data import (
    TokenizerConfig, NumpyDatasetConfig, NumpyDataLoaderConfig, NumpyDatasetType
)
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed

def prepare_data(data_dir, total_sequences, sequence_length, use_small_dataset=True):
    os.makedirs(data_dir, exist_ok=True)
    token_file = os.path.join(data_dir, "wiki_tokens.npy")

    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")

    if not os.path.exists(token_file):
        dataset = load_dataset("wikipedia", "20220301.en", split="train")
        all_tokens = []

        if use_small_dataset:
            dataset = dataset.select(range(1000))
            print("Using a small subset of Wikipedia (1000 articles) for testing.")
        else:
            print("Using full Wikipedia dataset for training.")

        for article in tqdm(dataset, desc="Tokenizing"):
            tokens = tokenizer.encode(article["text"])
            tokens = [t for t in tokens if t != 0]
            all_tokens.extend(tokens)
            if len(all_tokens) >= total_sequences * sequence_length:
                break

        tokens = all_tokens[:total_sequences * sequence_length]
        np.save(token_file, np.array(tokens, dtype=np.int32).reshape(-1, sequence_length))

    dataset_config = NumpyDatasetConfig(
        tokenizer=tokenizer_config,
        name=NumpyDatasetType.fsl,
        paths=[token_file],
        sequence_length=sequence_length,
        work_dir=os.path.join(data_dir, "dataset_work")
    )
    dataset = dataset_config.build()

    loader_config = NumpyDataLoaderConfig(
        global_batch_size=sequence_length,  # still here
        seed=42,
        num_workers=0,
    )
    loader = loader_config.build(dataset)

    return loader, tokenizer_config

def create_distributed_dataloader(dataset, config):
    """Create a distributed-aware dataloader."""
    
    # Calculate global batch size in tokens
    global_batch_size = config["batch_size"] * config["sequence_length"]
    world_size = get_world_size() if is_distributed() else 1
    
    # Ensure batch size is divisible by world size
    assert global_batch_size % world_size == 0, \
        f"Global batch size {global_batch_size} must be divisible by world size {world_size}"
    
    dataloader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size,
        seed=config.get("seed", 42),
        num_workers=config.get("num_workers", 8),
        prefetch_factor=2,
    )
    
    # Build the dataloader - let it automatically handle distributed settings
    # The build() method will internally call get_world_size() and get_rank()
    # to determine dp_world_size and dp_rank
    return dataloader_config.build(dataset)