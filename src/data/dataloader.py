"""
Data loading utilities for OLMo training with FSDP support.
"""

import os
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from tqdm import tqdm

from datasets import load_dataset
import torch
from torch.utils.data import DistributedSampler

from olmo_core.data import (
    NumpyDatasetConfig,
    NumpyDataLoaderConfig,
    NumpyDatasetType
)
from olmo_core.distributed.utils import (
    is_distributed,
    get_rank,
    get_world_size
)

from .tokenizer import get_tokenizer_config, load_tokenizer

logger = logging.getLogger(__name__)

def prepare_wikipedia_data(
    output_dir: str,
    sequence_length: int,
    tokenizer_name: str = "allenai/gpt-neox-olmo-dolma-v1_5",
    cache_dir: Optional[str] = None,
    max_articles: Optional[int] = None,
    min_total_tokens: Optional[int] = None,
    wikipedia_version: str = "20220301.en",
    output_file: str = "wiki_tokens.npy",
    overwrite: bool = False
) -> str:
    """
    Prepare Wikipedia data by downloading, tokenizing, and saving for training.
    
    Args:
        output_dir: Directory to save processed data
        sequence_length: Length of sequences for training
        tokenizer_name: Name of the tokenizer to use
        cache_dir: Directory to cache tokenizer files
        max_articles: Maximum number of articles to process (None for all)
        min_total_tokens: Minimum number of tokens to collect (overrides max_articles)
        wikipedia_version: Version of Wikipedia to use
        output_file: Name of the output file
        overwrite: Whether to overwrite existing processed data
        
    Returns:
        str: Path to the processed token file
    """
    os.makedirs(output_dir, exist_ok=True)
    token_file = os.path.join(output_dir, output_file)
    
    # Skip if file exists and not overwriting
    if os.path.exists(token_file) and not overwrite:
        logger.info(f"Using existing tokenized data: {token_file}")
        return token_file
    
    logger.info(f"Processing Wikipedia data (version: {wikipedia_version})")
    logger.info(f"{'All' if max_articles is None else max_articles} articles will be processed")
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_name, cache_dir)
    
    # Load Wikipedia dataset
    dataset = load_dataset("wikipedia", wikipedia_version, split="train")
    
    # Limit to max_articles if specified
    if max_articles is not None:
        dataset = dataset.select(range(min(max_articles, len(dataset))))
    
    all_tokens = []
    
    # Process and tokenize articles
    for article in tqdm(dataset, desc="Tokenizing Wikipedia articles"):
        tokens = tokenizer.encode(article["text"])
        # Remove padding tokens (0)
        tokens = [t for t in tokens if t != 0]
        all_tokens.extend(tokens)
        
        # Check if we have enough tokens
        if min_total_tokens is not None and len(all_tokens) >= min_total_tokens:
            logger.info(f"Collected {len(all_tokens)} tokens, stopping collection")
            break
    
    # Truncate to multiple of sequence_length
    num_sequences = len(all_tokens) // sequence_length
    tokens_to_use = all_tokens[:num_sequences * sequence_length]
    
    # Reshape and save
    tokens_array = np.array(tokens_to_use, dtype=np.int32).reshape(-1, sequence_length)
    np.save(token_file, tokens_array)
    
    logger.info(f"Saved {len(tokens_to_use)} tokens ({tokens_array.shape[0]} sequences) to {token_file}")
    
    return token_file

def create_dataset_config(
    data_path: str,
    sequence_length: int,
    tokenizer_config: Optional[Any] = None,
    work_dir: Optional[str] = None,
    dataset_type: NumpyDatasetType = NumpyDatasetType.fsl
) -> NumpyDatasetConfig:
    """
    Create configuration for the dataset.
    
    Args:
        data_path: Path to the data file
        sequence_length: Sequence length for training
        tokenizer_config: Tokenizer configuration
        work_dir: Working directory for the dataset
        dataset_type: Type of dataset to use
        
    Returns:
        NumpyDatasetConfig: Dataset configuration
    """
    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(data_path), "dataset_work")
    
    os.makedirs(work_dir, exist_ok=True)
    
    return NumpyDatasetConfig(
        tokenizer=tokenizer_config,
        name=dataset_type,
        paths=[data_path],
        sequence_length=sequence_length,
        work_dir=work_dir
    )

def get_distributed_sampler(dataset: torch.utils.data.Dataset) -> Optional[DistributedSampler]:
    """
    Get a distributed sampler for FSDP training.
    
    Args:
        dataset: Dataset to sample from
        
    Returns:
        Optional[DistributedSampler]: Distributed sampler or None if not in distributed mode
    """
    if is_distributed():
        return DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            seed=42
        )
    return None

def create_dataloader_config(
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
    sampler: Optional[torch.utils.data.Sampler] = None
) -> NumpyDataLoaderConfig:
    """
    Create configuration for the data loader.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        seed: Random seed
        pin_memory: Whether to pin memory for faster data transfer to GPU
        sampler: Optional sampler for the data loader
        
    Returns:
        NumpyDataLoaderConfig: Data loader configuration
    """
    return NumpyDataLoaderConfig(
        global_batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        # pin_memory=pin_memory,
        sampler=sampler
    )

def build_dataloader(
    config: Dict[str, Any],
    distributed: bool = False
) -> Tuple[torch.utils.data.DataLoader, Any, str]:
    """
    Build the complete data pipeline with FSDP support.
    
    Args:
        config: Configuration dictionary with data settings
        distributed: Whether to use distributed training
        
    Returns:
        Tuple containing:
            - DataLoader: The configured data loader
            - TokenizerConfig: The tokenizer configuration
            - str: Path to the processed data file
    """
    # Setup directories
    data_dir = config.get("data_dir", "./data")
    os.makedirs(data_dir, exist_ok=True)
    
    sequence_length = config.get("sequence_length", 1024)
    batch_size = config.get("batch_size", 8)
    
    # Calculate needed tokens based on training steps
    steps = config.get("steps", 1000)
    total_tokens = steps * batch_size * sequence_length
    min_tokens_needed = int(total_tokens * 1.1)  # 10% margin
    
    # Get tokenizer config
    tokenizer_config = get_tokenizer_config(
        cache_dir=os.path.join(data_dir, "tokenizer_cache"),
        tokenizer_name=config.get("tokenizer_name", "allenai/gpt-neox-olmo-dolma-v1_5")
    )
    
    # Prepare Wikipedia data
    token_file = prepare_wikipedia_data(
        output_dir=data_dir,
        sequence_length=sequence_length,
        tokenizer_name=config.get("tokenizer_name", "allenai/gpt-neox-olmo-dolma-v1_5"),
        cache_dir=os.path.join(data_dir, "tokenizer_cache"),
        max_articles=config.get("max_articles", 1000 if config.get("use_small_dataset", True) else None),
        min_total_tokens=min_tokens_needed,
        wikipedia_version=config.get("wikipedia_version", "20220301.en"),
        output_file=config.get("output_file", "wiki_tokens.npy"),
        overwrite=config.get("overwrite_data", False)
    )
    
    # Create dataset
    dataset_config = create_dataset_config(
        data_path=token_file,
        sequence_length=sequence_length,
        tokenizer_config=tokenizer_config,
        work_dir=os.path.join(data_dir, "dataset_work")
    )
    dataset = dataset_config.build()
    
    # Create sampler for distributed training if needed
    sampler = get_distributed_sampler(dataset) if distributed else None
    
    # Create data loader
    dataloader_config = create_dataloader_config(
        batch_size=batch_size,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
        sampler=sampler
    )
    dataloader = dataloader_config.build(dataset)
    
    return dataloader, tokenizer_config, token_file