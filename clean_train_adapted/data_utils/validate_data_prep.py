"""
Validation module for OLMo data preparation.

Validates tokenized datasets to ensure they're ready for training.
"""

import os
import json
import numpy as np
from transformers import AutoTokenizer


def validate_tokenized_data(data_path, sequence_length):
    """
    Validate tokenized data and create metadata.
    
    Args:
        data_path: Path to the tokenized data file
        sequence_length: Expected sequence length
    
    Returns:
        bool: True if validation successful
    """
    print(f"Validating tokenized data at {data_path}")
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found")
        return False
    
    # Load the data
    try:
        data = np.load(data_path)
        print(f"Loaded data with shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Check data shape
    if len(data.shape) != 2:
        print(f"Error: Expected 2D data, but got shape {data.shape}")
        return False
    
    # Check sequence length
    if data.shape[1] != sequence_length:
        print(f"Error: Expected sequence length {sequence_length}, but got {data.shape[1]}")
        return False
    
    # Check data type
    if data.dtype != np.int32:
        print(f"Warning: Data type is {data.dtype}, converting to int32")
        data = data.astype(np.int32)
        np.save(data_path, data)
    
    # Check for token ID 0
    zero_tokens = np.sum(data == 0)
    if zero_tokens > 0:
        print(f"Warning: Found {zero_tokens} instances of token ID 0, replacing with token ID 1")
        data[data == 0] = 1
        np.save(data_path, data)
    
    # Load tokenizer for decoding samples
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/gpt-neox-olmo-dolma-v1_5",
        cache_dir=os.path.join(os.environ["TRANSFORMERS_CACHE"], "tokenizers")
    )
    
    # Sample 200 tokens from two different sequences for demonstration
    num_sequences = data.shape[0]
    if num_sequences >= 2:
        # Take from two different sequences
        sample1_idx = np.random.randint(0, num_sequences)
        sample2_idx = (sample1_idx + num_sequences // 2) % num_sequences  # Take from a different part
        
        sample1 = data[sample1_idx, :200].tolist()
        sample2 = data[sample2_idx, :200].tolist()
        
        # Decode the samples
        decoded_sample1 = tokenizer.decode(sample1)
        decoded_sample2 = tokenizer.decode(sample2)
        
        print(f"Sample 1 (Sequence {sample1_idx}):\n{decoded_sample1[:100]}...")
        print(f"Sample 2 (Sequence {sample2_idx}):\n{decoded_sample2[:100]}...")
    
    # Create metadata
    metadata = {
        "num_sequences": data.shape[0],
        "sequence_length": data.shape[1],
        "total_tokens": data.shape[0] * data.shape[1],
        "dtype": str(data.dtype),
        "file_size_mb": os.path.getsize(data_path) / (1024 * 1024),
        "token_range": {
            "min": int(np.min(data)),
            "max": int(np.max(data))
        },
        "samples": {
            "sample1 decoded": decoded_sample1,
            "sample2 decoded": decoded_sample2
            
        }
    }
    
    # Save metadata
    metadata_path = data_path.replace(".npy", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Validation successful. Created metadata file at {metadata_path}")
    print(f"Dataset statistics:")
    print(f"- Number of sequences: {metadata['num_sequences']:,}")
    print(f"- Sequence length: {metadata['sequence_length']}")
    print(f"- Total tokens: {metadata['total_tokens']:,}")
    print(f"- Token ID range: {metadata['token_range']['min']} to {metadata['token_range']['max']}")
    print(f"- Added token samples to metadata")
    
    return True 