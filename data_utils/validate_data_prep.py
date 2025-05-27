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
        number_of_samples: Number of sequence samples to decode and print (default: 2)
    
    Returns:
        bool: True if validation successful
    """

    number_of_samples = 15
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
    
    # Calculate and print token ID distribution
    print("\n--- Token ID Distribution ---")
    token_ids, counts = np.unique(data, return_counts=True)
    token_id_distribution_meta = {str(tid): int(c) for tid, c in zip(token_ids, counts)}
    
    print(f"Total unique token IDs: {len(token_ids)}")
    sorted_indices = np.argsort(counts)[::-1]  # Sort by counts descending
    print("Top 10 most frequent token IDs:")
    for i in range(min(10, len(token_ids))):
        idx = sorted_indices[i]
        tid = token_ids[idx]
        count = counts[idx]
        percentage = (count / data.size) * 100
        print(f"  Token ID {tid}: {count:,} occurrences ({percentage:.2f}%)")
    print("Full token ID distribution saved in metadata JSON.")
    print("-----------------------------")
    
    # Load tokenizer for decoding samples
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/gpt-neox-olmo-dolma-v1_5",
        cache_dir=os.path.join(os.environ.get("TRANSFORMERS_CACHE", ".cache/transformers"), "tokenizers")
    )
    
    # Sample tokens for demonstration
    decoded_samples_for_metadata = []
    num_sequences = data.shape[0]
    
    actual_num_to_sample = 0
    if num_sequences > 0:
        actual_num_to_sample = min(number_of_samples, num_sequences)
        if number_of_samples > num_sequences:
            print(f"\nWarning: Requested {number_of_samples} samples, but only {num_sequences} sequences are available. Sampling {num_sequences} sequences.")
    elif number_of_samples > 0:
        print(f"\nWarning: Requested {number_of_samples} samples, but there are 0 sequences available.")

    if actual_num_to_sample > 0:
        sample_indices = np.random.choice(num_sequences, size=actual_num_to_sample, replace=False)
        if isinstance(sample_indices, np.int64): # Handle single sample case from np.random.choice
            sample_indices = [sample_indices.item()]
        
        print(f"\n--- Decoded Samples (first 200 tokens from each, showing first 100 chars) ---")
        for i, seq_idx in enumerate(sample_indices):
            sample_tokens = data[seq_idx, :200].tolist()
            decoded_sample_text = tokenizer.decode(sample_tokens)
            
            print(f"Sample {i+1} (from Sequence {seq_idx}): \n{decoded_sample_text[:100]}...")
            decoded_samples_for_metadata.append({
                "original_sequence_index": int(seq_idx),
                "decoded_sample_200_tokens": decoded_sample_text
            })
        print("--------------------------------------------------------------------")
    
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
        "samples": decoded_samples_for_metadata,
        "token_id_distribution": token_id_distribution_meta
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
    print(f"- Added {len(decoded_samples_for_metadata)} token sample(s) to metadata")
    print(f"- Token ID distribution statistics added to metadata")
    
    return True 