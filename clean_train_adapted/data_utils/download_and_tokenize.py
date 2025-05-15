"""
Dataset preparation and loading module.
"""

import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import itertools 

from olmo_core.data import TokenizerConfig


def _download_and_tokenize_wiki(wiki_data_path, sequence_length, total_tokens_with_margin):
    """
    Download and tokenize Wikipedia data efficiently, collecting tokens up to
    total_tokens_with_margin.

    Args:
        wiki_data_path: Path to save tokenized data
        sequence_length: Sequence length
        total_tokens_with_margin: Total tokens needed with margin
    """

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/gpt-neox-olmo-dolma-v1_5",
        cache_dir=os.path.join(os.environ["TRANSFORMERS_CACHE"], "tokenizers")
    )

    # Get tokenizer configuration from OLMo-core
    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    eos_token_id = tokenizer_config.eos_token_id

    # Download Wikipedia dataset
    print("Downloading Wikipedia dataset...")
    wiki_dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )
    print(f"Downloaded full Wikipedia dataset with {len(wiki_dataset)} articles")

    # Define the tokenization function to be mapped
    def tokenize_function(examples):
        # Tokenize texts - IMPORTANT: Use the tokenizer directly on the batch
        tokenized_output = tokenizer(examples["text"], truncation=False, padding=False)
        # Add EOS token manually if needed (tokenizer might handle this depending on config)
        for ids in tokenized_output["input_ids"]:
            if not ids or ids[-1] != eos_token_id:
                ids.append(eos_token_id)
        return tokenized_output


    # Concatenate using intermediate NumPy arrays (Option 2)
    list_of_token_arrays = []
    collected_count = 0
    tokenizer_processing_batch_size = 1000  # Number of articles to tokenize at once

    print("Iteratively tokenizing and collecting articles until target token count is reached...")
   

    with tqdm(total=total_tokens_with_margin, desc="Collecting tokens", unit="token") as progress_bar:
        for raw_batch in wiki_dataset.iter(batch_size=tokenizer_processing_batch_size):
            # raw_batch is like {'id': [...], 'url': [...], 'title': [...], 'text': [...]}
            # Tokenize the 'text' field of the current batch
            tokenized_batch_output = tokenize_function(raw_batch)

            # tokenized_batch_output is {'input_ids': [ [tokens_article1], [tokens_article2], ... ], ...}
            # Flatten the list of lists of input_ids for this batch
            tokens_in_batch = list(itertools.chain.from_iterable(tokenized_batch_output['input_ids']))

            if not tokens_in_batch:
                continue

            list_of_token_arrays.append(np.array(tokens_in_batch, dtype=np.int32))
            
            newly_collected = len(tokens_in_batch)
            collected_count += newly_collected
            progress_bar.update(min(newly_collected, total_tokens_with_margin - progress_bar.n))


            if collected_count >= total_tokens_with_margin:
                print(f"\nCollected enough tokens ({collected_count:,}), stopping tokenization and collection.")
                if progress_bar.n < total_tokens_with_margin: # Ensure bar reaches 100% if we collected slightly more
                    progress_bar.update(total_tokens_with_margin - progress_bar.n)
                break

    if not list_of_token_arrays:
        print("Error: No token arrays were collected. Cannot proceed.")
        return

    print(f"Collected {collected_count:,} total tokens in {len(list_of_token_arrays)} arrays.")
    print("Concatenating NumPy arrays...")
    # Concatenate all the collected numpy arrays into one large array
    all_tokens_np = np.concatenate(list_of_token_arrays)
    # It's good practice to clear the intermediate list to free up memory sooner
    del list_of_token_arrays
    print(f"Total concatenated tokens: {len(all_tokens_np):,}")


    # --- Reshape and Save --- (Using the NumPy array 'all_tokens_np')

    # Calculate the number of sequences we can actually make from collected tokens
    num_sequences_possible = len(all_tokens_np) // sequence_length

    # Calculate the number of sequences theoretically needed based on the target
    sequences_needed = (total_tokens_with_margin + sequence_length - 1) // sequence_length

    # Determine the number of sequences to actually create (the minimum of possible and needed)
    num_sequences_to_create = min(num_sequences_possible, sequences_needed)

    if num_sequences_possible == 0:
         print("Error: Collected tokens are less than the sequence length. Cannot create any sequences.")
         return # Exit if no sequences can be formed

    if num_sequences_possible < sequences_needed:
        print(f"Warning: Only able to create {num_sequences_possible:,} sequences from collected tokens.")
        print(f"Target token count ({total_tokens_with_margin:,}) would ideally yield {sequences_needed:,} sequences.")
        print(f"Creating {num_sequences_to_create:,} sequences.")
    else:
         print(f"Collected enough tokens ({len(all_tokens_np):,}) to create the target {sequences_needed:,} sequences.")
         print(f"Creating {num_sequences_to_create:,} sequences.")


    # Take only the tokens needed for the complete sequences we are creating
    # Ensure we don't try to slice beyond the collected tokens if concatenation stopped early
    final_token_count = num_sequences_to_create * sequence_length
    # Slice the final NumPy array directly
    tokens_to_use = all_tokens_np[:final_token_count]
    # Optionally clear the large concatenated array if memory is extremely tight *after* slicing
    # del all_tokens_np

    if tokens_to_use.size == 0:
        print("Error: No tokens available after slicing. Cannot create sequences.")
        return

    # Reshape the final NumPy array directly
    sequences = tokens_to_use.reshape(num_sequences_to_create, sequence_length)

    print(f"Created {sequences.shape[0]:,} sequences of length {sequence_length}")

    # Save tokenized data
    np.save(wiki_data_path, sequences)
    print(f"Saved tokenized data to {wiki_data_path}") 