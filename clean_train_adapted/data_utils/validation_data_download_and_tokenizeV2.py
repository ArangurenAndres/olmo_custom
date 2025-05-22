import os
import numpy as np
from tqdm import tqdm
import itertools
from datasets import load_dataset
from transformers import AutoTokenizer
from olmo_core.data import TokenizerConfig

def validation_download_and_tokenize(data_path, sequence_length, total_tokens_with_margin):
    """
    Download and tokenize multiple datasets according to specified proportions,
    collecting tokens up to total_tokens_with_margin.

    Args:
        data_path: Path to save tokenized data (directory)
        sequence_length: Sequence length
        total_tokens_with_margin: Total tokens needed with margin
    """

    total_tokens_with_margin = 1_000_000
    tokenizer_processing_batch_size = 100  # Number of articles/entries to tokenize at once


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/gpt-neox-olmo-dolma-v1_5",
        cache_dir=os.path.join(os.environ["TRANSFORMERS_CACHE"], "tokenizers")
    )

    # Get tokenizer configuration from OLMo-core
    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    eos_token_id = tokenizer_config.eos_token_id

    # allenai/c4, en
    DATASET_PROPORTIONS = {
        "en": 1.0
    }
    
    
    # Define the tokenization function to be mapped
    def tokenize_function(examples):
        # Tokenize texts - IMPORTANT: Use the tokenizer directly on the batch
        tokenized_output = tokenizer(examples["text"], truncation=False, padding=False)
        # Add EOS token manually if needed (tokenizer might handle this depending on config)
        for ids in tokenized_output["input_ids"]:
            if not ids or ids[-1] != eos_token_id:
                ids.append(eos_token_id)
        return tokenized_output



    all_collected_token_arrays = []
    overall_collected_tokens_count = 0

    print("Starting dataset processing...")

    for dataset_hf_name, proportion in DATASET_PROPORTIONS.items():
        if overall_collected_tokens_count >= total_tokens_with_margin:
            print("Overall token target reached. Stopping further dataset processing.")
            break

        # Calculate tokens needed for this dataset, adjusted by remaining global quota
        tokens_to_collect_for_this_dataset = int(total_tokens_with_margin * proportion)

        print(f"\nProcessing dataset: {dataset_hf_name}. Target tokens for this dataset: {tokens_to_collect_for_this_dataset:,}")

        current_dataset_object = None
        # Assumes other datasets are loadable directly by name
        print(f"Loading dataset {dataset_hf_name} with streaming...")
        try:
            current_dataset_object = load_dataset(
                "allenai/c4", dataset_hf_name, split="validation", streaming=True,
                cache_dir=os.environ.get("HF_DATASETS_CACHE")
            )    

            print(f"Successfully initiated streaming for {dataset_hf_name}.")
        except Exception as e:
            print(f"Warning: Could not load dataset {dataset_hf_name}. Skipping. Error: {e}")
            continue

        list_of_token_arrays_for_current_dataset = []
        collected_tokens_for_current_dataset = 0
            
        dataset_iterator = current_dataset_object.iter(batch_size=tokenizer_processing_batch_size)

        with tqdm(total=tokens_to_collect_for_this_dataset, desc="Overall token collection", unit="token") as overall_progress_bar:

            for raw_batch in dataset_iterator:
                
                if "text" not in raw_batch:
                    print(f"Warning: Column 'text' not found in a sample. Skipping this sample.")
                    # print(f"Sample keys available: {list(sample.keys())}") # Uncomment to see available keys
                    continue

                tokenized_batch_output = tokenize_function(raw_batch)
                tokens_in_batch = list(itertools.chain.from_iterable(tokenized_batch_output['input_ids']))
                if not tokens_in_batch:
                    continue

                list_of_token_arrays_for_current_dataset.append(np.array(tokens_in_batch, dtype=np.int32))
            
                newly_collected = len(tokens_in_batch)
                collected_tokens_for_current_dataset += newly_collected
                overall_collected_tokens_count += newly_collected
                overall_progress_bar.update(newly_collected)

                if collected_tokens_for_current_dataset >= tokens_to_collect_for_this_dataset:
                    print(f"Collected {collected_tokens_for_current_dataset:,} tokens for {dataset_hf_name}. Breaking from batch loop.")
                    break  # Break from the batch processing loop for the current dataset
            
            # Add collected token arrays from the current dataset to the main list
            if list_of_token_arrays_for_current_dataset:
                all_collected_token_arrays.extend(list_of_token_arrays_for_current_dataset)
                print(f"Added {len(list_of_token_arrays_for_current_dataset)} token arrays from {dataset_hf_name} to the main collection of {len(all_collected_token_arrays)} arrays.")
            else:
                print(f"No token arrays were collected for {dataset_hf_name} in this iteration.")

            # Check if the overall token collection goal has been met or exceeded
            if overall_collected_tokens_count >= total_tokens_with_margin:
                print(f"Overall token target ({total_tokens_with_margin:,}) reached with {overall_collected_tokens_count:,} tokens. Stopping further dataset processing.")
                break  # Break from the main dataset loop
        
        


    # --- End of all dataset processing ---
    if not all_collected_token_arrays:
        print("Error: No token arrays were collected from any dataset. Cannot proceed.")
        return

    print(f"\nCollected {overall_collected_tokens_count:,} total tokens from {len(all_collected_token_arrays)} dataset segments.")
    print("Concatenating all collected NumPy arrays...")
    all_tokens_np = np.concatenate(all_collected_token_arrays)
    del all_collected_token_arrays
    print(f"Total concatenated tokens: {len(all_tokens_np):,}")



    # --- Reshape and Save ---
    num_sequences_possible = len(all_tokens_np) // sequence_length
    # Use the actual collected tokens for theoretical max sequences, or original target for 'needed'
    sequences_needed_based_on_target = (total_tokens_with_margin + sequence_length - 1) // sequence_length
    
    # To not cap, we create all possible sequences from the collected tokens
    num_sequences_to_create = num_sequences_possible

    if num_sequences_to_create == 0:
         print("Error: Collected tokens are less than the sequence length. Cannot create any sequences.")
         return

    print(f"Original target token count was {total_tokens_with_margin:,} (ideally {sequences_needed_based_on_target:,} sequences).")
    print(f"Total actual collected tokens: {len(all_tokens_np):,}.")
    
    if num_sequences_to_create < sequences_needed_based_on_target:
        print(f"Warning: Created {num_sequences_to_create:,} sequences, which is less than the target of {sequences_needed_based_on_target:,} sequences due to insufficient collected tokens.")
    elif num_sequences_to_create > sequences_needed_based_on_target:
        print(f"Created {num_sequences_to_create:,} sequences, exceeding the original target of {sequences_needed_based_on_target:,} sequences because more tokens were collected and capping is removed.")
    else: # num_sequences_to_create == sequences_needed_based_on_target
         print(f"Created {num_sequences_to_create:,} sequences, meeting the original target.")

    print(f"Final decision: Creating {num_sequences_to_create:,} sequences of length {sequence_length}.")

    final_token_count = num_sequences_to_create * sequence_length
    tokens_to_use = all_tokens_np[:final_token_count]
    # del all_tokens_np # Optional clear

    if tokens_to_use.size == 0:
        print("Error: No tokens available after slicing for sequences. Cannot create sequences.")
        return

    sequences = tokens_to_use.reshape(num_sequences_to_create, sequence_length)
    print(f"Created {sequences.shape[0]:,} sequences of length {sequence_length}")

    # Save tokenized data
    np.save(data_path, sequences)
    print(f"Saved tokenized mixed data to {data_path}")