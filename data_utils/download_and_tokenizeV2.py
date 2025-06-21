import os
import numpy as np
from tqdm import tqdm
import itertools
from datasets import load_dataset
from transformers import AutoTokenizer
from olmo_core.data import TokenizerConfig
import torch
from torch.utils.data import DataLoader, IterableDataset
from numpy.lib.format import open_memmap  # allows writing large .npy chunks without loading them fully into RAM

def download_and_tokenize(data_path, sequence_length, total_tokens_with_margin, tokenizer_processing_batch_size, dataset_proportions, validation_token_target):
    """
    Download and tokenize multiple datasets according to specified proportions,
    collecting tokens up to total_tokens_with_margin.

    Args:
        data_path: Path to save tokenized data (directory)
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




    # Define the tokenization function to be mapped
    def tokenize_function(examples, dataset_name_for_logic=""):
        current_examples = examples

        if "text" not in examples:
            if "ai2_arc" in dataset_name_for_logic.lower():
                texts = []
                num_examples = len(examples.get("question", []))
                for i in range(num_examples):
                    question = examples["question"][i]
                    choices_text_list = []
                    if ("choices" in examples and examples["choices"] and 
                        isinstance(examples["choices"], list) and i < len(examples["choices"]) and 
                        examples["choices"][i] and 
                        "text" in examples["choices"][i] and "label" in examples["choices"][i]):
                        for label, text_choice in zip(examples["choices"][i]["label"], examples["choices"][i]["text"]):
                            choices_text_list.append(f"{label}) {text_choice}")
                    choices_str = "\\n".join(choices_text_list)
                    answer_key = examples["answerKey"][i] if "answerKey" in examples and isinstance(examples["answerKey"], list) and i < len(examples["answerKey"]) else "N/A"
                    
                    combined_text = f"Question: {question}\\nChoices:\\n{choices_str}\\nAnswer: {answer_key}"
                    texts.append(combined_text)
                
                if texts: 
                    current_examples = {"text": texts}

            elif "sciq" in dataset_name_for_logic.lower():
                texts = []
                num_examples = len(examples.get("question", []))
                for i in range(num_examples):
                    question = examples["question"][i]
                    support = examples["support"][i] if "support" in examples and isinstance(examples["support"], list) and i < len(examples["support"]) else ""
                    correct_answer = examples["correct_answer"][i] if "correct_answer" in examples and isinstance(examples["correct_answer"], list) and i < len(examples["correct_answer"]) else "N/A"
                    
                    combined_text = f"Question: {question}\\\\nSupport: {support}\\\\nAnswer: {correct_answer}"
                    texts.append(combined_text)

                if texts:
                    current_examples = {"text": texts}

        if "text" not in current_examples or not isinstance(current_examples["text"], list) or not current_examples["text"]:
            return {"input_ids": [], "attention_mask": []}

        tokenized_output = tokenizer(current_examples["text"], truncation=False, padding=False)
        for ids in tokenized_output["input_ids"]:
            if not ids or ids[-1] != eos_token_id:
                ids.append(eos_token_id)
        return tokenized_output



    # ------------------------------------------------------------------
    # Incremental saving configuration (10B-token chunks)
    # ------------------------------------------------------------------
    CHUNK_SIZE_TOKENS = 100_000_000  # 10B tokens per intermediate file

    # Runtime bookkeeping
    chunk_index: int = 0                                 # current chunk id
    chunk_token_buffer: list[np.ndarray] = []            # list of small arrays that belong to the current chunk
    tokens_in_current_chunk: int = 0                     # counter for current chunk size
    saved_chunk_paths: list[str] = []                    # paths of all chunks written to disk, in order
    saved_chunk_sizes: list[int] = []                    # number of valid tokens in each saved chunk

    overall_collected_tokens_count = 0                   # global counter

    print("Starting dataset processing...")

    

    for dataset_hf_name, proportion in dataset_proportions.items():
        if overall_collected_tokens_count >= total_tokens_with_margin:
            print("Overall token target reached. Stopping further dataset processing.")
            break

        # Calculate tokens needed for this dataset, adjusted by remaining global quota
        tokens_to_collect_for_this_dataset = int(total_tokens_with_margin * proportion)

        print(f"\nProcessing dataset: {dataset_hf_name}. Target tokens for this dataset: {tokens_to_collect_for_this_dataset:,}")

        current_dataset_object = None
        
        # Parse dataset name and potential subset/config name
        dataset_name_parts = dataset_hf_name.rsplit('/', 1) # Split only on the last '/'
        main_dataset_name = dataset_name_parts[0]
        subset_name = dataset_name_parts[1] if len(dataset_name_parts) > 1 else None

        print(f"Loading dataset {main_dataset_name} (subset: {subset_name}) with streaming...")
        try:
            if subset_name == "sciq":
                current_dataset_object = load_dataset(
                    "allenai/sciq",
                    split="train",
                    streaming=True,
                    cache_dir=os.environ.get("HF_DATASETS_CACHE"),
                )
            else:
                current_dataset_object = load_dataset(
                    main_dataset_name,
                    name=subset_name,
                    split="train",
                    streaming=True,
                    cache_dir=os.environ.get("HF_DATASETS_CACHE"),
                )

            print(f"Successfully initiated streaming for {dataset_hf_name}.")
        except Exception as e:
            print(f"Warning: Could not load dataset {dataset_hf_name}. Skipping. Error: {e}")
            continue



        # For large streaming datasets we must NOT keep every batch in RAM.
        # We therefore avoid storing token arrays long-term. Duplication for
        # small QA datasets is handled on the fly.
        collected_tokens_for_current_dataset = 0
            
        # ------------------ PyTorch DataLoader Setup ------------------
        class HFIterableDatasetWrapper(IterableDataset):
            """Wrap a HuggingFace streaming (Iterable) dataset so it can be fed to a PyTorch DataLoader."""

            def __init__(self, hf_iter_dataset):
                self.hf_iter_dataset = hf_iter_dataset

            def __iter__(self):
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is None:
                    # Single-worker behaviour – iterate over the full dataset
                    return iter(self.hf_iter_dataset)
                else:
                    # In multiple-worker setting, shard dataset by worker id
                    return iter(
                        self.hf_iter_dataset.shard(
                            num_shards=worker_info.num_workers,
                            index=worker_info.id,
                        )
                    )

        def collate_tokens_pytorch(batch):
            """Custom collate_fn that tokenises a list of raw samples and flattens to a 1-D NumPy array."""
            # Convert list[dict[str, Any]] -> dict[str, list[Any]] expected by tokenize_function
            aggregated = {}
            for sample in batch:
                for key, value in sample.items():
                    aggregated.setdefault(key, []).append(value)

            tokenized_batch_output = tokenize_function(aggregated, dataset_hf_name)
            tokens_flat = list(itertools.chain.from_iterable(tokenized_batch_output["input_ids"]))
            return np.array(tokens_flat, dtype=np.int32)

        torch_dataset = HFIterableDatasetWrapper(current_dataset_object)

        # Heuristic: allow more workers for very large, highly‐sharded corpora (e.g. Dolma or Wiki dumps).
        if ("dclm" in dataset_hf_name.lower()) or ("wiki" in dataset_hf_name.lower()):
            num_workers_dl = int(os.getenv("STREAMING_NUM_WORKERS", "8"))
        else:
            num_workers_dl = int(os.getenv("STREAMING_NUM_WORKERS", "1"))

        max_shards = getattr(current_dataset_object, "num_shards", None)

        num_workers_dl = min(num_workers_dl, max_shards)


        data_loader = DataLoader(
            torch_dataset,
            batch_size=tokenizer_processing_batch_size,
            num_workers=num_workers_dl,
            collate_fn=collate_tokens_pytorch,
            prefetch_factor=8 if num_workers_dl > 0 else None,
            pin_memory=False,
        )

        with tqdm(
            total=tokens_to_collect_for_this_dataset,
            desc="Overall token collection",
            unit="token",
        ) as overall_progress_bar:

            for tokens_np in data_loader:
                if tokens_np.size == 0:
                    continue

                # ------------------------------------------------------
                # Stream tokens into 10-B-token chunks on disk, with optional
                # up-weighting (duplication) of small datasets.
                # ------------------------------------------------------

                newly_collected = tokens_np.size
                collected_tokens_for_current_dataset += newly_collected
                overall_collected_tokens_count += newly_collected

                # Figure out duplication factor for weighting.
                if ("sciq" in dataset_hf_name.lower() or
                    "arc-challenge" in dataset_hf_name.lower() or
                    "arc-easy" in dataset_hf_name.lower()):
                    repeat = 10  # write 10 copies total (original + 9 dupes)
                else:
                    repeat = 1

                for _ in range(repeat):
                    chunk_token_buffer.append(tokens_np)
                    tokens_in_current_chunk += newly_collected

                    # Flush full chunks immediately after each copy so buffer
                    # never grows beyond CHUNK_SIZE_TOKENS + one batch.
                    if tokens_in_current_chunk >= CHUNK_SIZE_TOKENS:
                        concatenated = np.concatenate(chunk_token_buffer)[:CHUNK_SIZE_TOKENS]
                        chunk_path = f"{data_path}_chunk_{chunk_index}.npy"
                        np.save(chunk_path, concatenated)
                        saved_chunk_paths.append(chunk_path)
                        saved_chunk_sizes.append(concatenated.size)

                        overflow = np.concatenate(chunk_token_buffer)[CHUNK_SIZE_TOKENS:]
                        chunk_token_buffer = [overflow] if overflow.size > 0 else []
                        tokens_in_current_chunk = overflow.size
                        chunk_index += 1

                overall_progress_bar.update(int(newly_collected))

                if collected_tokens_for_current_dataset >= tokens_to_collect_for_this_dataset:
                    print(
                        f"Collected {collected_tokens_for_current_dataset:,} tokens for {dataset_hf_name}. Breaking from batch loop."
                    )
                    break  # Break from the batch processing loop for the current dataset
            
            # Already handled duplication above; nothing stored long-term, buffer managed per-batch.

            # Check if the overall token collection goal has been met or exceeded
            if overall_collected_tokens_count >= total_tokens_with_margin:
                print(f"Overall token target ({total_tokens_with_margin:,}) reached with {overall_collected_tokens_count:,} tokens. Stopping further dataset processing.")
                break  # Break from the main dataset loop
        
        


    # --- End of all dataset processing ---
    # No need to reference all_collected_token_arrays any longer – tokens were saved incrementally.

    print(f"\nCollected {overall_collected_tokens_count:,} total tokens across {len(saved_chunk_paths)} chunk files.")

    # After all datasets processed, flush remaining buffer (if any) as the last chunk
    if tokens_in_current_chunk > 0 and chunk_token_buffer:
        concatenated = np.concatenate(chunk_token_buffer)
        chunk_path = f"{data_path}_chunk_{chunk_index}.npy"
        np.save(chunk_path, concatenated)
        saved_chunk_paths.append(chunk_path)
        saved_chunk_sizes.append(concatenated.size)
        chunk_index += 1

    if not saved_chunk_paths:
        print("Error: No token files were saved. Cannot proceed.")
        return

    print(f"\nCollected {overall_collected_tokens_count:,} total tokens across {len(saved_chunk_paths)} chunk files.")

    # ------------------------------------------------------------------
    # Concatenate all chunks into a single memory-mapped array (RAM-safe)
    # ------------------------------------------------------------------
    final_tokens_path = f"{data_path}_all_tokens.npy"
    all_tokens_mm = open_memmap(final_tokens_path, mode="w+", dtype=np.int32, shape=(overall_collected_tokens_count,))

    write_position = 0
    for path, size in zip(saved_chunk_paths, saved_chunk_sizes):
        chunk_arr = np.load(path, mmap_mode="r")[:size]
        all_tokens_mm[write_position:write_position + size] = chunk_arr
        write_position += size

    all_tokens_mm.flush()
    all_tokens_np = all_tokens_mm  # memmap acts like a NumPy array without loading everything
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

    # ---------------- Train / Validation split -----------------
    # Reserve a fixed 100M tokens (approx) for validation
    val_sequences_target = validation_token_target // sequence_length

    if val_sequences_target == 0:
        print("Validation split skipped (sequence_length larger than 100M tokens target).")
        train_sequences = sequences
        val_sequences = np.empty((0, sequence_length), dtype=sequences.dtype)
    else:
        # Ensure we do not request more sequences than we actually have
        val_sequences_target = min(val_sequences_target, sequences.shape[0] // 20) if sequences.shape[0] < val_sequences_target else val_sequences_target
        if val_sequences_target > sequences.shape[0]:
            print(f"Warning: Requested {val_sequences_target:,} validation sequences but only {sequences.shape[0]:,} are available. Using all sequences for training.")
            train_sequences = sequences
            val_sequences = np.empty((0, sequence_length), dtype=sequences.dtype)
        else:
            train_sequences = sequences[:-val_sequences_target]
            val_sequences = sequences[-val_sequences_target:]

    # Determine output paths (keep original name for training set)
    train_path = data_path  # original path for training sequences
    val_path = data_path.replace(".npy", "_val.npy")

    # Save arrays
    np.save(train_path, train_sequences)
    np.save(val_path, val_sequences)

    print(f"Saved {train_sequences.shape[0]:,} training sequences ({train_sequences.shape[0] * sequence_length:,} tokens) to {train_path}")
    print(f"Saved {val_sequences.shape[0]:,} validation sequences ({val_sequences.shape[0] * sequence_length:,} tokens) to {val_path}")

    return  # Explicit return, nothing follows




