import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from olmo_core.data import (
    TokenizerConfig, NumpyDatasetConfig, NumpyDataLoaderConfig, NumpyDatasetType
)

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
