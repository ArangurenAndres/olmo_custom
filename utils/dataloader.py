import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from olmo_core.data import (
    TokenizerConfig, NumpyDatasetConfig, NumpyDataLoaderConfig, NumpyDatasetType
)

SEQUENCE_LENGTH = 1024

def prepare_data(data_dir, total_sequences):
    os.makedirs(data_dir, exist_ok=True)
    token_file = os.path.join(data_dir, "wiki_tokens.npy")

    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")

    if not os.path.exists(token_file):
        dataset = load_dataset("wikipedia", "20220301.en", split="train")
        all_tokens = []
        for article in tqdm(dataset.select(range(1000)), desc="Tokenizing"):
            tokens = tokenizer.encode(article["text"])
            tokens = [t for t in tokens if t != 0]
            all_tokens.extend(tokens)
            if len(all_tokens) >= total_sequences * SEQUENCE_LENGTH:
                break
        tokens = all_tokens[:total_sequences * SEQUENCE_LENGTH]
        np.save(token_file, np.array(tokens, dtype=np.int32).reshape(-1, SEQUENCE_LENGTH))

    dataset_config = NumpyDatasetConfig(
        tokenizer=tokenizer_config,
        name=NumpyDatasetType.fsl,
        paths=[token_file],
        sequence_length=SEQUENCE_LENGTH,
        work_dir=os.path.join(data_dir, "dataset_work")
    )
    dataset = dataset_config.build()

    loader_config = NumpyDataLoaderConfig(
        global_batch_size=SEQUENCE_LENGTH,
        seed=42,
        num_workers=0,
    )
    loader = loader_config.build(dataset)

    return loader, tokenizer_config
