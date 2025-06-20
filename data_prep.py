"""
Data preparation script for OLMo models.

This script handles downloading and tokenizing the entire Wikipedia dataset
for later use in training OLMo models.
"""

import os
import numpy as np
from data_utils.download_and_tokenizeV2 import download_and_tokenize
from data_utils.validate_data_prep import validate_tokenized_data
from transformers import AutoTokenizer
from olmo_core.data import TokenizerConfig
import yaml
from utils.setup_env_variables import setup_environment
from utils.load_config import load_config


def main():
    # Setup environment (cache paths)
    setup_environment() 

    config = load_config()

    print(f"Starting data preparation for OLMo training")

   # print(config["data_dir"])
   # print(config["data_preparation"]["output_file_name"])
   # print(config["sequence_length"])
   # print(config["data_preparation"]["total_tokens_to_collect"])
    
    # Set up data dir, data_path of .npy file, set up environment variables
    data_dir = config["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    
    # Configure tokenizer and report vocabulary size
    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Download and tokenize the data
    data_path = os.path.join(data_dir, config["data_preparation"]["output_file_name"] + ".npy")
    download_and_tokenize(
            data_path=data_path,
            sequence_length=config["sequence_length"],
            total_tokens_with_margin=config["data_preparation"]["total_tokens_to_collect"],
            tokenizer_processing_batch_size=config["data_preparation"]["tokenizer_processing_batch_size"],
            dataset_proportions=config["data_preparation"]["dataset_proportions"],
            validation_token_target=config["data_preparation"].get("validation_token_target", 10_000_000)
        )
    
    # Validate the tokenized data
    validate_tokenized_data(os.path.join(data_dir, config["data_preparation"]["output_file_name"] + ".npy"), config["sequence_length"])
    validate_tokenized_data(os.path.join(data_dir, config["data_preparation"]["output_file_name"] + "_val.npy"), config["sequence_length"])

    print(f"Data preparation complete!")

if __name__ == "__main__":
    main()
