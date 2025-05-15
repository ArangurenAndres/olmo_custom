"""
Data preparation script for OLMo models.

This script handles downloading and tokenizing the entire Wikipedia dataset
for later use in training OLMo models.
"""

import os
import numpy as np
from data_utils.download_and_tokenize import _download_and_tokenize_wiki
from data_utils.validate_data_prep import validate_tokenized_data
from transformers import AutoTokenizer
from olmo_core.data import TokenizerConfig
import yaml
from utils.setup_env_variables import setup_environment


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

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
    wiki_data_path = os.path.join(data_dir, config["data_preparation"]["output_file_name"])

    # Configure tokenizer and report vocabulary size
    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()

    # Download and tokenize Wikipedia
    _download_and_tokenize_wiki(wiki_data_path, config["sequence_length"], config["data_preparation"]["total_tokens_to_collect"])
    
    # Validate the tokenized data
    validate_tokenized_data(wiki_data_path, config["sequence_length"])

    print(f"Data preparation complete!")

if __name__ == "__main__":
    main()
