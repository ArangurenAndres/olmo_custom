"""
Data preparation script for OLMo models.

This script handles downloading and tokenizing the entire Wikipedia dataset
for later use in training OLMo models.
"""

import os
import argparse
import numpy as np
from utils.environment import setup_environment
from utils.paths import setup_data_directories
from data_utils.dataset import _download_and_tokenize_wiki
from data_utils.validate_data_prep import validate_tokenized_data
from data_utils.tokenizer import get_tokenizer_config
from configs.argparser import parse_args_data_prep


def main():
    # Parse command line arguments
    args = parse_args_data_prep()

    print(f"Starting data preparation for OLMo training")
    
    # Setup environment variables and directories
    setup_environment()
    data_dir = setup_data_directories(args)
    wiki_data_path = os.path.join(data_dir, args.output_file)

    # Configure tokenizer and report vocabulary size
    tokenizer_config = get_tokenizer_config()

    # Download and tokenize Wikipedia
    _download_and_tokenize_wiki(wiki_data_path, args.sequence_length, args.target_tokens)
    
    # Validate the tokenized data
    validate_tokenized_data(wiki_data_path, args.sequence_length)

    print(f"Data preparation complete!")

if __name__ == "__main__":
    main()
