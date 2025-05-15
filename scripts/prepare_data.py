#!/usr/bin/env python3
"""
Data preprocessing script for OLMo training with Wikipedia.
This script downloads, tokenizes, and saves Wikipedia data for training.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataloader import prepare_wikipedia_data
from src.utils.logging import setup_logging, log_dict

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare Wikipedia data for OLMo training")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="Path to configuration file (YAML)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./data", 
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--tokenizer", 
        type=str, 
        default="allenai/gpt-neox-olmo-dolma-v1_5", 
        help="Tokenizer to use"
    )
    parser.add_argument(
        "--sequence-length", 
        type=int, 
        default=1024, 
        help="Sequence length for training"
    )
    parser.add_argument(
        "--max-articles", 
        type=int, 
        default=None, 
        help="Maximum number of articles to process"
    )
    parser.add_argument(
        "--target-tokens", 
        type=int, 
        default=None, 
        help="Target number of tokens to collect"
    )
    parser.add_argument(
        "--wiki-version", 
        type=str, 
        default="20220301.en", 
        help="Wikipedia version to use"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite existing data"
    )
    parser.add_argument(
        "--use-small", 
        action="store_true", 
        help="Use a small subset of data (1000 articles)"
    )
    parser.add_argument(
        "--log-file", 
        type=str, 
        default=None, 
        help="Path to log file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(args.log_file, log_level)
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        log_dict(config, logger, prefix="Loaded configuration:")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine parameters, with CLI args overriding config
    output_dir = args.output_dir or config.get("data_dir", "./data")
    tokenizer_name = args.tokenizer or config.get("tokenizer_name", "allenai/gpt-neox-olmo-dolma-v1_5")
    sequence_length = args.sequence_length or config.get("sequence_length", 1024)
    
    # Determine data size parameters
    if args.use_small:
        max_articles = 1000
        min_total_tokens = None
    else:
        max_articles = args.max_articles or config.get("max_articles", None)
        min_total_tokens = args.target_tokens or config.get("min_total_tokens", None)
        
        # If neither is specified, set a reasonable default
        if max_articles is None and min_total_tokens is None:
            # Default: Enough tokens for ~10k training steps with batch size 32
            min_total_tokens = 10_000 * 32 * sequence_length
    
    # Log configuration
    logger.info(f"Preparing Wikipedia data:")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Tokenizer: {tokenizer_name}")
    logger.info(f"  Sequence length: {sequence_length}")
    logger.info(f"  Max articles: {max_articles if max_articles is not None else 'all'}")
    logger.info(f"  Min total tokens: {min_total_tokens if min_total_tokens is not None else 'not specified'}")
    logger.info(f"  Wikipedia version: {args.wiki_version}")
    logger.info(f"  Overwrite: {args.overwrite}")
    
    # Process the data
    try:
        token_file = prepare_wikipedia_data(
            output_dir=output_dir,
            sequence_length=sequence_length,
            tokenizer_name=tokenizer_name,
            cache_dir=os.path.join(output_dir, "tokenizer_cache"),
            max_articles=max_articles,
            min_total_tokens=min_total_tokens,
            wikipedia_version=args.wiki_version,
            output_file="wiki_tokens.npy",
            overwrite=args.overwrite
        )
        
        logger.info(f"Successfully prepared data: {token_file}")
        logger.info("Data preparation complete!")
        
        # Write out a config file for training
        data_config = {
            "data_dir": output_dir,
            "sequence_length": sequence_length,
            "tokenizer_name": tokenizer_name
        }
        
        config_path = os.path.join(output_dir, "data_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(data_config, f)
        
        logger.info(f"Saved data configuration to {config_path}")
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())