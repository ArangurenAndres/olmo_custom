"""
Command line argument parsing module.
"""

import argparse


def parse_args_train():
    """
    Parse command line arguments for OLMo training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train an OLMo model on text data")
    
    # Training parameters

    ### set up for testing not for training - 10 steps, 2 batch size, 1 micro batch size 
    parser.add_argument("--gpu", type=int, default=0, 
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--steps", type=int, default=10, 
                        help="Total training steps (default: 100)")
    parser.add_argument("--batch-size", type=int, default=2, 
                        help="Effective GLOBAL batch size across all devices after gradient accumulation (default: 8)")
    parser.add_argument("--micro-batch-size", type=int, default=1,
                        help="Batch size processed by a single device in one forward/backward pass (adjust based on GPU memory) (default: 2)")
    """
    GRADIENT ACCUMULATION AND MICRO BATCH SIZE explained:

    batch size - should be the total number of tokens processed in a single update step !  - should be around 0.5M tokens
    micro batch size - the batch size that fits on the GPU - 

    batch size = 600 means 600 * 1024 = 614400 tokens on one update step
    micro batch size = 24 means 24 * 1024 = 24576 tokens processed in one forward/backward pass

    TransformerTrainModuleConfig takes the micro batch IN the tokens (not the sequences) 
    """

    # Inference parameters
    parser.add_argument("--inference-interval", type=int, default=50, 
                       help="Run inference every N steps (default: 200)")
    parser.add_argument("--inference-prompt", type=str, default="Dutch is a very ", 
                       help="Prompt to use for inference (default: 'Dutch is a very ')")

    # wandb parameters
    parser.add_argument("--wandb-project", type=str, default="olmo-training",
                        help="Wandb project name (default: 'olmo-training')")
    parser.add_argument("--wandb-name", type=str, default="olmo-train",
                        help="Wandb run name (default: 'olmo-train')")
                        

    # data prep parameters
    parser.add_argument("--sequence-length", type=int, default=1024,
                        help="Sequence length for tokenization (default: 1024)")
                    
    parser.add_argument("--data-dir", type=str, default=None, 
                    help="Data directory to use")
    
    return parser.parse_args() 

def parse_args_data_prep():
    """
    Parse command line arguments for data preparation.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Prepare data for OLMo training")
    
    parser.add_argument("--sequence-length", type=int, default=1024,
                        help="Sequence length for tokenization (default: 1024)")
    parser.add_argument("--output-file", type=str, default="wiki_tokens.npy",
                        help="Output filename for tokenized data (default: wiki_tokens.npy)")
    parser.add_argument("--target-tokens", type=int, default=300_000,
                        help="Target number of tokens to collect ")
    parser.add_argument("--data-dir", type=str, default=None, 
                    help="Data directory to use")
    
    return parser.parse_args()