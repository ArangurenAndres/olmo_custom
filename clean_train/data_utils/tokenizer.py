import os
from transformers import AutoTokenizer
from olmo_core.data import TokenizerConfig

def get_tokenizer_config():
    """Configure the tokenizer."""
    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    print(f"Configured tokenizer with vocab size {tokenizer_config.padded_vocab_size()}")
    print(f"Tokenizer config: {tokenizer_config}")
    return tokenizer_config

def load_tokenizer(paths):
    """Load the HuggingFace tokenizer."""
    return AutoTokenizer.from_pretrained(
        "allenai/gpt-neox-olmo-dolma-v1_5",
        cache_dir=os.path.join(os.environ["TRANSFORMERS_CACHE"], "tokenizers")
    ) 