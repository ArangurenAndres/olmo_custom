import os
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_environment():
    """
    Set up environment variables for OLMo training using paths from config.yaml.
    """
    config = load_config()

    # Get cache paths from config, making them absolute from CWD
    base_project_dir = os.getcwd() 
    cfg_cache_paths = config["hf_cache_paths"]

    # Set environment variables for cache directories
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(base_project_dir, cfg_cache_paths["transformers"])
    os.environ["HF_DATASETS_CACHE"] = os.path.join(base_project_dir, cfg_cache_paths["datasets"])
    os.environ["HF_HOME"] =  os.path.join(base_project_dir, cfg_cache_paths["huggingface"]) 
    
    # Create cache directories
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    os.environ["WANDB_API_KEY"] = config["WANDB_API_KEY"]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"


   # print(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
   # print(f"HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")
   # print(f"HF_HOME: {os.environ['HF_HOME']}")
