import os
import yaml

def load_config(config_filename="config.yaml"):
    """
    Loads a YAML configuration file located one level above this script.
    Args:
        config_filename (str): Name of the config YAML file.
    Returns:
        dict: Parsed configuration as a dictionary.
    """
    # Go up one level from utils/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_filename)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config