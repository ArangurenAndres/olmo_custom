import os
import yaml
import argparse

def load_config():
    """
    Loads a YAML configuration file.
    The configuration directory is expected to be '../configs/' relative to this script.
    Uses argparse to allow specifying a config name via '--config <name>',
    which loads '<name>.yaml'. Defaults to 'config.yaml' if not specified.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}. Searched for --config '{config_name}'.")
    
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    return config_data