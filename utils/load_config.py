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
    parser = argparse.ArgumentParser(description="Load a specific YAML configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='config',
        help="Name of the configuration file (without .yaml extension) to load from the 'configs' directory. E.g., 'flan' for 'flan.yaml'."
    )
    
    args, _ = parser.parse_known_args()

    config_name = args.config
    config_filename = f"{config_name}.yaml"

    configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs'))
    config_path = os.path.join(configs_dir, config_filename)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}. Searched for --config '{config_name}'.")
    
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    return config_data