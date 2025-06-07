import yaml
from pathlib import Path
from typing import Any

class DotDict(dict):
    """A dictionary that allows dot notation access."""
    
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

def load_config(config_path: str) -> DotDict:
    """
    Loads a YAML configuration file and returns it as a DotDict object.
    
    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        DotDict: A nested dictionary with dot notation access.
    """
    
    path = Path(config_path)
    
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(path, 'r') as f:
        config_data = yaml.safe_load(f)
        
    return DotDict(config_data)