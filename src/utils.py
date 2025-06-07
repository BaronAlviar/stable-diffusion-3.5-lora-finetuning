import torch
import gc
import json
import logging
from pathlib import Path
from typing import Union, Dict
from accelerate import Accelerator
from accelerate.utils import set_seed

logger = logging.getLogger(__name__)

def setup_environment(config: Dict, accelerator: Accelerator):
    
    """Sets up the training environment, including seed and output directories."""
    
    set_seed(config.training.seed)
    if accelerator.is_main_process:
        
        output_dir = Path(config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = output_dir / "training_config.json"
        
        try:
            
            with open(config_path, "w") as f:
                serializable_config = {k: dict(v) if hasattr(v, 'items') else v for k, v in config.items()}
                json.dump(serializable_config, f, indent=4)
                
            logger.info(f"Training config saved to: {config_path}")
            
        except Exception as e:
            logger.warning(f"Could not save training config: {e}")
            
        logger.info(f"Output directory created at: {output_dir}")

def cleanup_memory():
    
    """Performs garbage collection and clears CUDA cache."""
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_checkpoint(
    lora_models: Dict[str, torch.nn.Module], 
    accelerator: Accelerator, 
    output_dir: str,
    step: Union[int, str]):
    
    """Saves the LoRA model checkpoints."""
    
    if not accelerator.is_main_process:
        return

    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator.print(f"\nSaving checkpoint to {checkpoint_dir}...")
    
    for name, model in lora_models.items():
        unwrapped_model = accelerator.unwrap_model(model)
        lora_save_path = checkpoint_dir / f"{name}_lora"
        unwrapped_model.save_pretrained(str(lora_save_path))
    
    accelerator.print(f"Checkpoint saved successfully.")

def print_training_summary(config: Dict, accelerator: Accelerator):
    """Prints a summary of the training configuration."""
    
    if accelerator.is_main_process:
        
        logger.info("\n" + "="*20 + " Training Configuration Summary " + "="*20)
        config_dict = {k: dict(v) if hasattr(v, 'items') else v for k, v in config.items()}
        
        logger.info(json.dumps(config_dict, indent=2))
        logger.info("="*70 + "\n")