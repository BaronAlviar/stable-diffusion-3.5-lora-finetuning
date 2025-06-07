import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Dict, List
from accelerate import Accelerator

def load_metadata(dataset_path: Path, accelerator: Accelerator) -> List[Dict]:
    """
    Loads dataset metadata from a 'metadata.jsonl' file.
    """
    
    metadata_file = dataset_path / "metadata.jsonl"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
    image_dir = dataset_path / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    entries = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                
                data = json.loads(line)
                file_name, text = data.get("file_name"), data.get("text")
                
                if file_name and text:
                    image_path = image_dir / file_name
                    
                    if image_path.exists():
                        entries.append({"image_path": image_path, "caption": text})
                        
                    elif accelerator.is_main_process:
                        accelerator.print(f"Warning: Image file '{file_name}' not found. Skipping.")
                        
                elif accelerator.is_main_process:
                    accelerator.print(f"Warning: Line {i+1} in metadata is missing 'file_name' or 'text'.")
                    
            except json.JSONDecodeError as e:
                if accelerator.is_main_process:
                    accelerator.print(f"Warning: Invalid JSON on line {i+1}. Error: {e}")

    if not entries:
        raise ValueError("No valid image-caption pairs found. Check metadata file and image directory.")
        
    return entries

class SD3ImageCaptionDataset(Dataset):
    """PyTorch Dataset for Stable Diffusion 3 image-caption pairs."""
    
    def __init__(self, config: Dict, tokenizers: Dict, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.dataset_path = Path(config.paths.dataset_path)
        
        if accelerator.is_main_process:
            accelerator.print(f"Loading metadata from: {self.dataset_path}")
            
        self.entries = load_metadata(self.dataset_path, accelerator)
        
        if accelerator.is_main_process:
            accelerator.print(f"Successfully loaded {len(self.entries)} samples.")
        
        self.tokenizer_clip_l = tokenizers['clip_l']
        self.tokenizer_clip_g = tokenizers['clip_g']
        self.tokenizer_t5 = tokenizers['t5']
        
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        
        entry = self.entries[idx]
        image_path, caption = entry["image_path"], entry["caption"]
        
        try:
            image = Image.open(image_path).convert("RGB")
            res = self.config.data.resolution
            image = image.resize((res, res), Image.LANCZOS)

            # Normalize to [-1, 1]
            image_np = np.array(image, dtype=np.float32)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1) / 127.5 - 1.0

        except Exception as e:
            if self.accelerator.is_main_process:
                self.accelerator.print(f"Error loading image {image_path}: {e}. Using placeholder.")
                
            image_tensor = torch.zeros((3, self.config.data.resolution, self.config.data.resolution))
            caption = "placeholder caption due to image error" # Avoid training on bad data

        # Tokenize captions
        inputs_1 = self.tokenizer_clip_l(
            caption, 
            max_length=self.config.data.tokenizer_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        inputs_2 = self.tokenizer_clip_g(
            caption, 
            max_length=self.config.data.tokenizer_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        inputs_3 = self.tokenizer_t5(
            caption, 
            max_length=self.config.data.tokenizer_t5_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        return {
            "pixel_values": image_tensor,
            "input_ids_1": inputs_1.input_ids.squeeze(0), "attention_mask_1": inputs_1.attention_mask.squeeze(0),
            "input_ids_2": inputs_2.input_ids.squeeze(0), "attention_mask_2": inputs_2.attention_mask.squeeze(0),
            "input_ids_3": inputs_3.input_ids.squeeze(0), "attention_mask_3": inputs_3.attention_mask.squeeze(0),
        }

def collate_fn(examples: List[Dict]) -> Dict:
    """Custom collate function to stack tensors from a list of samples."""
    
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    
    input_ids_1 = torch.stack([e["input_ids_1"] for e in examples])
    attention_mask_1 = torch.stack([e["attention_mask_1"] for e in examples])
    
    input_ids_2 = torch.stack([e["input_ids_2"] for e in examples])
    attention_mask_2 = torch.stack([e["attention_mask_2"] for e in examples])
    
    input_ids_3 = torch.stack([e["input_ids_3"] for e in examples])
    attention_mask_3 = torch.stack([e["attention_mask_3"] for e in examples])
    
    return {
        "pixel_values": pixel_values,
        "input_ids_1": input_ids_1, "attention_mask_1": attention_mask_1,
        "input_ids_2": input_ids_2, "attention_mask_2": attention_mask_2,
        "input_ids_3": input_ids_3, "attention_mask_3": attention_mask_3,
    }