import torch
import os
from pathlib import Path
from typing import Dict, Any
from accelerate import Accelerator
from peft import PeftModel
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from transformers import CLIPTextModelWithProjection, T5EncoderModel
import logging

from .utils import cleanup_memory

# Setup logger
logger = logging.getLogger(__name__)

def run_validation_inference(
    config: Dict[str, Any],
    accelerator: Accelerator,
    pipeline_components: Dict[str, Any],
    lora_checkpoint_dir: str,
    output_filename_suffix: str = ""
):
    
    """
    Runs a test inference using a saved LoRA checkpoint during training.
    This function is designed to be called only on the main process.
    """
    
    if not accelerator.is_main_process:
        return

    logger.info("Running validation inference...")
    
    try:
        
        if config.training.mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            
        else:
            torch.float16
            
        inference_device = accelerator.device
        model_id = config.model.id
        
        # Load base models on the fly to avoid keeping them in memory
        text_encoder_l_base = CLIPTextModelWithProjection.from_pretrained(
            model_id, 
            subfolder="text_encoder", 
            torch_dtype=dtype
        )
        
        text_encoder_g_base = CLIPTextModelWithProjection.from_pretrained(
            model_id, 
            subfolder="text_encoder_2", 
            torch_dtype=dtype
        )
        
        text_encoder_t5_base = T5EncoderModel.from_pretrained(
            model_id, 
            subfolder="text_encoder_3", 
            torch_dtype=dtype
        )
        
        transformer_base = SD3Transformer2DModel.from_pretrained(
            model_id, 
            subfolder="transformer", 
            torch_dtype=dtype
        )

        # Check if LoRA adapter paths exist
        lora_transformer_path = Path(lora_checkpoint_dir) / "transformer_lora"
        if not lora_transformer_path.exists():
            logger.warning(f"LoRA adapters not found at {lora_transformer_path}. Skipping inference.")
            return

        # Load PEFT models
        transformer_peft = PeftModel.from_pretrained(
            transformer_base, 
            str(lora_transformer_path)
        )
        
        text_encoder_l_peft = PeftModel.from_pretrained(
            text_encoder_l_base, 
            str(Path(lora_checkpoint_dir) / "text_encoder_clip_l_lora")
        )
        
        text_encoder_g_peft = PeftModel.from_pretrained(
            text_encoder_g_base, 
            str(Path(lora_checkpoint_dir) / "text_encoder_clip_g_lora")
        )
        
        text_encoder_t5_peft = PeftModel.from_pretrained(
            text_encoder_t5_base, 
            str(Path(lora_checkpoint_dir) / "text_encoder_t5_lora")
        )

        # Merge LoRA weights
        transformer = transformer_peft.merge_and_unload()
        text_encoder_l = text_encoder_l_peft.merge_and_unload()
        text_encoder_g = text_encoder_g_peft.merge_and_unload()
        text_encoder_t5 = text_encoder_t5_peft.merge_and_unload()
        
        # Use VAE and scheduler from the initial components
        vae = pipeline_components['vae']
        scheduler = pipeline_components['scheduler']
        
        # Build the pipeline
        pipe = StableDiffusion3Pipeline(
            transformer=transformer, 
            text_encoder=text_encoder_l,
            text_encoder_2=text_encoder_g, 
            text_encoder_3=text_encoder_t5,
            vae=vae, 
            scheduler=scheduler,
            tokenizer=None, 
            tokenizer_2=None, 
            tokenizer_3=None # Tokenizers already handled
        )
        pipe.to(inference_device)

        # Generate image
        seed = torch.randint(0, 1000000, (1,)).item()
        generator = torch.Generator(device=inference_device).manual_seed(seed)
        
        logger.info(f"Generating image with prompt: '{config.validation.prompt}'")
        logger.info(f"Seed: {seed}, Steps: {config.validation.steps}, Guidance: {config.validation.guidance_scale}")

        with torch.autocast(device_type=inference_device.type, dtype=dtype, enabled=inference_device.type=="cuda"):
            image = pipe(
                prompt=config.validation.prompt,
                negative_prompt=config.validation.negative_prompt,
                num_inference_steps=config.validation.steps,
                guidance_scale=config.validation.guidance_scale,
                width=config.validation.resolution,
                height=config.validation.resolution,
                generator=generator,
            ).images[0]

        # Save image
        output_dir = Path(config.paths.output_dir) / "validation_images"
        output_dir.mkdir(exist_ok=True)
        
        image_path = output_dir / f"validation{output_filename_suffix}.png"
        image.save(image_path)
        
        logger.info(f"Validation image saved to: {image_path}")

    except Exception as e:
        logger.error(f"Error during validation inference: {e}", exc_info=True)
        
    finally:
        if 'pipe' in locals():
            del pipe
            
        if 'transformer' in locals():
            del transformer, text_encoder_l, text_encoder_g, text_encoder_t5
            
        cleanup_memory()
        logger.info("Validation inference finished and memory cleaned.")