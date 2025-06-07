import torch
import argparse
from pathlib import Path
from diffusers import StableDiffusion3Pipeline, AutoencoderKL, SD3Transformer2DModel
from transformers import CLIPTextModelWithProjection, T5EncoderModel
from peft import PeftModel
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def generate_image(args):
    """Loads a fine-tuned SD3 LoRA model and generates an image."""
    
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Running on CPU, which will be very slow.")
        args.device = "cpu"
    
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    
    logger.info(f"Using device: {device}, dtype: {dtype}")
    
    try:
        # --- 1. Load Base Models ---
        logger.info(f"Loading base model: {args.base_model_id}")
        
        text_encoder_l_base = CLIPTextModelWithProjection.from_pretrained(
            args.base_model_id, 
            subfolder="text_encoder", 
            torch_dtype=dtype
        )
        
        text_encoder_g_base = CLIPTextModelWithProjection.from_pretrained(
            args.base_model_id,
            subfolder="text_encoder_2", 
            torch_dtype=dtype
        )
        
        text_encoder_t5_base = T5EncoderModel.from_pretrained(
            args.base_model_id, 
            subfolder="text_encoder_3", 
            torch_dtype=dtype
        )
        
        transformer_base = SD3Transformer2DModel.from_pretrained(
            args.base_model_id, 
            subfolder="transformer", 
            torch_dtype=dtype
        )

        # --- 2. Load and Merge LoRA Adapters ---
        lora_path = Path(args.lora_checkpoint_dir)
        logger.info(f"Loading LoRA adapters from: {lora_path}")
        
        transformer = PeftModel.from_pretrained(transformer_base, str(lora_path / "transformer_lora")).merge_and_unload()
        text_encoder_l = PeftModel.from_pretrained(text_encoder_l_base, str(lora_path / "text_encoder_clip_l_lora")).merge_and_unload()
        text_encoder_g = PeftModel.from_pretrained(text_encoder_g_base, str(lora_path / "text_encoder_clip_g_lora")).merge_and_unload()
        text_encoder_t5 = PeftModel.from_pretrained(text_encoder_t5_base, str(lora_path / "text_encoder_t5_lora")).merge_and_unload()
        
        logger.info("LoRA adapters merged successfully.")

        # --- 3. Build Pipeline ---
        # Load VAE with float32 for precision, it's not a large model
        vae = AutoencoderKL.from_pretrained(
            args.base_model_id,
            subfolder="vae", 
            torch_dtype=torch.float32
        )

        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.base_model_id,
            transformer=transformer,
            text_encoder=text_encoder_l,
            text_encoder_2=text_encoder_g,
            text_encoder_3=text_encoder_t5,
            vae=vae.to(dtype=dtype),
            torch_dtype=dtype
        )
        pipe.to(device)
        logger.info("Inference pipeline created and moved to device.")

        # --- 4. Generate Image ---
        seed = args.seed if args.seed is not None else torch.randint(0, 1_000_000, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)
        
        logger.info(f"\n--- Generating Image ---")
        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"Negative Prompt: {args.negative_prompt}")
        logger.info(f"Settings: Steps={args.steps}, Guidance={args.guidance_scale}, Seed={seed}, Resolution={args.resolution}")
        
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type=="cuda"):
            image = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                width=args.resolution,
                height=args.resolution,
                generator=generator,
            ).images[0]

        # --- 5. Save Image ---
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        image.save(output_path)
        logger.info(f"Image saved successfully to: {output_path}")

    except Exception as e:
        logger.error(f"An error occurred during inference: {e}", exc_info=True)
        
    finally:
        if 'pipe' in locals():
            del pipe
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Inference finished.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate an image using a fine-tuned SD3 LoRA model.")
    
    # Paths
    parser.add_argument("--base_model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium", help="Base model ID from Hugging Face Hub.")
    parser.add_argument("--lora_checkpoint_dir", type=str, required=True, help="Path to the directory containing the saved LoRA adapters (e.g., 'outputs/final_model').")
    parser.add_argument("--output_path", type=str, default="outputs/images/generated_image.png", help="Path to save the generated image.")

    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True, help="The text prompt to generate an image from.")
    parser.add_argument("--negative_prompt", type=str, default="ugly, blurry, deformed, bad anatomy, extra limbs, watermark, signature", help="The negative prompt.")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale (CFG).")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution (must be square).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. If None, a random seed is used.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on ('cuda' or 'cpu').")
    
    args = parser.parse_args()
    generate_image(args)