import torch
from transformers import CLIPTextModelWithProjection, T5EncoderModel, CLIPTokenizer
from transformers import BitsAndBytesConfig, T5TokenizerFast
from diffusers import SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler, AutoencoderKL
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from accelerate import Accelerator
import logging
from typing import Dict, List, Any


# Setup logger
logger = logging.getLogger(__name__)


def _get_sd3_transformer_target_modules(model: SD3Transformer2DModel) -> List[str]:
    """Dynamically finds all linear layers in the SD3 Transformer for LoRA."""
    
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            target_modules.add(name)
            
    return sorted(list(target_modules))

def _get_clip_target_modules(model: CLIPTextModelWithProjection) -> List[str]:
    """Dynamically finds all linear layers in the CLIP Text Encoder for LoRA."""
    
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            target_modules.add(name)
            
    # Ensure the projection layer is included if it's a Linear layer
    if hasattr(model, 'text_projection') and isinstance(model.text_projection, torch.nn.Linear):
        target_modules.add("text_projection")
        
    return sorted(list(target_modules))

def _get_t5_target_modules(model: T5EncoderModel) -> List[str]:
    """Dynamically finds all linear layers in the T5 Text Encoder for LoRA."""
    
    target_modules = set()
    for name, module in model.named_modules():
        # T5 uses specific Linear layers in its blocks
        if isinstance(module, torch.nn.Linear):
             target_modules.add(name)
             
    return sorted(list(target_modules))


def initialize_models_and_tokenizers(config: Dict, accelerator: Accelerator) -> Dict[str, Any]:
    """
    Loads all necessary models, tokenizers, and the scheduler from Hugging Face.
    """
    
    accelerator.print("Initializing models, tokenizers, and scheduler...")
    
    quant_config = None
    if config.model.use_quantization:
        
        accelerator.print("Quantization is enabled. Setting up BitsAndBytesConfig.")
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model_dtype = torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float16
    model_id = config.model.id

    # --- Load Tokenizers ---
    tokenizer_clip_l = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_clip_g = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    tokenizer_t5 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3", legacy=False)

    # --- Load Models ---
    text_encoder_clip_l = CLIPTextModelWithProjection.from_pretrained(
        model_id, 
        subfolder="text_encoder", 
        quantization_config=quant_config,
        torch_dtype=model_dtype, 
        low_cpu_mem_usage=True,
    )
    
    text_encoder_clip_g = CLIPTextModelWithProjection.from_pretrained(
        model_id, 
        subfolder="text_encoder_2", 
        quantization_config=quant_config,
        torch_dtype=model_dtype, 
        low_cpu_mem_usage=True,
    )
    
    text_encoder_t5 = T5EncoderModel.from_pretrained(
        model_id, 
        subfolder="text_encoder_3", 
        quantization_config=quant_config,
        torch_dtype=model_dtype, 
        low_cpu_mem_usage=True,
    )
    
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, 
        subfolder="transformer", 
        quantization_config=quant_config,
        torch_dtype=model_dtype, 
        low_cpu_mem_usage=True,
    )
    
    # VAE is not trained, so it's loaded in full precision for stability
    vae = AutoencoderKL.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=model_dtype
    )
    
    vae.requires_grad_(False)

    # --- Load Scheduler ---
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    accelerator.print("All base components loaded successfully.")
    
    return {
        "text_encoder_clip_l": text_encoder_clip_l, 
        "text_encoder_clip_g": text_encoder_clip_g,
        "text_encoder_t5": text_encoder_t5, 
        "transformer": transformer, 
        "vae": vae,
        "tokenizer_clip_l": tokenizer_clip_l, 
        "tokenizer_clip_g": tokenizer_clip_g,
        "tokenizer_t5": tokenizer_t5, 
        "scheduler": scheduler
    }


def setup_lora_training(
    components: Dict, 
    config: Dict, 
    accelerator: Accelerator) -> Dict[str, PeftModel]:
    
    """
    Applies LoRA configuration to the trainable models (transformer and text encoders).
    """
    
    accelerator.print("Setting up LoRA for trainable models...")
    
    # --- Helper to enable gradient checkpointing robustly ---
    def _enable_gradient_checkpointing(model):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    # --- Define LoRA configs ---
    lora_config_transformer = LoraConfig(
        r=config.lora.rank, 
        lora_alpha=config.lora.alpha,
        target_modules=_get_sd3_transformer_target_modules(components['transformer']),
        lora_dropout=config.lora.dropout, 
        bias="none"
    )
    
    lora_config_clip = LoraConfig(
        r=config.lora.rank, 
        lora_alpha=config.lora.alpha,
        target_modules=_get_clip_target_modules(components['text_encoder_clip_l']),
        lora_dropout=config.lora.dropout, 
        bias="none"
    )
    
    lora_config_t5 = LoraConfig(
        r=config.lora.rank, 
        lora_alpha=config.lora.alpha,
        target_modules=_get_t5_target_modules(components['text_encoder_t5']),
        lora_dropout=config.lora.dropout, 
        bias="none"
    )
    
    # --- Prepare models for k-bit training if quantized ---
    models_to_wrap = {
        "transformer": (components['transformer'], lora_config_transformer),
        "text_encoder_clip_l": (components['text_encoder_clip_l'], lora_config_clip),
        "text_encoder_clip_g": (components['text_encoder_clip_g'], lora_config_clip), # Use same CLIP config
        "text_encoder_t5": (components['text_encoder_t5'], lora_config_t5)
    }
    
    lora_models = {}
    
    for name, (model, lora_config) in models_to_wrap.items():
        model.requires_grad_(False)
        
        if config.model.use_quantization:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        
        lora_model = get_peft_model(model, lora_config)
        _enable_gradient_checkpointing(lora_model)
        
        lora_models[name] = lora_model
        
        if accelerator.is_main_process:
            lora_model.print_trainable_parameters()
            
    return lora_models