import argparse
import logging
import torch
import bitsandbytes as bnb
import math
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.config_loader import load_config
from src.utils import setup_environment, print_training_summary, cleanup_memory
from src.data_processing import SD3ImageCaptionDataset, collate_fn
from src.model_setup import initialize_models_and_tokenizers, setup_lora_training
from src.trainer import Trainer

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main(args):
    """Main function to orchestrate the training process."""
    
    try:
        # 1. Load Configuration
        logger.info("Loading configuration...")
        config = load_config(args.config_path)

        # 2. Initialize Accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision=config.training.mixed_precision,
            log_with="tensorboard",
            project_dir=f"{config.paths.output_dir}/logs"
        )
        
        logger.info(f"Accelerator initialized for {accelerator.device} with mixed precision: {accelerator.mixed_precision}")

        # 3. Setup Environment
        setup_environment(config, accelerator)
        print_training_summary(config, accelerator)
        
        # 4. Load Models and Tokenizers
        components = initialize_models_and_tokenizers(config, accelerator)
        
        # 5. Setup LoRA Models
        lora_models = setup_lora_training(components, config, accelerator)
        
        # 6. Create Dataset and DataLoader
        logger.info("Creating dataset and dataloader...")
        tokenizers = {
            'clip_l': components['tokenizer_clip_l'],
            'clip_g': components['tokenizer_clip_g'],
            't5': components['tokenizer_t5'],
        }
        
        train_dataset = SD3ImageCaptionDataset(config, tokenizers, accelerator)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )

        # 7. Setup Optimizer
        logger.info("Setting up optimizer...")
        params_to_optimize = [
            {"params": lora_models['transformer'].parameters(), "lr": config.optimizer.transformer_lr},
            {"params": lora_models['text_encoder_clip_l'].parameters(), "lr": config.optimizer.text_encoder_lr},
            {"params": lora_models['text_encoder_clip_g'].parameters(), "lr": config.optimizer.text_encoder_lr},
            {"params": lora_models['text_encoder_t5'].parameters(), "lr": config.optimizer.text_encoder_t5_lr},
        ]
        
        optimizer = bnb.optim.AdamW8bit(
            params_to_optimize,
            betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
            weight_decay=config.optimizer.adam_weight_decay,
            eps=config.optimizer.adam_epsilon
        )

        # 8. Setup Learning Rate Scheduler
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
        max_train_steps = config.training.num_epochs * num_update_steps_per_epoch
        num_warmup_steps = int(max_train_steps * config.lr_scheduler.warmup_steps_ratio)

        lr_scheduler = get_scheduler(
            name=config.lr_scheduler.type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        # 9. Prepare everything with Accelerator
        logger.info("Preparing components with Accelerator...")
        # Prepare models individually to handle them in a dict
        prepared_lora_models = {}
        for name, model in lora_models.items():
            prepared_lora_models[name] = accelerator.prepare(model)
        
        prepared_optimizer, prepared_dataloader, prepared_scheduler = accelerator.prepare(
            optimizer, train_dataloader, lr_scheduler
        )
        
        # VAE needs to be on the correct device but is not trained
        vae = components['vae'].to(accelerator.device, dtype=torch.bfloat16)

        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {config.training.num_epochs}")
        print(f"  Batch size = {config.training.batch_size}")
        print(f"  Total train batch size (w. parallel, dist. & accum.) = {config.training.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps}")
        print(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_train_steps}")
        print(f"  Number of warmup steps = {num_warmup_steps}")
        print(f"  Saving a checkpoint every {config.logging.save_steps} steps.")


        # 10. Instantiate and Run Trainer
        trainer = Trainer(
            config=config,
            accelerator=accelerator,
            lora_models=prepared_lora_models,
            vae=vae,
            optimizer=prepared_optimizer,
            dataloader=prepared_dataloader,
            lr_scheduler=prepared_scheduler,
            components_for_inference=components,
            max_train_steps=max_train_steps,
        )
        
        logger.info("Starting training process...")
        
        trainer.train()
        
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        
    finally:
        logger.info("Running final cleanup...")
        cleanup_memory()
        logger.info("Cleanup finished.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion 3 with LoRA.")
    
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/training_config.yaml",
        help="Path to the training configuration YAML file."
    )
    
    args = parser.parse_args()
    main(args)