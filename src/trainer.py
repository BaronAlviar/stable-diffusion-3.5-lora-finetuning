import torch
import math
from tqdm.auto import tqdm
from accelerate import Accelerator
from typing import Dict, Optional, Union
import logging
from pathlib import Path

from . import utils
from .inference_during_training import run_validation_inference

# Setup logger
logger = logging.getLogger(__name__)


def _add_noise_rectified_flow(x0: torch.Tensor, noise: torch.Tensor, t_norm: torch.Tensor):
    
    if t_norm.ndim == 1:
        t_norm = t_norm.view(-1, 1, 1, 1)
        
    return (1.0 - t_norm) * x0 + t_norm * noise

def _get_rf_target_velocity(x0: torch.Tensor, noise: torch.Tensor):
    return noise - x0

def _calculate_rf_loss_weights(t_norm: torch.Tensor, clamp_val: Optional[float], epsilon: float = 1e-5):
    
    if t_norm.ndim == 1:
        t_norm = t_norm.view(-1, 1, 1, 1)
        
    weights = t_norm + epsilon
    
    if clamp_val is not None:
        weights = torch.clamp(weights, min=0.0, max=clamp_val)
        
    return weights

def _sample_timesteps_logit_normal(bsz: int, num_steps: int, device: torch.device, mu=0.0, sigma=1.0, eps=1e-5):
    
    u = torch.randn(bsz, device=device) * sigma + mu
    t_norm = torch.sigmoid(u).clamp(eps, 1.0 - eps)
    timesteps_int = (t_norm * (num_steps - 1)).long()
    
    return timesteps_int, t_norm


class Trainer:
    def __init__(
        self,
        config: Dict,
        accelerator: Accelerator,
        lora_models: Dict[str, torch.nn.Module],
        vae: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        components_for_inference: Dict,
        max_train_steps: int):
        
        self.config = config
        self.accelerator = accelerator
        self.lora_models = lora_models
        self.vae = vae
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.lr_scheduler = lr_scheduler
        self.components_for_inference = components_for_inference
        self.max_train_steps = max_train_steps
    
    def train(self):
        """Main training loop."""
        
        cfg = self.config
        
        global_step = 0
        progress_bar = tqdm(
            range(self.max_train_steps), 
            disable=not self.accelerator.is_local_main_process, 
            desc="Training Steps"
        )
        
        for epoch in range(cfg.training.num_epochs):
            
            for model in self.lora_models.values():
                model.train()
            
            for step, batch in enumerate(self.dataloader):
                
                with self.accelerator.accumulate(*self.lora_models.values()):
                    
                    with torch.no_grad():
                        pixel_values = batch["pixel_values"].to(dtype=self.vae.dtype)
                        latents = self.vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor
                        latents = latents.to(dtype=list(self.lora_models.values())[0].dtype)

                    pooled_embeds_l = self.lora_models['text_encoder_clip_l'](batch["input_ids_1"], attention_mask=batch["attention_mask_1"])[0]
                    pooled_embeds_g = self.lora_models['text_encoder_clip_g'](batch["input_ids_2"], attention_mask=batch["attention_mask_2"])[0]
                    prompt_embeds_t5 = self.lora_models['text_encoder_t5'](batch["input_ids_3"], attention_mask=batch["attention_mask_3"])[0]

                    encoder_hidden_states = prompt_embeds_t5
                    pooled_projections = torch.cat([pooled_embeds_l, pooled_embeds_g], dim=-1)

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps, t_norm = _sample_timesteps_logit_normal(
                        bsz, 
                        cfg.rectified_flow.num_train_timesteps, 
                        latents.device
                    )
                    
                    noisy_latents = _add_noise_rectified_flow(latents, noise, t_norm)
                    target_velocity = _get_rf_target_velocity(latents, noise)

                    model_pred = self.lora_models['transformer'](
                        hidden_states=noisy_latents, timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        pooled_projections=pooled_projections,
                    )[0]

                    loss_weights = _calculate_rf_loss_weights(t_norm, cfg.rectified_flow.loss_weight_clamp)
                    loss = (torch.nn.functional.mse_loss(model_pred.float(), target_velocity.float(), reduction="none") * loss_weights).mean()
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        
                        self.accelerator.clip_grad_norm_(
                            [p for model in self.lora_models.values() for p in model.parameters() if p.requires_grad],
                            cfg.training.max_grad_norm
                        )
                        
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                if self.accelerator.sync_gradients:
                    
                    progress_bar.update(1)
                    global_step += 1
                    
                    if global_step % cfg.logging.log_steps == 0:
                    
                        loss_value = loss.detach().item()
                        
                        logs = {"loss": loss_value, "epoch": epoch, "step": global_step}
                        
                        lr_groups = {}
                        for i, param_group in enumerate(self.optimizer.param_groups):
                            group_name = f"lr_group_{i}"
                            if i == 0: group_name = "lr_transformer"
                            elif i == 1: group_name = "lr_clip_l"
                            elif i == 2: group_name = "lr_clip_g"
                            elif i == 3: group_name = "lr_t5"
                            lr_groups[group_name] = param_group['lr']
                        
                        print()
                        
                        logs.update(lr_groups)
                        progress_bar.set_postfix(logs)
                        self.accelerator.log(logs, step=global_step)
                    
                    if global_step > 0 and global_step % cfg.logging.save_steps == 0:
                        utils.save_checkpoint(self.lora_models, self.accelerator, cfg.paths.output_dir, global_step)
                        
                        if cfg.validation.run_during_training:
                            self._run_validation_inference(global_step)

                if global_step >= self.max_train_steps:
                    break
            
            if global_step >= self.max_train_steps:
                break

        progress_bar.close()
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            logger.info("Training finished. Saving final LoRA adapters.")
            
            utils.save_checkpoint(self.lora_models, self.accelerator, cfg.paths.output_dir, "final")
            
            if cfg.validation.run_during_training:
                self._run_validation_inference("final")
                
        self.accelerator.end_training()
        utils.cleanup_memory()

    def _run_validation_inference(self, step_name: Union[int, str]):
        
        if not self.accelerator.is_main_process:
            return
        
        logger.info(f"\nRunning validation inference for step: {step_name}...")
        checkpoint_dir = Path(self.config.paths.output_dir) / f"checkpoint-{step_name}"
        

        original_modes = {}
        for name, model in self.lora_models.items():
            original_modes[name] = model.training
            model.eval()

        with torch.no_grad():
            
            run_validation_inference(
                config=self.config,
                accelerator=self.accelerator,
                pipeline_components=self.components_for_inference,
                lora_checkpoint_dir=str(checkpoint_dir),
                output_filename_suffix=f"_step_{step_name}"
            )

        for name, mode in original_modes.items():
            self.lora_models[name].train(mode)

        utils.cleanup_memory()