# src/training/optimizer.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ..utils.device_utils import get_device_and_scaler


class ModelOptimizer:
    def __init__(self, model, config):
        """
        Initialize ModelOptimizer

        Args:
            model: The model to optimize
            config: TrainExperimentConfig containing all configuration
        """
        self.model = model
        params = self._get_layer_specific_lrs(model)
        self.config = config

        # Get appropriate device and scaler
        self.device, self.scaler = get_device_and_scaler(
            config.training.mixed_precision
        )
        self.model = self.model.to(self.device)

        # Create optimizer and scheduler
        self.optimizer, self.scheduler = self._create_optimizer(params=params)

    def _get_layer_specific_lrs(self, model):
        """Configure different learning rates for different parts of the model"""
        encoder_params = []
        decoder_params = []

        for name, param in model.named_parameters():
            if "conv_blocks_down" in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        return [
            {"params": encoder_params, "lr": 1e-5},  # Lower LR for encoder
            {"params": decoder_params, "lr": 5e-5},  # Higher LR for decoder
        ]

    def _create_optimizer(self, params):
        # Create optimizer with weight decay
        optimizer = AdamW(
            params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Create scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,  # Initial restart interval
            T_mult=2,  # Multiply interval by 2 after each restart
            eta_min=1e-7,  # Minimum learning rate
        )

        return optimizer, scheduler

    def optimize_step(self, loss, accumulation=False):
        if accumulation:
            loss = loss / self.config.training.gradient_accumulation_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if not accumulation:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.max_grad_norm
                )
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        if self.scaler and state_dict["scaler"]:
            self.scaler.load_state_dict(state_dict["scaler"])
