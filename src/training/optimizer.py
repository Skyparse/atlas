# src/training/optimizer.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.utils.prune as prune
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
        self.config = config

        # Get appropriate device and scaler
        self.device, self.scaler = get_device_and_scaler(
            config.training.mixed_precision
        )
        self.model = self.model.to(self.device)

        # Create optimizer and scheduler
        self.optimizer, self.scheduler = self._create_optimizer()

    def _create_optimizer(self):
        # Filter out parameters that don't require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Create optimizer with weight decay
        optimizer = AdamW(
            params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
        )

        # Create scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.training.learning_rate,
            epochs=self.config.training.num_epochs,
            steps_per_epoch=1000,  # This should be calculated based on dataset size
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
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
