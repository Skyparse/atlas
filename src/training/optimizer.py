# src/training/optimizer.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.utils.prune as prune


class OptimizerFactory:
    @staticmethod
    def create_optimizer(model, config):
        # Filter out parameters that don't require gradients
        params = [p for p in model.parameters() if p.requires_grad]

        # Create optimizer with weight decay
        optimizer = AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )

        # Create scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=config.steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        return optimizer, scheduler


class ModelOptimizer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer, self.scheduler = OptimizerFactory.create_optimizer(
            model, config
        )
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        if config.pruning:
            self.setup_pruning()
        if config.quantization:
            self.setup_quantization()

    def setup_pruning(self):
        self.pruned_modules = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(
                    module, "weight", amount=self.config.pruning_ratio
                )
                self.pruned_modules.append(module)

    def setup_quantization(self):
        self.model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        self.model = torch.quantization.prepare(self.model)

    def optimize_step(self, loss, accumulation=False):
        if accumulation:
            loss = loss / self.config.gradient_accumulation_steps

        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if not accumulation:
            if self.config.mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

            if self.config.pruning:
                for module in self.pruned_modules:
                    prune.remove(module, "weight")

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
