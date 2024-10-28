# src/training/trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm
from ..utils.metrics import calculate_metrics
from .losses import CombinedLoss
from .optimizer import ModelOptimizer


class ModelTrainer:
    def __init__(self, model, config, callbacks, logger):
        self.model = model
        self.config = config
        self.callbacks = callbacks
        self.logger = logger

        self.optimizer = ModelOptimizer(model, config)
        self.criterion = CombinedLoss(
            boundary_weight=config.training.boundary_weight,
            dice_weight=config.training.dice_weight,
        )

    def train(self, train_loader, val_loader):
        self.logger.info("Starting training")
        num_batches = len(train_loader)

        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.config.training.num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.config.training.num_epochs}")

            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)

            # Training phase
            self.model.train()
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validation phase
            self.model.eval()
            val_metrics = self._validate(val_loader)

            # Combine metrics
            metrics = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, metrics)

        for callback in self.callbacks:
            callback.on_train_end(self)

    def _train_epoch(self, train_loader, epoch):
        running_loss = 0.0
        running_metrics = {}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (imageA, imageB, target) in enumerate(pbar):
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)

            if torch.cuda.is_available():
                imageA, imageB = imageA.cuda(), imageB.cuda()
                target = target.cuda()

            # Forward pass
            outputs = self.model(imageA, imageB)
            loss = self.criterion(outputs, target)

            # Optimization step
            self.optimizer.optimize_step(
                loss,
                accumulation=(batch_idx + 1)
                % self.config.training.gradient_accumulation_steps
                != 0,
            )

            # Calculate metrics
            with torch.no_grad():
                batch_metrics = calculate_metrics(outputs, target)

            # Update running metrics
            running_loss += loss.item()
            for k, v in batch_metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + v

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": running_loss / (batch_idx + 1),
                    **{k: v / (batch_idx + 1) for k, v in running_metrics.items()},
                }
            )

            for callback in self.callbacks:
                callback.on_batch_end(
                    self, batch_idx, {"loss": loss.item(), **batch_metrics}
                )

        # Calculate epoch metrics
        num_batches = len(train_loader)
        epoch_metrics = {
            "loss": running_loss / num_batches,
            **{k: v / num_batches for k, v in running_metrics.items()},
        }

        return epoch_metrics

    @torch.no_grad()
    def _validate(self, val_loader):
        running_loss = 0.0
        running_metrics = {}

        for imageA, imageB, target in tqdm(val_loader, desc="Validation"):
            if torch.cuda.is_available():
                imageA, imageB = imageA.cuda(), imageB.cuda()
                target = target.cuda()

            outputs = self.model(imageA, imageB)
            loss = self.criterion(outputs, target)

            batch_metrics = calculate_metrics(outputs, target)

            running_loss += loss.item()
            for k, v in batch_metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + v

        # Calculate validation metrics
        num_batches = len(val_loader)
        val_metrics = {
            "loss": running_loss / num_batches,
            **{k: v / num_batches for k, v in running_metrics.items()},
        }

        return val_metrics
