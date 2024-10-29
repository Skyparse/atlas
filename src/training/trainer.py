# src/training/trainer.py
import torch
from tqdm import tqdm
from .losses import CombinedLoss
from .optimizer import ModelOptimizer
from ..utils.device_utils import move_to_device
from torch.nn import functional as F


class ModelTrainer:
    def __init__(self, model, config, callbacks, logger):
        self.model = model
        self.config = config
        self.callbacks = callbacks
        self.logger = logger

        self.optimizer = ModelOptimizer(model, config)
        self.criterion = CombinedLoss()
        self.device = self.optimizer.device

        self.logger.info(f"Using device: {self.device}")

    def train(self, train_loader, val_loader):

        # Training start callback
        for callback in self.callbacks:
            callback.on_train_begin(self)

        try:
            # Epoch loop
            for epoch in range(self.config.training.num_epochs):
                self.logger.info(f"Epoch {epoch+1}/{self.config.training.num_epochs}")

                # Epoch start callback
                for callback in self.callbacks:
                    callback.on_epoch_begin(self, epoch)

                # Training phase
                self.model.train()
                train_metrics = self._train_epoch(train_loader, epoch)

                # Log training metrics
                self.logger.info(
                    f"Epoch {epoch+1} - Train: "
                    + " - ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
                )

                # Validation phase if it's evaluation epoch
                should_evaluate = (
                    epoch + 1 == self.config.training.num_epochs  # Last epoch
                    or (
                        self.config.training.eval_frequency > 0  # Regular evaluation
                        and (epoch + 1) % self.config.training.eval_frequency == 0
                    )
                )

                metrics = {"train_" + k: v for k, v in train_metrics.items()}

                if should_evaluate:
                    self.logger.info(f"Performing evaluation at epoch {epoch+1}")
                    self.model.eval()
                    val_metrics = self._validate(val_loader)
                    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                    # Log validation metrics
                    self.logger.info(
                        f"Epoch {epoch+1} - Validation: "
                        + " - ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                    )

                # Epoch end callback
                for callback in self.callbacks:
                    callback.on_epoch_end(self, epoch, metrics)

        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise
        finally:
            # Training end callback
            for callback in self.callbacks:
                callback.on_train_end(self)

    def _train_epoch(self, train_loader, epoch):
        running_loss = 0.0
        running_metrics = {}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (imageA, imageB, target) in enumerate(pbar):
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)

            # Move data to device
            imageA, imageB, target = move_to_device(
                (imageA, imageB, target), self.device
            )

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

            # For metrics, use the final output if deep supervision is enabled
            if isinstance(outputs, list):
                outputs = outputs[-1]  # Use the final prediction for metrics

            # Update metrics
            running_loss += loss.item()
            batch_metrics = self._compute_metrics(outputs, target)
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
            # Move data to device
            imageA, imageB, target = move_to_device(
                (imageA, imageB, target), self.device
            )

            # Forward pass
            outputs = self.model(imageA, imageB)
            loss = self.criterion(outputs, target)

            # For metrics, use the final output if deep supervision is enabled
            if isinstance(outputs, list):
                outputs = outputs[-1]  # Use the final prediction for metrics

            # Update metrics
            running_loss += loss.item()
            batch_metrics = self._compute_metrics(outputs, target)
            for k, v in batch_metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + v

        num_batches = len(val_loader)
        val_metrics = {
            "loss": running_loss / num_batches,
            **{k: v / num_batches for k, v in running_metrics.items()},
        }

        return val_metrics

    def _compute_metrics(self, outputs, target):
        """
        Compute metrics for a batch of predictions

        Args:
            outputs: Model predictions (B, C, H, W)
            target: One-hot encoded ground truth (B, C, H, W)
        """
        with torch.no_grad():
            # Resize predictions to match target size
            outputs = F.interpolate(
                outputs, size=target.shape[2:], mode="bilinear", align_corners=False
            )

            # Convert predictions to class probabilities
            pred_softmax = F.softmax(outputs, dim=1)

            # Get class indices
            preds = outputs.argmax(dim=1)
            target_indices = target.argmax(dim=1)

            # Calculate accuracy
            accuracy = (preds == target_indices).float().mean().item()

            # Calculate IoU for each class
            num_classes = outputs.size(1)
            ious = []
            f1_scores = []

            for cls in range(num_classes):
                # IoU calculation
                pred_mask = preds == cls
                target_mask = target_indices == cls
                intersection = (pred_mask & target_mask).float().sum()
                union = (pred_mask | target_mask).float().sum()
                iou = (intersection + 1e-6) / (union + 1e-6)
                ious.append(iou.item())

                # F1 score calculation
                pred_cls = (pred_softmax[:, cls] > 0.5).float()
                target_cls = target[:, cls].float()

                tp = (pred_cls * target_cls).sum()
                fp = (pred_cls * (1 - target_cls)).sum()
                fn = ((1 - pred_cls) * target_cls).sum()

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)
                f1_scores.append(f1.item())

            mean_iou = sum(ious) / len(ious)
            mean_f1 = sum(f1_scores) / len(f1_scores)

            return {
                "accuracy": accuracy,
                "iou": mean_iou,
                "f1": mean_f1,
                **{f"iou_class_{i}": iou for i, iou in enumerate(ious)},
                **{f"f1_class_{i}": f1 for i, f1 in enumerate(f1_scores)},
            }
