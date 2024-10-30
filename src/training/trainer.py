# src/training/trainer.py
import torch
from tqdm import tqdm
from .cd_losses import ChangeDetectionLoss
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
        self.criterion = ChangeDetectionLoss(config)
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
        Compute metrics for a batch of predictions.

        Args:
            outputs: Model predictions (B, C, H, W)
            target: Ground truth masks (B, H, W) or (B, C, H, W)
        """
        with torch.no_grad():
            # Resize predictions to match target size
            outputs = F.interpolate(
                outputs, size=target.shape[2:], mode="bilinear", align_corners=False
            )

            num_classes = outputs.size(1)
            epsilon = 1e-7  # To prevent division by zero

            if num_classes == 1:
                # Binary Segmentation Case
                # Apply sigmoid activation
                pred_probs = torch.sigmoid(outputs)  # Shape: (B, 1, H, W)
                pred_probs = pred_probs.squeeze(1)  # Shape: (B, H, W)

                # Apply threshold to get binary predictions
                threshold = 0.6  # You can adjust this threshold
                preds = (pred_probs > threshold).long()  # Shape: (B, H, W)

                # Ensure target is of shape (B, H, W)
                if target.dim() == 4 and target.size(1) == 1:
                    target = target.squeeze(1)
                elif target.dim() == 4 and target.size(1) > 1:
                    target = target.argmax(dim=1)

                # Compute metrics
                tp = ((preds == 1) & (target == 1)).sum().float()
                tn = ((preds == 0) & (target == 0)).sum().float()
                fp = ((preds == 1) & (target == 0)).sum().float()
                fn = ((preds == 0) & (target == 1)).sum().float()

                accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
                precision = tp / (tp + fp + epsilon)
                recall = tp / (tp + fn + epsilon)
                f1 = 2 * (precision * recall) / (precision + recall + epsilon)
                iou = tp / (tp + fp + fn + epsilon)

                metrics = {
                    "accuracy": accuracy.item(),
                    "precision": precision.item(),
                    "recall": recall.item(),
                    "f1_score": f1.item(),
                    "iou": iou.item(),
                }

            else:
                # Multi-Class Segmentation Case
                # Apply softmax activation
                pred_probs = F.softmax(outputs, dim=1)  # Shape: (B, C, H, W)
                preds = pred_probs.argmax(dim=1)  # Shape: (B, H, W)

                # Ensure target is of shape (B, H, W)
                if target.dim() == 4 and target.size(1) > 1:
                    target = target.argmax(dim=1)
                elif target.dim() == 4 and target.size(1) == 1:
                    target = target.squeeze(1)

                # Compute overall accuracy
                accuracy = (preds == target).float().mean().item()

                # Compute per-class IoU and F1 scores
                ious = []
                f1_scores = []
                for cls in range(num_classes):
                    pred_cls = preds == cls
                    target_cls = target == cls

                    intersection = (pred_cls & target_cls).float().sum()
                    union = (pred_cls | target_cls).float().sum()
                    iou = (intersection + epsilon) / (union + epsilon)
                    ious.append(iou.item())

                    # Compute precision, recall, F1 for each class
                    tp = (pred_cls & target_cls).float().sum()
                    fp = (pred_cls & ~target_cls).float().sum()
                    fn = (~pred_cls & target_cls).float().sum()

                    precision = tp / (tp + fp + epsilon)
                    recall = tp / (tp + fn + epsilon)
                    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
                    f1_scores.append(f1.item())

                mean_iou = sum(ious) / num_classes
                mean_f1 = sum(f1_scores) / num_classes

                metrics = {
                    "accuracy": accuracy,
                    "mean_iou": mean_iou,
                    "mean_f1_score": mean_f1,
                }

                # Optionally, include per-class metrics
                for cls_idx in range(num_classes):
                    metrics[f"iou_class_{cls_idx}"] = ious[cls_idx]
                    metrics[f"f1_class_{cls_idx}"] = f1_scores[cls_idx]

        return metrics
