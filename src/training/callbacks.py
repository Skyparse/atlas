# src/training/callbacks.py
import torch
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import shutil
import logging
from typing import Dict, Any, Optional

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Callback:
    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch, logs):
        pass

    def on_batch_begin(self, trainer, batch):
        pass

    def on_batch_end(self, trainer, batch, logs):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor="val_loss", mode="min", save_best_only=True):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def save_checkpoint(self, trainer, filepath, is_best=False):
        """Save model checkpoint safely"""
        try:
            # Save only model weights
            torch.save(
                trainer.model.state_dict(),
                filepath,
                _use_new_zipfile_serialization=True,  # Use new format
            )
        except Exception as e:
            trainer.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise

    def on_epoch_end(self, trainer, epoch, logs):
        if self.monitor not in logs:
            return

        current = logs[self.monitor]
        filepath = self.filepath / f"model_epoch_{epoch+1}.pt"
        best_path = self.filepath / "best_model.pt"

        if self.mode == "min":
            is_best = current < self.best_value
        else:
            is_best = current > self.best_value

        if is_best or not self.save_best_only:
            if is_best:
                self.best_value = current

            # Save checkpoint
            self.save_checkpoint(trainer, filepath, is_best)

            # Update best model if needed
            if is_best:
                if best_path.exists():
                    best_path.unlink(missing_ok=True)
                shutil.copy2(filepath, best_path)

                # Save metadata separately if needed
                metadata = {
                    "epoch": epoch + 1,
                    "best_value": self.best_value,
                    "monitor": self.monitor,
                    "date": datetime.now().isoformat(),
                }
                metadata_path = best_path.with_suffix(".json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)


class TensorBoardLogger(Callback):
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.writer = None
        if not TENSORBOARD_AVAILABLE:
            logging.warning(
                "TensorBoard not available. Install with 'pip install tensorboard' for enhanced logging."
            )

    def on_train_begin(self, trainer):
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.log_dir)
            # Log initial hyperparameters
            if hasattr(trainer, "config"):
                hparams = {}
                # Convert config to flat dict for tensorboard
                self._add_config_to_hparams(trainer.config, hparams)
                self.writer.add_hparams(hparams, {})

    def _add_config_to_hparams(self, config, hparams, prefix=""):
        """Recursively add config parameters to hparams dict"""
        for key, value in vars(config).items():
            if isinstance(value, (int, float, str, bool)):
                full_key = f"{prefix}_{key}" if prefix else key
                hparams[full_key] = value
            elif hasattr(value, "__dict__"):
                new_prefix = f"{prefix}_{key}" if prefix else key
                self._add_config_to_hparams(value, hparams, new_prefix)

    def on_epoch_end(self, trainer, epoch, logs):
        if self.writer:
            for name, value in logs.items():
                self.writer.add_scalar(name, value, epoch)

            # Log learning rate
            if hasattr(trainer, "optimizer") and hasattr(
                trainer.optimizer, "optimizer"
            ):
                for i, param_group in enumerate(
                    trainer.optimizer.optimizer.param_groups
                ):
                    self.writer.add_scalar(
                        f"learning_rate/group_{i}", param_group["lr"], epoch
                    )

    def on_train_end(self, trainer):
        if self.writer:
            self.writer.close()


class ProgressLogger(Callback):
    def __init__(self, logger):
        self.logger = logger
        self.epoch_start_time = None

    def on_train_begin(self, trainer):
        self.start_time = datetime.now()

    def on_epoch_begin(self, trainer, epoch):
        self.epoch_start_time = datetime.now()
        self.logger.info(
            f"Starting epoch {epoch+1}/{trainer.config.training.num_epochs}"
        )

    def on_epoch_end(self, trainer, epoch, logs):
        epoch_time = datetime.now() - self.epoch_start_time
        log_str = f"Epoch {epoch+1}/{trainer.config.training.num_epochs} "
        log_str += f"[{epoch_time}] - "
        log_str += " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())
        self.logger.info(log_str)

    def on_train_end(self, trainer):
        total_time = datetime.now() - self.start_time
        self.logger.info(f"Training completed in {total_time}")


class MetricsHistory(Callback):
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.history = {}

    def on_train_begin(self, trainer):
        # Initialize history file
        self.save_history()

    def on_epoch_end(self, trainer, epoch, logs):
        # Update history with new metrics
        for metric, value in logs.items():
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(float(value))

        # Save updated history
        self.save_history()

    def save_history(self):
        """Save metrics history to file"""
        try:
            with open(self.filepath / "metrics_history.json", "w") as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save metrics history: {str(e)}")


def create_default_callbacks(exp_dir: Path, logger) -> list:
    """
    Create default set of callbacks for training

    Args:
        exp_dir: Experiment directory path
        logger: Logger instance

    Returns:
        List of callback instances
    """
    return [
        ModelCheckpoint(
            filepath=exp_dir / "checkpoints",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        TensorBoardLogger(exp_dir / "logs"),
        ProgressLogger(logger),
        MetricsHistory(exp_dir / "logs"),
    ]
