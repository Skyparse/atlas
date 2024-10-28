# src/training/callbacks.py
import torch
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter


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

    def on_epoch_end(self, trainer, epoch, logs):
        current = logs.get(self.monitor)
        if current is None:
            return

        filepath = self.filepath / f"model_epoch_{epoch}.pt"
        is_best = (self.mode == "min" and current < self.best_value) or (
            self.mode == "max" and current > self.best_value
        )

        if is_best or not self.save_best_only:
            self.best_value = current if is_best else self.best_value
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "logs": logs,
                    "best_value": self.best_value,
                },
                filepath,
            )

            if is_best:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": trainer.model.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "logs": logs,
                        "best_value": self.best_value,
                    },
                    self.filepath / "best_model.pt",
                )


class TensorBoardLogger(Callback):
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.writer = None

    def on_train_begin(self, trainer):
        self.writer = SummaryWriter(self.log_dir)

    def on_epoch_end(self, trainer, epoch, logs):
        for name, value in logs.items():
            self.writer.add_scalar(name, value, epoch)

    def on_train_end(self, trainer):
        if self.writer:
            self.writer.close()


class ProgressLogger(Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_epoch_begin(self, trainer, epoch):
        self.logger.info(f"Starting epoch {epoch}")

    def on_epoch_end(self, trainer, epoch, logs):
        log_str = f"Epoch {epoch} - "
        log_str += " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())
        self.logger.info(log_str)


class MetricsHistory(Callback):
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.history = {}

    def on_epoch_end(self, trainer, epoch, logs):
        for metric, value in logs.items():
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(float(value))

        with open(self.filepath / "metrics_history.json", "w") as f:
            json.dump(self.history, f, indent=4)
