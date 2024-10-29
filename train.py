# src/train.py
import torch
import yaml
from pathlib import Path
from datetime import datetime
import logging
import sys

from src.utils.config import TrainExperimentConfig
from src.models.enhanced_sunet import EnhancedSNUNet
from src.training.trainer import ModelTrainer
from src.training.callbacks import (
    ModelCheckpoint,
    TensorBoardLogger,
    ProgressLogger,
    MetricsHistory,
)
from src.data.dataset import create_train_val_datasets
from src.utils.logger_visuals import setup_logger
from src.data.dataloader import create_dataloaders


def setup_experiment_dir(config):
    """Setup experiment directory and logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = (
        Path(config.output.output_dir) / f"{config.output.experiment_name}_{timestamp}"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"
    checkpoints_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    return exp_dir, logs_dir, checkpoints_dir


def train():
    # Initialize basic logger first
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        # Load configuration
        logger.info("Loading configuration...")
        with open("configs/train_config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)

        # Create and validate config
        config = TrainExperimentConfig.from_dict(config_dict)

        # Setup experiment directory
        exp_dir, logs_dir, checkpoints_dir = setup_experiment_dir(config)

        # Setup proper logger with file output
        logger = setup_logger("training", "INFO", logs_dir)
        logger.info(f"Starting experiment in {exp_dir}")

        # Validate configuration
        config.validate()

        # Save configuration
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f)

        # Set random seeds
        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.training.seed)

        # Create datasets
        logger.info("Creating datasets...")
        train_dataset, val_dataset = create_train_val_datasets(config)

        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, config
        )

        # Setup device
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info(f"Using device: {device}")

        # Create model
        logger.info("Creating model...")
        model = EnhancedSNUNet(config.model)
        model = model.to(device)

        # Setup callbacks
        callbacks = [
            ModelCheckpoint(filepath=checkpoints_dir, monitor="val_loss", mode="min"),
            TensorBoardLogger(logs_dir),
            ProgressLogger(logger),
            MetricsHistory(logs_dir),
        ]

        # Create trainer
        trainer = ModelTrainer(
            model=model, config=config, callbacks=callbacks, logger=logger
        )

        # Train model
        logger.info("Starting training...")
        trainer.train(train_loader, val_loader)
        logger.info("Training completed successfully")

    except Exception as e:
        if logger:
            logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        # Save final model
        if "model" in locals() and "checkpoints_dir" in locals():
            final_model_path = checkpoints_dir / "final_model.pt"
            torch.save(model.state_dict(), final_model_path)
            if logger:
                logger.info(f"Saved final model to {final_model_path}")


if __name__ == "__main__":
    train()
