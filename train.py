# src/train.py
import torch
import yaml
from pathlib import Path
from datetime import datetime

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
from src.data.dataloader import create_dataloaders
from src.utils.logger_visuals import setup_logger


def train():
    # Load configuration
    with open("configs/train_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    # Create and validate config
    config = TrainExperimentConfig.from_dict(config_dict)
    config.validate()

    # Setup experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = (
        Path(config.output.output_dir) / f"{config.output.experiment_name}_{timestamp}"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup directories
    checkpoints_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"
    checkpoints_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Save configuration
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config_dict, f)

    # Setup logger
    logger = setup_logger("training", "INFO", logs_dir)
    logger.info(f"Starting experiment in {exp_dir}")

    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        xA_path=config.data.xA_path,
        xB_path=config.data.xB_path,
        mask_path=config.data.mask_path,
        val_split=config.data.val_split,
    )

    # Create dataloaders
    train_loader = create_dataloaders(
        train_dataset, vars(config.training)  # Convert training config to dict
    )
    val_loader = create_dataloaders(
        val_dataset, vars(config.training)  # Convert training config to dict
    )

    # Create model
    model = EnhancedSNUNet(config.model)
    if torch.cuda.is_available():
        model = model.cuda()
        if config.training.parallel and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=exp_dir / "checkpoints", monitor="val_loss", mode="min"
        ),
        TensorBoardLogger(exp_dir / "logs"),
        ProgressLogger(logger),
        MetricsHistory(exp_dir / "logs"),
    ]

    # Create trainer and start training
    trainer = ModelTrainer(
        model=model, config=config, callbacks=callbacks, logger=logger
    )

    try:
        trainer.train(train_loader, val_loader)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        # Save final model
        torch.save(model.state_dict(), exp_dir / "checkpoints" / "final_model.pt")


if __name__ == "__main__":
    train()
