# src/train.py
import torch
import yaml
from pathlib import Path
from datetime import datetime

from src.utils.config import ModelConfig
from src.models.enhanced_sunet import EnhancedSNUNet
from src.training.trainer import ModelTrainer
from src.training.callbacks import (
    ModelCheckpoint,
    TensorBoardLogger,
    ProgressLogger,
    MetricsHistory,
)
from src.data.dataset import create_train_val_datasets
from src.data.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from src.data.dataloader import create_dataloaders
from src.utils.logger_visuals import setup_logger


def train():
    # Load configuration
    with open("configs/train_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    # Extract paths and create experiment directory
    data_config = config_dict["data"]
    output_config = config_dict["output"]

    # Validate data paths
    for path_name in ["xA_path", "xB_path", "mask_path"]:
        if not Path(data_config[path_name]).exists():
            raise FileNotFoundError(f"Data file not found: {data_config[path_name]}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = output_config.get("experiment_name", "experiment")
    exp_dir = Path(output_config["output_dir"]) / f"{exp_name}_{timestamp}"
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
        xA_path=data_config["xA_path"],
        xB_path=data_config["xB_path"],
        mask_path=data_config["mask_path"],
        val_split=data_config["val_split"],
    )

    logger.info(
        f"Created datasets - Training: {len(train_dataset)}, Validation: {len(val_dataset)}"
    )

    # Create dataloaders
    train_loader = create_dataloaders(train_dataset, config_dict["training"])
    val_loader = create_dataloaders(val_dataset, config_dict["training"])

    # Create and setup model
    model_config = ModelConfig.from_dict(config_dict["model"])
    model_config.validate()

    model = EnhancedSNUNet(model_config)
    if torch.cuda.is_available():
        model = model.cuda()
        if (
            config_dict["training"].get("parallel", False)
            and torch.cuda.device_count() > 1
        ):
            model = torch.nn.DataParallel(model)

    # Load pretrained weights if specified
    if "pretrained_weights" in config_dict["model"]:
        pretrained_path = Path(config_dict["model"]["pretrained_weights"])
        if pretrained_path.exists():
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path))

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
        model=model, config=model_config, callbacks=callbacks, logger=logger
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
