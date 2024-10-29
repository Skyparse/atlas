# src/predict.py
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Optional

from src.utils.config import PredictExperimentConfig
from src.models.enhanced_sunet import EnhancedSNUNet
from src.data.dataset import create_prediction_dataset
from src.utils.logger_visuals import setup_logger
from src.data.dataloader import create_prediction_dataloader


def setup_device():
    """Setup compute device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple M-series GPU"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    return device, device_name


def save_predictions(predictions: np.ndarray, path: Path, logger) -> None:
    """Save prediction arrays with proper error handling"""
    try:
        np.save(path, predictions)
        logger.info(f"Saved predictions to {path}")
    except Exception as e:
        logger.error(f"Failed to save predictions to {path}: {str(e)}")
        raise


def process_batch(model, batch, device) -> torch.Tensor:
    """Process a single batch of data"""
    imageA, imageB = [x.to(device) for x in batch]
    with torch.no_grad():
        outputs = model(imageA, imageB)
        if isinstance(outputs, list):  # Handle deep supervision case
            outputs = outputs[-1]  # Take final output
    return outputs


def load_model_weights(model, weights_path: Path, device, logger):
    """
    Load model weights safely from checkpoint file

    Args:
        model: Model instance to load weights into
        weights_path: Path to weights file
        device: Device to load weights to
        logger: Logger instance

    Returns:
        Model with loaded weights
    """
    try:
        logger.info(f"Loading weights from {weights_path}")

        try:
            # Try loading with weights_only first
            checkpoint = torch.load(
                weights_path, map_location=device, weights_only=True
            )

            # Handle both checkpoint dict and direct state dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(
                    f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
                )
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded state dict directly")

        except RuntimeError as e:
            logger.warning(
                "Failed to load with weights_only=True, attempting legacy loading..."
            )
            # Fallback for older checkpoints
            checkpoint = torch.load(weights_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info("Successfully loaded weights using legacy method")

        return model

    except Exception as e:
        logger.error(f"Failed to load model weights: {str(e)}")
        raise


def predict():
    # Initialize basic logger
    logger = setup_logger("prediction", "INFO", Path("logs"))

    try:
        # Load configuration
        logger.info("Loading configuration...")
        with open("configs/predict_config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)

        # Create and validate config
        config = PredictExperimentConfig.from_dict(config_dict)
        config.validate()

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.prediction.output_dir) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup proper logger with file output
        logger = setup_logger("prediction", "INFO", output_dir)

        # Setup device
        device, device_name = setup_device()
        logger.info(f"Using device: {device_name}")

        # Create dataset
        logger.info("Creating prediction dataset...")
        predict_dataset = create_prediction_dataset(
            xA_path=config.data.xA_path,
            xB_path=config.data.xB_path,
        )
        logger.info(f"Created dataset with {len(predict_dataset)} samples")

        # Create dataloader
        predict_loader = create_prediction_dataloader(
            dataset=predict_dataset, config=config
        )

        # Create and setup model
        logger.info("Loading model...")
        model = EnhancedSNUNet(config.model)

        # Load weights
        weights_path = Path(config.prediction.weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # Load model weights safely
        model = load_model_weights(model, weights_path, device, logger)
        model = model.to(device)

        if config.prediction.parallel and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        # Perform prediction
        logger.info("Starting prediction...")
        predictions = []

        try:
            for batch in tqdm(predict_loader, desc="Predicting"):
                outputs = process_batch(model, batch, device)
                predictions.append(outputs.cpu().numpy())

                # Save individual predictions if requested
                if config.prediction.save_individual:
                    for i, pred in enumerate(outputs):
                        save_predictions(
                            pred.cpu().numpy(),
                            output_dir / f"pred_{len(predictions)}_{i}.npy",
                            logger,
                        )

            # Concatenate and save all predictions
            all_predictions = np.concatenate(predictions, axis=0)
            save_predictions(all_predictions, output_dir / "predictions.npy", logger)

            # Generate and save confidence maps
            if config.prediction.save_probabilities:
                logger.info("Generating probability maps...")
                probs = torch.softmax(torch.from_numpy(all_predictions), dim=1).numpy()
                save_predictions(probs, output_dir / "probabilities.npy", logger)

            logger.info("Prediction completed successfully")

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Prediction failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    predict()
