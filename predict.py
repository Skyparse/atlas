# src/predict.py
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils.config import ModelConfig
from src.models.enhanced_sunet import EnhancedSNUNet
from src.data.dataset import create_prediction_dataset
from src.data.transforms import Compose, Normalize
from src.utils.logger_visuals import setup_logger
from src.data.dataloader import create_dataloaders


def predict():
    # Load configuration
    with open("configs/predict_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    # Extract paths and create output directory
    predict_config = config_dict["prediction"]
    data_config = config_dict["data"]

    # Validate input paths
    for path_name in ["xA_path", "xB_path"]:
        if not Path(data_config[path_name]).exists():
            raise FileNotFoundError(f"Data file not found: {data_config[path_name]}")

    if not Path(predict_config["weights_path"]).exists():
        raise FileNotFoundError(
            f"Weights file not found: {predict_config['weights_path']}"
        )

    output_dir = Path(predict_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger("prediction", "INFO", output_dir)
    logger.info(f"Starting prediction using {predict_config['weights_path']}")

    # Create prediction dataset
    predict_dataset = create_prediction_dataset(
        xA_path=data_config["xA_path"],
        xB_path=data_config["xB_path"],
    )

    logger.info(f"Created prediction dataset with {len(predict_dataset)} samples")

    # Create dataloader
    predict_loader = create_dataloaders(predict_dataset, predict_config)

    # Create and setup model
    model_config = ModelConfig.from_dict(config_dict["model"])
    model = EnhancedSNUNet(model_config)

    # Load weights
    model.load_state_dict(torch.load(predict_config["weights_path"]))

    if torch.cuda.is_available():
        model = model.cuda()
        if predict_config.get("parallel", False) and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    model.eval()

    # Perform prediction
    predictions = []
    try:
        with torch.no_grad():
            for batch_idx, (imageA, imageB) in enumerate(tqdm(predict_loader)):
                if torch.cuda.is_available():
                    imageA, imageB = imageA.cuda(), imageB.cuda()

                # Forward pass
                outputs = model(imageA, imageB)
                predictions.append(outputs.cpu().numpy())

                # Save batch predictions
                if predict_config.get("save_individual", False):
                    for i, pred in enumerate(outputs):
                        pred_path = output_dir / f"prediction_{batch_idx}_{i}.npy"
                        np.save(pred_path, pred.cpu().numpy())

        # Concatenate all predictions
        predictions = np.concatenate(predictions, axis=0)

        # Save full predictions
        output_path = output_dir / "predictions.npy"
        np.save(output_path, predictions)
        logger.info(f"Saved predictions to {output_path}")

        # Calculate and save confidence maps if requested
        if predict_config.get("save_confidence", False):
            confidence_maps = torch.softmax(
                torch.from_numpy(predictions), dim=1
            ).numpy()
            confidence_path = output_dir / "confidence_maps.npy"
            np.save(confidence_path, confidence_maps)
            logger.info(f"Saved confidence maps to {confidence_path}")

    except Exception as e:
        logger.error(f"Prediction failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    predict()
