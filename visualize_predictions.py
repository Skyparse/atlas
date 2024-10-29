# src/visualize_predictions.py
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Tuple, Optional
from src.utils.config import PredictExperimentConfig
from src.utils.logger_visuals import setup_logger


def load_data(config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load original images and predictions

    Args:
        config: Configuration containing file paths

    Returns:
        Tuple of (imageA, imageB, predictions)
    """
    # Load original images
    imageA = np.load(config.data.xA_path)
    imageB = np.load(config.data.xB_path)

    # Load predictions
    pred_path = Path(config.prediction.output_dir) / "predictions.npy"
    predictions = np.load(pred_path)

    return imageA, imageB, predictions


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range"""
    if image.dtype == np.uint8:
        return image / 255.0

    min_val = image.min()
    max_val = image.max()
    if max_val == min_val:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[float, float, float] = (1, 0, 0),
) -> np.ndarray:
    """
    Create an overlay of mask on image

    Args:
        image: Base image (H,W,C) or (H,W)
        mask: Binary mask (H,W)
        alpha: Transparency of overlay
        color: RGB color for overlay

    Returns:
        Overlay image
    """
    # Ensure image is RGB
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 1:
        image = np.concatenate([image] * 3, axis=-1)

    # Create colored mask
    colored_mask = np.zeros_like(image)
    for i in range(3):
        colored_mask[..., i] = mask * color[i]

    # Create overlay
    overlay = (
        image * (1 - alpha * mask[..., None]) + colored_mask * alpha * mask[..., None]
    )
    return np.clip(overlay, 0, 1)


def visualize_prediction(
    imageA: np.ndarray,
    imageB: np.ndarray,
    pred: np.ndarray,
    idx: int,
    save_path: Optional[Path] = None,
):
    """
    Visualize a single prediction

    Args:
        imageA: First timepoint image
        imageB: Second timepoint image
        pred: Prediction mask
        idx: Sample index
        save_path: Optional path to save visualization
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f"Sample {idx}", fontsize=16)

    # Normalize images
    imageA_norm = normalize_image(imageA)
    imageB_norm = normalize_image(imageB)

    # Get prediction mask
    if pred.shape[0] == 1:  # Binary case
        pred_mask = (pred[0] > 0.5).astype(np.float32)
    else:  # Multi-class case
        pred_mask = pred.argmax(0).astype(np.float32)

    # Create overlays
    overlayA = create_overlay(imageA_norm, pred_mask)
    overlayB = create_overlay(imageB_norm, pred_mask)

    # Plot images
    axes[0, 0].imshow(imageA_norm)
    axes[0, 0].set_title("Image A")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(imageB_norm)
    axes[0, 1].set_title("Image B")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pred_mask, cmap="hot")
    axes[1, 0].set_title("Prediction Mask")
    axes[1, 0].axis("off")

    diff_image = np.abs(imageB_norm - imageA_norm).mean(axis=-1)
    axes[1, 1].imshow(diff_image, cmap="viridis")
    axes[1, 1].set_title("Difference Map")
    axes[1, 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_batch(
    images_A: np.ndarray,
    images_B: np.ndarray,
    predictions: np.ndarray,
    indices: list,
    output_dir: Path,
):
    """
    Visualize a batch of predictions

    Args:
        images_A: First timepoint images
        images_B: Second timepoint images
        predictions: Prediction masks
        indices: List of indices to visualize
        output_dir: Directory to save visualizations
    """
    for idx in indices:
        save_path = output_dir / f"visualization_{idx}.png"
        visualize_prediction(
            images_A[idx], images_B[idx], predictions[idx], idx, save_path
        )


def main():
    # Setup logger
    logger = setup_logger("visualization", "INFO", Path("logs"))

    try:
        # Load configuration
        logger.info("Loading configuration...")
        with open("configs/predict_config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)

        config = PredictExperimentConfig.from_dict(config_dict)

        # Create output directory
        output_dir = Path(config.prediction.output_dir) / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        logger.info("Loading data...")
        images_A, images_B, predictions = load_data(config)

        # Create visualizations
        logger.info("Creating visualizations...")
        num_samples = len(images_A)

        # Visualize first 10 samples or all if less
        indices = list(range(min(10, num_samples)))
        visualize_batch(images_A, images_B, predictions, indices, output_dir)

        logger.info(f"Saved visualizations to {output_dir}")

        # Optional: create a summary figure
        create_summary = True
        if create_summary:
            fig, axes = plt.subplots(len(indices), 4, figsize=(20, 5 * len(indices)))

            for i, idx in enumerate(indices):
                # Normalize images
                imageA_norm = normalize_image(images_A[idx])
                imageB_norm = normalize_image(images_B[idx])
                pred_mask = (
                    predictions[idx].argmax(0)
                    if predictions[idx].shape[0] > 1
                    else predictions[idx][0] > 0.5
                )

                # Plot
                axes[i, 0].imshow(imageA_norm)
                axes[i, 0].set_title(f"Sample {idx} - Image A")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(imageB_norm)
                axes[i, 1].set_title("Image B")
                axes[i, 1].axis("off")

                axes[i, 2].imshow(pred_mask, cmap="hot")
                axes[i, 2].set_title("Prediction")
                axes[i, 2].axis("off")

                overlay = create_overlay(imageB_norm, pred_mask)
                axes[i, 3].imshow(overlay)
                axes[i, 3].set_title("Overlay")
                axes[i, 3].axis("off")

            plt.tight_layout()
            plt.savefig(output_dir / "summary.png")
            plt.close()

    except Exception as e:
        logger.error(f"Visualization failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
