# src/visualize_predictions.py
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import logging
from datetime import datetime

from src.utils.config import VisualizationConfig
from src.utils.logger_visuals import setup_logger


class PredictionVisualizer:
    def __init__(self, config: VisualizationConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load all required data for visualization

        Returns:
            Tuple containing:
            - Time point A images
            - Time point B images
            - Predictions
            - Optional ground truth masks
        """
        self.logger.info("Loading data...")
        try:
            # Load and check imageA
            images_A = np.load(self.config.input.imageA_path)
            self.logger.info(f"Loaded imageA shape: {images_A.shape}")

            # Load and check imageB
            images_B = np.load(self.config.input.imageB_path)
            self.logger.info(f"Loaded imageB shape: {images_B.shape}")

            # Load and check predictions
            predictions = np.load(self.config.input.predictions_path)
            self.logger.info(f"Loaded predictions shape: {predictions.shape}")

            # Validate shapes
            assert (
                len(images_A) == len(images_B) == len(predictions)
            ), "Number of samples must match across all inputs"

            # Load optional masks
            masks = None
            if self.config.input.mask_path:
                masks = np.load(self.config.input.mask_path)
                self.logger.info(f"Loaded masks shape: {masks.shape}")
                assert len(masks) == len(
                    images_A
                ), "Number of masks must match number of images"

            # Extract RGB bands if needed
            images_A = self.extract_rgb_bands(images_A)
            images_B = self.extract_rgb_bands(images_B)

            return images_A, images_B, predictions, masks

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def extract_rgb_bands(self, images: np.ndarray) -> np.ndarray:
        """
        Extract RGB bands from multi-band images

        Args:
            images: Input images array (B, C, H, W)

        Returns:
            RGB images array (B, H, W, 3)
        """
        original_shape = images.shape
        self.logger.info(f"Original image shape: {original_shape}")

        # For (B, C, H, W) format
        if len(images.shape) == 4 and images.shape[1] > 3:
            # Take first three channels and transpose to (B, H, W, 3)
            rgb_images = images[:, :3, :, :].transpose(0, 2, 3, 1)
            self.logger.info(f"Extracted first 3 bands and transposed to (B, H, W, 3)")
        else:
            raise ValueError(f"Unexpected input shape: {original_shape}")

        self.logger.info(f"Final RGB image shape: {rgb_images.shape}")
        return rgb_images

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-1 range

        Args:
            image: Input image (H, W, 3)
        """
        if len(image.shape) != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")

        if image.dtype == np.uint8:
            return image / 255.0

        # Normalize each channel separately
        normalized = np.zeros_like(image, dtype=np.float32)
        for i in range(3):  # RGB channels
            channel = image[..., i]
            min_val = channel.min()
            max_val = channel.max()
            if max_val != min_val:
                normalized[..., i] = (channel - min_val) / (max_val - min_val)

        return normalized

    def process_prediction(self, pred: np.ndarray) -> np.ndarray:
        """
        Process binary prediction

        Args:
            pred: Prediction array (C, H, W) for binary prediction

        Returns:
            Binary prediction map (H, W)
        """
        # Handle binary prediction (1 channel)
        if pred.shape[0] == 1:
            return (pred[0] > 0.5).astype(np.float32)
        else:
            return np.argmax(pred, axis=0)

    def create_colored_mask(self, class_indices: np.ndarray) -> np.ndarray:
        """
        Create RGB colored mask from class indices

        Args:
            class_indices: Array of class indices (H, W)

        Returns:
            Colored mask (H, W, 3)
        """
        H, W = class_indices.shape
        colored_mask = np.zeros((H, W, 3), dtype=np.float32)

        # Create colored mask based on class colors
        for class_idx, color in enumerate(self.config.style.class_colors):
            mask = class_indices == class_idx
            for channel in range(3):
                colored_mask[..., channel] += mask * color[channel]

        return colored_mask

    def create_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create overlay of binary mask on image

        Args:
            image: RGB image array (H, W, 3)
            mask: Binary mask array (H, W)
        """
        if len(image.shape) != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
        if len(mask.shape) != 2:
            raise ValueError(f"Expected mask shape (H, W), got {mask.shape}")

        # Create red overlay for binary mask
        overlay_color = [1.0, 0.0, 0.0]  # Red
        colored_mask = np.zeros_like(image)
        for i in range(3):
            colored_mask[..., i] = mask * overlay_color[i]

        # Create overlay
        overlay = (
            image * (1 - self.config.style.overlay_alpha)
            + colored_mask * self.config.style.overlay_alpha
        )
        return np.clip(overlay, 0, 1)

    def visualize_sample(
        self,
        imageA: np.ndarray,
        imageB: np.ndarray,
        pred: np.ndarray,
        idx: int,
        mask: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ):
        """Visualize a single sample"""
        # Create figure
        fig = plt.figure(figsize=self.config.style.figure_size)

        # Create grid
        if mask is not None:
            gs = plt.GridSpec(3, 3, figure=fig)
            gs.update(hspace=0.3)
        else:
            gs = plt.GridSpec(2, 3, figure=fig)
            gs.update(hspace=0.3)

        # Normalize images
        imageA_norm = self.normalize_image(imageA)
        imageB_norm = self.normalize_image(imageB)

        # Process prediction
        pred_mask = self.process_prediction(pred)

        # First row
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(imageA_norm)
        ax1.set_title("Time Point A")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(imageB_norm)
        ax2.set_title("Time Point B")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(pred_mask, cmap="hot")
        ax3.set_title("Prediction")
        ax3.axis("off")

        # Second row
        overlayA = self.create_overlay(imageA_norm, pred_mask)
        overlayB = self.create_overlay(imageB_norm, pred_mask)

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(overlayA)
        ax4.set_title("Overlay on A")
        ax4.axis("off")

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(overlayB)
        ax5.set_title("Overlay on B")
        ax5.axis("off")

        ax6 = fig.add_subplot(gs[1, 2])
        diff_image = np.abs(imageB_norm - imageA_norm).mean(axis=-1)
        im6 = ax6.imshow(diff_image, cmap="viridis")
        ax6.set_title("Difference Map")
        ax6.axis("off")

        # Add ground truth if available
        if mask is not None:
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = (mask[0] > 0.5).astype(np.float32)

            ax7 = fig.add_subplot(gs[2, :])
            im7 = ax7.imshow(mask, cmap="hot")
            ax7.set_title("Ground Truth")
            ax7.axis("off")

        # Add colorbars
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        plt.colorbar(im3, cax=cbar_ax)

        # Add title
        plt.suptitle(f"Sample {idx+1}", fontsize=16, y=1.05)

        # Save or show
        if save_path:
            plt.savefig(
                save_path,
                dpi=self.config.style.dpi,
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close()
        else:
            plt.show()

    def create_summary(
        self,
        images_A: np.ndarray,
        images_B: np.ndarray,
        predictions: np.ndarray,
        masks: Optional[np.ndarray],
        save_path: Path,
    ):
        """
        Create summary visualization of multiple samples

        Args:
            images_A: First timepoint images (B, H, W, 3)
            images_B: Second timepoint images (B, H, W, 3)
            predictions: Predictions (B, 1, H, W)
            masks: Optional ground truth masks (B, 1, H, W)
            save_path: Path to save summary visualization
        """
        num_samples = len(images_A)
        cols = 4 if masks is not None else 3
        fig = plt.figure(figsize=(5 * cols, 5 * num_samples))

        # Create grid for subplots
        gs = plt.GridSpec(num_samples, cols, figure=fig)
        gs.update(wspace=0.1, hspace=0.3)

        for i in range(num_samples):
            # Normalize images
            imageA = self.normalize_image(images_A[i])
            imageB = self.normalize_image(images_B[i])

            # Process prediction
            pred = predictions[i]
            if pred.shape[0] == 1:  # Binary prediction
                pred_mask = (pred[0] > 0.5).astype(np.float32)
            else:
                pred_mask = pred.argmax(axis=0)

            # Create overlay
            overlay = self.create_overlay(imageB, pred_mask)

            # Create subplots
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(imageA)
            ax1.set_title(f"Sample {i+1}\nTime Point A")
            ax1.axis("off")

            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(imageB)
            ax2.set_title("Time Point B")
            ax2.axis("off")

            ax3 = fig.add_subplot(gs[i, 2])
            ax3.imshow(pred_mask, cmap="hot")
            ax3.set_title("Prediction")
            ax3.axis("off")

            if masks is not None:
                # Process ground truth mask
                mask = masks[i]
                if mask.shape[0] == 1:  # Binary mask
                    gt_mask = (mask[0] > 0.5).astype(np.float32)
                else:
                    gt_mask = mask.argmax(axis=0)

                ax4 = fig.add_subplot(gs[i, 3])
                ax4.imshow(gt_mask, cmap="hot")
                ax4.set_title("Ground Truth")
                ax4.axis("off")

        # Add color bar for predictions
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, 1))
        plt.colorbar(sm, cax=cbar_ax)

        # Save figure
        plt.savefig(
            save_path, dpi=self.config.style.dpi, bbox_inches="tight", pad_inches=0.1
        )
        plt.close()


def main():
    # Setup logger
    logger = setup_logger("visualization", "INFO", Path("logs"))

    try:
        # Load configuration
        logger.info("Loading configuration...")
        with open("configs/visualize_config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)

        config = VisualizationConfig.from_dict(config_dict)
        config.validate()

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.output.save_dir) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize visualizer
        visualizer = PredictionVisualizer(config, logger)

        # Load data
        images_A, images_B, predictions, masks = visualizer.load_data()

        # Create visualizations
        logger.info("Creating visualizations...")
        num_samples = min(config.output.num_samples, len(images_A))

        for idx in range(num_samples):
            if config.output.save_individual:
                save_path = output_dir / f"sample_{idx}.{config.output.save_format}"
                visualizer.visualize_sample(
                    images_A[idx],
                    images_B[idx],
                    predictions[idx],
                    idx,
                    masks[idx] if masks is not None else None,
                    save_path,
                )

        # Create summary visualization if requested
        if config.output.save_summary:
            logger.info("Creating summary visualization...")
            visualizer.create_summary(
                images_A[:num_samples],
                images_B[:num_samples],
                predictions[:num_samples],
                masks[:num_samples] if masks is not None else None,
                output_dir / f"summary.{config.output.save_format}",
            )

        logger.info(f"Visualizations saved to {output_dir}")

    except Exception as e:
        logger.error(f"Visualization failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
