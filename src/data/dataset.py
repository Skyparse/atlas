# src/data/dataset.py
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging


class ChangeDetectionDataset(Dataset):
    """Dataset for change detection with efficient data handling"""

    def __init__(
        self,
        xA: np.ndarray,
        xB: np.ndarray,
        masks: Optional[np.ndarray] = None,
        transform=None,
        cache_size: int = 1000,
    ):
        """
        Initialize the dataset

        Args:
            xA: First timepoint images (B,C,H,W)
            xB: Second timepoint images (B,C,H,W)
            masks: Optional mask labels (B,C,H,W)
            transform: Optional transforms to apply
            cache_size: Number of items to cache in memory
        """
        self.xA = torch.from_numpy(xA).float()
        self.xB = torch.from_numpy(xB).float()
        self.masks = torch.from_numpy(masks).float() if masks is not None else None
        self.transform = transform

        # Initialize cache
        self.cache = {}
        self.cache_size = cache_size

        # Validate shapes
        assert len(self.xA) == len(self.xB), "Image pairs must have same length"
        if self.masks is not None:
            assert len(self.xA) == len(
                self.masks
            ), "Images and masks must have same length"
            assert len(self.masks.shape) == 4, "Masks must be 4D (B,C,H,W)"

    def __len__(self) -> int:
        return len(self.xA)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]

        # Get images
        imageA = self.xA[idx]
        imageB = self.xB[idx]

        # Apply transforms if any
        if self.transform:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)

        # Prepare output
        if self.masks is not None:
            output = (imageA, imageB, self.masks[idx])
        else:
            output = (imageA, imageB)

        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = output

        return output


def create_train_val_datasets(config) -> Tuple[Dataset, Dataset]:
    """
    Create training and validation datasets

    Args:
        config: Configuration object containing data paths and settings

    Returns:
        train_dataset, val_dataset: Training and validation datasets
    """
    logger = logging.getLogger(__name__)

    # Load data
    logger.info("Loading datasets...")
    try:
        xA = np.load(config.data.xA_path)
        xB = np.load(config.data.xB_path)
        masks = np.load(config.data.mask_path)

        logger.info(
            f"Loaded data shapes - xA: {xA.shape}, xB: {xB.shape}, masks: {masks.shape}"
        )
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

    # Create indices for split
    indices = np.arange(len(xA))
    train_indices, val_indices = train_test_split(
        indices, test_size=config.data.val_split, random_state=config.training.seed
    )

    # Create the full dataset
    full_dataset = ChangeDetectionDataset(
        xA=xA,
        xB=xB,
        masks=masks,
        cache_size=1000,  # Adjust based on your memory constraints
    )

    # Create train and validation datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    logger.info(
        f"Created datasets - Training: {len(train_dataset)}, Validation: {len(val_dataset)}"
    )

    return train_dataset, val_dataset


def create_prediction_dataset(xA_path: str, xB_path: str) -> Dataset:
    """
    Create dataset for prediction (no masks)

    Args:
        xA_path: Path to first timepoint images
        xB_path: Path to second timepoint images

    Returns:
        Dataset for prediction
    """
    xA = np.load(xA_path)
    xB = np.load(xB_path)

    return ChangeDetectionDataset(xA=xA, xB=xB, masks=None, cache_size=1000)
