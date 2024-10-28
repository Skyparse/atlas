# src/data/dataset.py
import torch
from torch.utils.data import Dataset, random_split
import numpy as np
from pathlib import Path


class ChangeDetectionDataset(Dataset):
    def __init__(self, xA_path, xB_path, mask_path=None, transform=None):
        """
        Args:
            xA_path: path to the first timepoint images array
            xB_path: path to the second timepoint images array
            mask_path: path to the mask labels array (optional for prediction)
            transform: optional transform to be applied on images
        """
        self.xA = np.load(xA_path)
        self.xB = np.load(xB_path)
        self.masks = np.load(mask_path) if mask_path else None
        self.transform = transform

        assert len(self.xA) == len(self.xB), "Image pairs must have same length"
        if self.masks is not None:
            assert len(self.xA) == len(
                self.masks
            ), "Images and masks must have same length"

    def __len__(self):
        return len(self.xA)

    def __getitem__(self, idx):
        # Get images
        imageA = torch.from_numpy(self.xA[idx]).float()
        imageB = torch.from_numpy(self.xB[idx]).float()

        # Apply transforms if any
        if self.transform:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)

        # Return with or without mask depending on if it's for training
        if self.masks is not None:
            mask = torch.from_numpy(self.masks[idx]).long()
            return imageA, imageB, mask
        return imageA, imageB


def create_train_val_datasets(
    xA_path, xB_path, mask_path, val_split=0.2, transform=None
):
    """
    Create training and validation datasets from numpy arrays.

    Args:
        xA_path: path to first timepoint images array
        xB_path: path to second timepoint images array
        mask_path: path to mask labels array
        val_split: fraction of data to use for validation
        transform: optional transform to apply to images
    """
    # Create full dataset
    full_dataset = ChangeDetectionDataset(xA_path, xB_path, mask_path, transform)

    # Calculate split sizes
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Split dataset
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    return train_dataset, val_dataset


def create_prediction_dataset(xA_path, xB_path, transform=None):
    """Create dataset for prediction only (no masks required)."""
    return ChangeDetectionDataset(xA_path, xB_path, mask_path=None, transform=transform)
