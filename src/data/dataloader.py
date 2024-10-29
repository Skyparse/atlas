# src/data/dataloader.py
import torch
from torch.utils.data import DataLoader
import logging
from typing import Tuple


class PrefetchDataLoader:
    """Custom DataLoader with prefetching for faster data loading"""

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    ):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def create_dataloaders(
    train_dataset, val_dataset, config, num_workers: int = None, pin_memory: bool = None
) -> Tuple[PrefetchDataLoader, PrefetchDataLoader]:
    """
    Create optimized dataloaders for training and validation

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        num_workers: Optional override for number of workers
        pin_memory: Optional override for pin_memory

    Returns:
        train_loader, val_loader: Tuple of train and validation dataloaders
    """
    logger = logging.getLogger(__name__)

    # Use config values or defaults
    num_workers = num_workers if num_workers is not None else 4
    pin_memory = pin_memory if pin_memory is not None else True

    # Adjust num_workers based on system
    if num_workers > 0 and not hasattr(torch, "get_num_threads"):
        logger.warning(
            "PyTorch was not built with parallel support. Setting num_workers to 0"
        )
        num_workers = 0

    # Create train loader
    train_loader = PrefetchDataLoader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
    )

    # Create validation loader
    val_loader = PrefetchDataLoader(
        dataset=val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
    )

    logger.info(
        f"Created dataloaders with {num_workers} workers and pin_memory={pin_memory}"
    )
    logger.info(
        f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}"
    )

    return train_loader, val_loader


def create_prediction_dataloader(
    dataset, config, num_workers: int = None, pin_memory: bool = None
) -> PrefetchDataLoader:
    """
    Create optimized dataloader for prediction

    Args:
        dataset: Prediction dataset
        config: Prediction configuration
        num_workers: Optional override for number of workers
        pin_memory: Optional override for pin_memory

    Returns:
        PrefetchDataLoader for prediction
    """
    logger = logging.getLogger(__name__)

    # Use config values or defaults
    num_workers = num_workers if num_workers is not None else 4
    pin_memory = pin_memory if pin_memory is not None else True

    # Adjust num_workers based on system
    if num_workers > 0 and not hasattr(torch, "get_num_threads"):
        logger.warning(
            "PyTorch was not built with parallel support. Setting num_workers to 0"
        )
        num_workers = 0

    # Create prediction loader
    predict_loader = PrefetchDataLoader(
        dataset=dataset,
        batch_size=config.prediction.batch_size,
        shuffle=False,  # No shuffling needed for prediction
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
    )

    logger.info(
        f"Created prediction dataloader with {num_workers} workers and pin_memory={pin_memory}"
    )
    logger.info(f"Prediction batches: {len(predict_loader)}")

    return predict_loader
