# src/utils/device_utils.py
import torch
from typing import Tuple
import logging


def get_device_and_scaler(
    mixed_precision: bool = True,
) -> Tuple[torch.device, torch.amp.GradScaler]:
    """
    Get appropriate device and scaler based on available hardware.

    Args:
        mixed_precision: Whether to use mixed precision training

    Returns:
        device: torch.device to use
        scaler: GradScaler instance or None
    """
    # Check available devices in order of preference
    if torch.cuda.is_available():
        device = torch.device("cuda")
        scaler = torch.amp.GradScaler() if mixed_precision else None
        logging.info("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # MPS doesn't support mixed precision yet
        scaler = None
        if mixed_precision:
            logging.warning("Mixed precision not supported on MPS, disabled")
    else:
        device = torch.device("cpu")
        scaler = None
        if mixed_precision:
            logging.warning("Mixed precision not supported on CPU, disabled")

    return device, scaler


def move_to_device(data, device: torch.device):
    """
    Recursively move data to device.

    Args:
        data: Data to move (can be tensor, list, tuple, or dict)
        device: Device to move data to

    Returns:
        Data moved to device
    """
    if isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data
