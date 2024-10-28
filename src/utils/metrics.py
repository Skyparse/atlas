# src/utils/metrics.py
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy.ndimage import distance_transform_edt, binary_erosion as erosion


def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou.item())

    return np.mean(ious)


def calculate_metrics(outputs, targets):
    """Calculate various metrics for change detection."""
    predictions = outputs.argmax(1)

    # Move tensors to CPU and convert to numpy
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets.flatten(), predictions.flatten(), average="binary", zero_division=0
    )

    # Calculate accuracy
    accuracy = (predictions == targets).mean()

    # Calculate IoU
    intersection = np.logical_and(targets, predictions)
    union = np.logical_or(targets, predictions)
    iou = np.sum(intersection) / np.sum(union)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def calculate_confusion_matrix(pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask],
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return hist


def calculate_boundary_metrics(pred, target, boundary_width=2):
    """Calculate metrics specifically for boundary regions."""

    # Convert to numpy arrays
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    # Calculate distance transform for target boundaries
    target_boundaries = target - erosion(target, iterations=1)
    dist_transform = distance_transform_edt(~target_boundaries)
    boundary_region = dist_transform <= boundary_width

    # Calculate metrics in boundary regions
    boundary_pred = pred[boundary_region]
    boundary_target = target[boundary_region]

    precision, recall, f1, _ = precision_recall_fscore_support(
        boundary_target, boundary_pred, average="binary", zero_division=0
    )

    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
    }
