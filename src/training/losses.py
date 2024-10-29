# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAwareLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def compute_boundary_map(self, target):
        """Compute boundary map from one-hot encoded target mask"""
        target_indices = target.argmax(dim=1, keepdim=True).float()

        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=target.device)
            .float()
            .view(1, 1, 3, 3)
        )
        sobel_y = sobel_x.transpose(2, 3)

        grad_x = F.conv2d(target_indices, sobel_x, padding=1)
        grad_y = F.conv2d(target_indices, sobel_y, padding=1)
        boundary_map = torch.sqrt(grad_x**2 + grad_y**2)

        return (boundary_map > 0).float()

    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions (B, C, H, W) or list of predictions
            target: One-hot encoded ground truth (B, C, H, W)
        """
        target_indices = target.argmax(dim=1)  # Convert to class indices

        if isinstance(pred, list):
            losses = []
            for p in pred:
                # Resize predictions to match target size
                p = F.interpolate(
                    p, size=target.shape[2:], mode="bilinear", align_corners=False
                )
                loss = F.cross_entropy(p, target_indices, reduction="none")
                boundary_map = self.compute_boundary_map(target)
                weighted_loss = (loss * boundary_map.squeeze(1)).mean()
                losses.append(weighted_loss)
            return sum(losses) / len(losses)

        # Resize single prediction to match target size
        pred = F.interpolate(
            pred, size=target.shape[2:], mode="bilinear", align_corners=False
        )
        loss = F.cross_entropy(pred, target_indices, reduction="none")
        boundary_map = self.compute_boundary_map(target)
        weighted_loss = (loss * boundary_map.squeeze(1)).mean()

        return weighted_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def _compute_dice_loss(self, pred, target):
        """
        Args:
            pred: Model predictions (B, C, H, W)
            target: One-hot encoded ground truth (B, C, H, W)
        """
        # Resize predictions to match target size
        pred = F.interpolate(
            pred, size=target.shape[2:], mode="bilinear", align_corners=False
        )
        pred = F.softmax(pred, dim=1)

        intersection = (pred * target.float()).sum(dim=(2, 3))
        cardinality = pred.sum(dim=(2, 3)) + target.float().sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1 - dice.mean()

    def forward(self, pred, target):
        if isinstance(pred, list):
            return sum(self._compute_dice_loss(p, target) for p in pred) / len(pred)
        return self._compute_dice_loss(pred, target)


class CombinedLoss(nn.Module):
    def __init__(self, boundary_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.boundary_loss = BoundaryAwareLoss(weight=boundary_weight)
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return self.boundary_loss(pred, target) + self.dice_weight * self.dice_loss(
            pred, target
        )
