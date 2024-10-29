# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAwareLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.base_criterion = nn.BCEWithLogitsLoss()

    def compute_boundary_map(self, target):
        """Compute boundary map from target mask"""
        target_float = target.float()

        # Compute gradients
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=target.device)
            .float()
            .view(1, 1, 3, 3)
        )
        sobel_y = sobel_x.transpose(2, 3)

        grad_x = F.conv2d(target_float.unsqueeze(1), sobel_x, padding=1)
        grad_y = F.conv2d(target_float.unsqueeze(1), sobel_y, padding=1)
        boundary_map = torch.sqrt(grad_x**2 + grad_y**2)

        return (boundary_map > 0).float()

    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions (B, C, H, W)
            target: Target masks (B, H, W) or (B, C, H, W)
        """
        if pred is None or target is None:
            raise ValueError("Predictions or targets are None")

        # Ensure target has same size as predictions
        if pred.shape[2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=True
            )

        # Handle one-hot encoded target
        if len(target.shape) == 4:  # (B, C, H, W)
            target = target.argmax(dim=1)  # Convert to class indices

        # Convert target to float for BCE loss
        target_float = target.float()

        # Compute base loss
        base_loss = self.base_criterion(pred.squeeze(1), target_float)

        # Compute boundary-aware component
        boundary_map = self.compute_boundary_map(target_float)
        boundary_loss = F.binary_cross_entropy_with_logits(
            pred.squeeze(1), target_float, reduction="none"
        )
        weighted_boundary_loss = (boundary_loss * boundary_map.squeeze(1)).mean()

        return base_loss + self.weight * weighted_boundary_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions (B, C, H, W)
            target: Target masks (B, H, W) or (B, C, H, W)
        """
        if pred is None or target is None:
            raise ValueError("Predictions or targets are None")

        # Ensure target has same size as predictions
        if pred.shape[2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=True
            )

        # Handle one-hot encoded target
        if len(target.shape) == 4:  # (B, C, H, W)
            target = target.argmax(dim=1)  # Convert to class indices

        # Apply sigmoid since we're using BCE-based loss
        pred = torch.sigmoid(pred.squeeze(1))
        target = target.float()

        # Compute Dice score
        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, boundary_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.boundary_loss = BoundaryAwareLoss(weight=boundary_weight)
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions (B, C, H, W)
            target: Target masks (B, H, W) or (B, C, H, W)
        """
        if isinstance(pred, list):  # Handle deep supervision
            total_loss = 0
            for p in pred:
                total_loss += self.boundary_loss(
                    p, target
                ) + self.dice_weight * self.dice_loss(p, target)
            return total_loss / len(pred)

        return self.boundary_loss(pred, target) + self.dice_weight * self.dice_loss(
            pred, target
        )
