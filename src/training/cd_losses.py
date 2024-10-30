# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Focal Loss for binary change detection

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter for hard examples
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Target masks (B, C, H, W)
        """
        pred = torch.sigmoid(pred)

        # Calculate binary cross entropy
        bce = F.binary_cross_entropy(pred, target, reduction="none")

        # Calculate focal weights
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma

        # Apply class balancing
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Combine weights and sum
        focal_loss = alpha_weight * focal_weight * bce

        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        """
        Dice Loss for binary change detection

        Args:
            smooth: Smoothing factor to prevent division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Flatten predictions and targets
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Calculate Dice score
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        # Calculate Dice loss
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score

        return dice_loss.mean()


class BinaryChangeDetectionLoss(nn.Module):
    def __init__(self, focal_weight=1.0, dice_weight=1.0, alpha=0.25, gamma=2.0):
        """
        Combined loss for binary change detection

        Args:
            focal_weight: Weight for focal loss
            dice_weight: Weight for dice loss
            alpha: Alpha parameter for focal loss
            gamma: Gamma parameter for focal loss
        """
        super().__init__()
        # Weighted BCE Loss for handling class imbalance
        self.bce_weights = None  # Will be computed dynamically

        self.focal_loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
        )

        # Dice Loss with class weights
        self.dice_loss = DiceLoss(smooth=1e-6)  # Reduced smoothing factor

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def update_class_weights(self, target):
        """Dynamically update class weights based on batch statistics"""
        pos_weight = torch.sum(target == 0) / (torch.sum(target == 1) + 1e-6)
        self.bce_weights = torch.tensor([pos_weight]).to(target.device)

    def forward(self, pred, target):
        # Calculate individual losses
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)

        # Weighted BCE for additional stability
        bce = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=self.bce_weights
        )

        return (
            self.focal_weight * focal + self.dice_weight * dice + 0.2 * bce
        )  # Added BCE component


class ChangeDetectionLoss(nn.Module):
    def __init__(self, config):
        """
        Complete loss function for change detection

        Args:
            config: Configuration object containing loss parameters
        """
        super().__init__()
        self.binary_loss = BinaryChangeDetectionLoss(
            focal_weight=config.loss.focal_weight,
            dice_weight=config.loss.dice_weight,
            alpha=config.loss.focal_alpha,
            gamma=config.loss.focal_gamma,
        )

    def forward(self, pred, target):
        loss = self.binary_loss(pred, target)

        return loss
