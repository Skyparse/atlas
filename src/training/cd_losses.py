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
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)

        return self.focal_weight * focal + self.dice_weight * dice


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.5):
        """
        Contrastive Loss for change detection features

        Args:
            margin: Margin for contrastive loss
            temperature: Temperature for scaling
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, feat1, feat2, target):
        # Normalize features
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)

        # Calculate distance
        dist = torch.pairwise_distance(feat1, feat2)

        # Calculate loss
        loss = target * dist.pow(2) + (1 - target) * F.relu(self.margin - dist).pow(2)

        return loss.mean()


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
        self.contrastive_loss = ContrastiveLoss(
            margin=config.loss.contrastive_margin,
            temperature=config.loss.contrastive_temperature,
        )
        self.contrastive_weight = config.loss.contrastive_weight

    def forward(self, pred, target, feat1=None, feat2=None):
        loss = self.binary_loss(pred, target)

        if feat1 is not None and feat2 is not None:
            contrastive = self.contrastive_loss(feat1, feat2, target)
            loss = loss + self.contrastive_weight * contrastive

        return loss
