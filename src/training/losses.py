# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAwareLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.base_criterion = nn.CrossEntropyLoss()

    def compute_boundary_map(self, target):
        # Compute gradients
        target_float = target.float().unsqueeze(1)
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=target.device)
            .float()
            .view(1, 1, 3, 3)
        )
        sobel_y = sobel_x.transpose(2, 3)

        grad_x = F.conv2d(target_float, sobel_x, padding=1)
        grad_y = F.conv2d(target_float, sobel_y, padding=1)
        boundary_map = torch.sqrt(grad_x**2 + grad_y**2)

        return (boundary_map > 0).float()

    def forward(self, pred, target):
        # Base cross-entropy loss
        base_loss = self.base_criterion(pred, target)

        # Compute boundary map
        boundary_map = self.compute_boundary_map(target)

        # Compute weighted loss for boundary pixels
        boundary_loss = F.cross_entropy(pred, target, reduction="none")
        boundary_loss = (boundary_loss * boundary_map.squeeze(1)).mean()

        return base_loss + self.weight * boundary_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2)

        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


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
