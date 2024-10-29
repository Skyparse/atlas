# src/models/components/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()

        # Ensure reduction doesn't make channels zero
        self.reduced_channels = max(in_channels // reduction_ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Use proper channel dimensions
        self.fc1 = nn.Conv2d(
            in_channels, self.reduced_channels, kernel_size=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            self.reduced_channels, in_channels, kernel_size=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        # Average branch
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))

        # Max branch
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))

        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat_out = torch.cat([avg_out, max_out], dim=1)

        out = self.sigmoid(self.conv(cat_out))
        return x * out


class CBAM(nn.Module):
    def __init__(
        self, channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7
    ):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
