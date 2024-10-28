# src/models/components/fpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        Feature Pyramid Network implementation

        Args:
            in_channels_list: List of input channels for each level (from bottom to top)
            out_channels: Number of output channels for each FPN level
        """
        super().__init__()

        # Create lateral connections (1x1 convs)
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                for in_channels in in_channels_list
            ]
        )

        # Create smooth layers (3x3 convs)
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(len(in_channels_list))
            ]
        )

    def forward(self, features):
        """
        Args:
            features: List of feature maps [P1, P2, P3, P4, P5] from bottom to top level
        Returns:
            List of FPN features from fine to coarse levels
        """
        # Process from top to bottom
        last_feature = self.lateral_convs[-1](features[-1])
        results = [self.fpn_convs[-1](last_feature)]

        for feature, lateral_conv, fpn_conv, prev_result in zip(
            reversed(features[:-1]),  # Previous features
            reversed(self.lateral_convs[:-1]),  # Lateral convs
            reversed(self.fpn_convs[:-1]),  # FPN convs
            results,  # Previous FPN results
        ):
            # 1x1 conv on current feature
            lateral_feature = lateral_conv(feature)

            # Upsample previous result
            top_down_feature = F.interpolate(
                prev_result, size=lateral_feature.shape[-2:], mode="nearest"
            )

            # Add and smooth
            fpn_feature = fpn_conv(lateral_feature + top_down_feature)
            results.insert(0, fpn_feature)

        return results
