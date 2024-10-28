# src/models/components/fpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 1)
                for in_channels in in_channels_list
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
                for _ in in_channels_list
            ]
        )

    def forward(self, features):
        results = []
        last_inner = self.lateral_convs[-1](features[-1])
        results.append(self.fpn_convs[-1](last_inner))

        for feature, lateral_conv, fpn_conv in zip(
            features[:-1][::-1],
            self.lateral_convs[:-1][::-1],
            self.fpn_convs[:-1][::-1],
        ):
            inner_lateral = lateral_conv(feature)
            inner_top_down = F.interpolate(
                last_inner, size=inner_lateral.shape[-2:], mode="nearest"
            )
            last_inner = inner_lateral + inner_top_down
            results.insert(0, fpn_conv(last_inner))

        return results
