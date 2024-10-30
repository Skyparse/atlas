import torch
import torch.nn as nn


class ConvBlockNested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(ConvBlockNested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)) + x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):
        return self.up(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(in_channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        return x * avg_out


class SNUNet_ECAM(nn.Module):
    def __init__(
        self, in_channels, num_classes, base_channel=32, depth=5, bilinear=False
    ):
        super(SNUNet_ECAM, self).__init__()
        self.depth = depth
        n1 = base_channel
        self.filters = [n1 * (2**i) for i in range(depth)]

        self.in_channels = in_channels
        self.bilinear = bilinear

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (downsampling)
        self.conv_blocks_down = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                in_ch = self.in_channels
            else:
                in_ch = self.filters[i - 1]
            out_ch = self.filters[i]
            self.conv_blocks_down.append(ConvBlockNested(in_ch, out_ch, out_ch))

        # Decoder (upsampling)
        self.up_convs = nn.ModuleList()
        self.conv_blocks_up = nn.ModuleList()
        for i in range(depth - 1):
            up_in_ch = self.filters[i + 1]
            up_out_ch = self.filters[i]
            self.up_convs.append(Up(up_in_ch, up_out_ch, bilinear))

            in_ch = (
                self.filters[i] * 2
            )  # concatenated channels from encoder and decoder
            out_ch = self.filters[i]
            self.conv_blocks_up.append(ConvBlockNested(in_ch, out_ch, out_ch))

        # Final attention and classification layers
        self.ca = ChannelAttention(self.filters[0], ratio=8)
        self.conv_final = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)

    def forward(self, xA, xB):
        # Encoder path
        encodersA = []
        encodersB = []
        xA_enc = xA
        xB_enc = xB
        for idx, down in enumerate(self.conv_blocks_down):
            xA_enc = down(xA_enc)
            xB_enc = down(xB_enc)
            if idx != self.depth - 1:
                encodersA.append(xA_enc)
                encodersB.append(xB_enc)
                xA_enc = self.pool(xA_enc)
                xB_enc = self.pool(xB_enc)

        # Decoder path
        x = xA_enc + xB_enc  # Combine deepest features
        for idx in reversed(range(self.depth - 1)):
            x = self.up_convs[idx](x)
            x = torch.cat([x, encodersA[idx] + encodersB[idx]], dim=1)
            x = self.conv_blocks_up[idx](x)

        # Attention and final classification
        x = self.ca(x)
        out = self.conv_final(x)
        return out
