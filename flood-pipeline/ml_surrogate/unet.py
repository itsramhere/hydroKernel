"""
2D U-Net Neural Network Architecture Specification for Flood Inundation (Phase 3).

Surrogate Model Interface:
- Input: 3 Channels (Normalized DEM, Slope Gradient, 6-hr Precipitation Depth).
- Output: 1 Channel (Continuous Predicted Water Surface Depth in meters).
- Invariant: Non-negative standing water depth h >= 0.0 m enforced via terminal ReLU.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """[Conv2d 3x3 -> BatchNorm2d -> LeakyReLU(0.2)] * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """DoubleConv followed by MaxPool2d(2)"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        pooled = self.pool(feat)
        return feat, pooled


class UpBlock(nn.Module):
    """ConvTranspose2d stride 2 + skip connection concatenation + double Conv2d"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dimensions have odd parity
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FloodUNet(nn.Module):
    """
    Hydrological Flood Surrogate 2D U-Net Model.
    - 4-level downsampling encoder: 3 -> 32 -> 64 -> 128 -> 256
    - Bottleneck: 256 -> 512
    - 4-level upsampling decoder: 512 -> 256 -> 128 -> 64 -> 32
    - Head: Conv2d(32, 1, kernel_size=1) -> terminal ReLU()
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 4-level Downsampling Encoder
        self.down1 = DownBlock(in_channels, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        self.down4 = DownBlock(128, 256)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # 4-level Upsampling Decoder
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)

        # Output Head: Conv2d(32, 1, kernel_size=1) -> ReLU()
        self.head = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass
        s1, p1 = self.down1(input_tensor)
        s2, p2 = self.down2(p1)
        s3, p3 = self.down3(p2)
        s4, p4 = self.down4(p3)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder forward pass with skip connections
        d1 = self.up1(b, s4)
        d2 = self.up2(d1, s3)
        d3 = self.up3(d2, s2)
        d4 = self.up4(d3, s1)

        # Terminal ReLU ensures non-negative depth h >= 0.0m
        water_depth = self.head(d4)
        return water_depth


class UNetArchitectureSpec:
    """Defines layer dimensions, channel mappings, and physical parameterizations."""

    INPUT_CHANNELS = 3    # Ch0: DEM, Ch1: Slope, Ch2: Precip (mm)
    OUTPUT_CHANNELS = 1   # Ch0: Inundation depth h (meters)

    # Physical scaling parameters
    MIN_DAMAGE_DEPTH_M = 0.15   # 15 cm crop/property damage threshold
    SURCHARGE_FACTOR = 0.004    # mm rain to ponded water depth conversion

    @classmethod
    def get_input_shape(cls, height: int = 256, width: int = 256) -> Tuple[int, int, int, int]:
        return (1, cls.INPUT_CHANNELS, height, width)
