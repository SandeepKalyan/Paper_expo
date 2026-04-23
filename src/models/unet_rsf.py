from __future__ import annotations

import torch
import torch.nn as nn

from src.models.rsf_conv import RSFConvBlock


class RSFUpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = RSFConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RSFUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.enc1 = RSFConvBlock(in_channels, c)
        self.enc2 = RSFConvBlock(c, c * 2)
        self.enc3 = RSFConvBlock(c * 2, c * 4)
        self.enc4 = RSFConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = RSFConvBlock(c * 8, c * 16)
        self.dec4 = RSFUpBlock(c * 16, c * 8, c * 8)
        self.dec3 = RSFUpBlock(c * 8, c * 4, c * 4)
        self.dec2 = RSFUpBlock(c * 4, c * 2, c * 2)
        self.dec1 = RSFUpBlock(c * 2, c, c)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)
