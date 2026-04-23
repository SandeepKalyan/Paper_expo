from __future__ import annotations

import torch

from src.models.rsf_conv import count_parameters
from src.models.unet import UNet
from src.models.unet_rsf import RSFUNet


def main() -> None:
    x = torch.randn(2, 3, 256, 256)
    unet = UNet(base_channels=32)
    rsf_unet = RSFUNet(base_channels=24)
    y1 = unet(x)
    y2 = rsf_unet(x)
    assert y1.shape == (2, 1, 256, 256), y1.shape
    assert y2.shape == (2, 1, 256, 256), y2.shape
    p1 = count_parameters(unet)
    p2 = count_parameters(rsf_unet)
    print({"unet_params": p1, "rsf_unet_params": p2, "ratio": round(p2 / p1, 4)})


if __name__ == "__main__":
    main()
