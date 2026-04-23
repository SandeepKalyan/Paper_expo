from __future__ import annotations

from typing import Any

from src.models.unet import UNet
from src.models.unet_rsf import RSFUNet


def build_model(cfg: dict[str, Any]):
    name = cfg["name"].lower()
    in_channels = cfg.get("in_channels", 3)
    out_channels = cfg.get("out_channels", 1)
    base_channels = cfg.get("base_channels", 32)
    if name == "unet":
        return UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if name in {"rsf_unet", "unet_rsf", "rsf+unet"}:
        return RSFUNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    raise ValueError(f"Unsupported model: {cfg['name']}")
