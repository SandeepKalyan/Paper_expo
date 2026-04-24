from __future__ import annotations

from typing import Any

from src.models.unet import UNet
from src.models.unet_rsf import RSFUNet
from src.models.rsf_paper import RSFConvUnet


def build_model(cfg: dict[str, Any]):
    name = cfg["name"].lower()
    in_channels = cfg.get("in_channels", 3)
    out_channels = cfg.get("out_channels", 1)
    base_channels = cfg.get("base_channels", 32)

    if name == "unet":
        return UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if name in {"rsf_unet", "unet_rsf", "rsf+unet"}:
        return RSFUNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if name in {"rsf_paper", "rsfconv_unet", "rsf-conv+unet"}:
        return RSFConvUnet(
            n_channels=in_channels,
            n_classes=out_channels,
            rotNum=int(cfg.get("rotNum", 8)),
            initS=float(cfg.get("initS", 3)),
            gapS=float(cfg.get("gapS", 1.25)),
            numS=int(cfg.get("numS", 4)),
            return_logits=True,
        )
    raise ValueError(f"Unsupported model: {cfg['name']}")
