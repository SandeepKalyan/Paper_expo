from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def predict_patches_overlap(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: int = 256,
    stride: int = 128,
    tta: bool = False,
) -> torch.Tensor:
    """Sliding-window prediction with overlapping patches.

    image: [B, C, H, W] normalized
    returns prob map [B, 1, H, W]
    """
    b, c, h, w = image.shape
    ph = pw = patch_size
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        H, W = image.shape[-2:]
    else:
        H, W = h, w

    if H > ph:
        ys = list(range(0, H - ph, stride)) + [H - ph]
    else:
        ys = [0]
    if W > pw:
        xs = list(range(0, W - pw, stride)) + [W - pw]
    else:
        xs = [0]

    accum = torch.zeros(b, 1, H, W, device=image.device)
    count = torch.zeros(b, 1, H, W, device=image.device)

    for y in ys:
        for x in xs:
            crop = image[:, :, y : y + ph, x : x + pw]
            logits = model(crop)
            probs = torch.sigmoid(logits)
            if tta:
                probs_h = torch.flip(torch.sigmoid(model(torch.flip(crop, dims=[3]))), dims=[3])
                probs_v = torch.flip(torch.sigmoid(model(torch.flip(crop, dims=[2]))), dims=[2])
                probs = (probs + probs_h + probs_v) / 3.0
            accum[:, :, y : y + ph, x : x + pw] += probs
            count[:, :, y : y + ph, x : x + pw] += 1.0

    prob = accum / count.clamp_min(1.0)
    return prob[:, :, :h, :w]
