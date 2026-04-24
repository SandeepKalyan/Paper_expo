from __future__ import annotations

import numpy as np


def random_patch_coords(
    h: int,
    w: int,
    ph: int,
    pw: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if h <= ph:
        y = 0
    else:
        y = int(rng.integers(0, h - ph + 1))
    if w <= pw:
        x = 0
    else:
        x = int(rng.integers(0, w - pw + 1))
    return y, x


def vessel_centered_patch_coords(
    mask: np.ndarray,
    ph: int,
    pw: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0:
        return None
    i = int(rng.integers(0, ys.size))
    cy, cx = int(ys[i]), int(xs[i])
    h, w = mask.shape[:2]
    y = max(0, min(cy - ph // 2, h - ph)) if h > ph else 0
    x = max(0, min(cx - pw // 2, w - pw)) if w > pw else 0
    return y, x


def crop(
    image: np.ndarray,
    mask: np.ndarray,
    fov: np.ndarray,
    y: int,
    x: int,
    ph: int,
    pw: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    if h < ph or w < pw:
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        fov = np.pad(fov, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return (
        image[y : y + ph, x : x + pw],
        mask[y : y + ph, x : x + pw],
        fov[y : y + ph, x : x + pw],
    )
