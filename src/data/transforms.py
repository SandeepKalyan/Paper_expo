from __future__ import annotations

import cv2
import numpy as np
import torch


def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def normalize_image(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def binarize(mask: np.ndarray) -> np.ndarray:
    return (mask > 127).astype(np.float32)


def _affine_warp(
    image: np.ndarray,
    mask: np.ndarray,
    fov: np.ndarray,
    angle_deg: float,
    scale: float,
    shear_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    if abs(shear_deg) > 1e-6:
        sh = np.tan(np.deg2rad(shear_deg))
        shear = np.array([[1.0, sh, -sh * cy], [0.0, 1.0, 0.0]], dtype=np.float32)
        rot3 = np.vstack([rot, [0.0, 0.0, 1.0]])
        shear3 = np.vstack([shear, [0.0, 0.0, 1.0]])
        full = (shear3 @ rot3)[:2]
    else:
        full = rot
    image = cv2.warpAffine(
        image, full, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    mask_u = (mask * 255.0).astype(np.uint8)
    fov_u = (fov * 255.0).astype(np.uint8)
    mask_u = cv2.warpAffine(mask_u, full, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    fov_u = cv2.warpAffine(fov_u, full, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return image, (mask_u > 127).astype(np.float32), (fov_u > 127).astype(np.float32)


def _color_jitter(
    image: np.ndarray,
    rng: np.random.Generator,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
) -> np.ndarray:
    img = image.astype(np.float32)
    if brightness > 0:
        img = img * float(1.0 + rng.uniform(-brightness, brightness))
    if contrast > 0:
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * float(1.0 + rng.uniform(-contrast, contrast)) + mean
    if saturation > 0:
        gray = img.mean(axis=2, keepdims=True)
        img = (img - gray) * float(1.0 + rng.uniform(-saturation, saturation)) + gray
    return _clip_uint8(img)


def augment_train(
    image: np.ndarray,
    mask: np.ndarray,
    fov: np.ndarray,
    rng: np.random.Generator,
    use_paper_augs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
        fov = np.ascontiguousarray(fov[:, ::-1])
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])
        fov = np.ascontiguousarray(fov[::-1, :])

    if use_paper_augs:
        angle = float(rng.uniform(0.0, 360.0))
        scale = float(rng.uniform(0.8, 1.4))
        shear = float(rng.uniform(-10.0, 10.0))
        image, mask, fov = _affine_warp(image, mask, fov, angle, scale, shear)
        image = _color_jitter(image, rng, brightness=0.2, contrast=0.2, saturation=0.2)
    else:
        k = int(rng.integers(0, 4))
        if k:
            image = np.ascontiguousarray(np.rot90(image, k=k, axes=(0, 1)))
            mask = np.ascontiguousarray(np.rot90(mask, k=k, axes=(0, 1)))
            fov = np.ascontiguousarray(np.rot90(fov, k=k, axes=(0, 1)))
    return image, mask, fov


def pad_to_multiple(
    image: np.ndarray,
    mask: np.ndarray,
    fov: np.ndarray,
    multiple: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return image, mask, fov
    image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant", constant_values=0)
    fov = np.pad(fov, ((0, ph), (0, pw)), mode="constant", constant_values=0)
    return image, mask, fov


def to_tensor_sample(
    image: np.ndarray,
    mask: np.ndarray,
    fov: np.ndarray,
) -> dict[str, torch.Tensor]:
    img_t = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
    mask_t = torch.from_numpy(mask[None, :, :].copy()).float()
    fov_t = torch.from_numpy(fov[None, :, :].copy()).float()
    return {"image": img_t, "mask": mask_t, "fov_mask": fov_t}
