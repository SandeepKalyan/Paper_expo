from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from src.data.sampler import crop, random_patch_coords, vessel_centered_patch_coords
from src.data.transforms import (
    augment_train,
    binarize,
    normalize_image,
    pad_to_multiple,
    to_tensor_sample,
)


@dataclass
class DatasetPaths:
    root: Path
    images: Path
    masks: Path
    fov_masks: Path
    splits: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "DatasetPaths":
        r = Path(root)
        return cls(
            root=r,
            images=r / "images",
            masks=r / "masks",
            fov_masks=r / "fov_masks",
            splits=r / "splits",
        )


def _read_ids(splits_dir: Path, split: str) -> list[str]:
    path = splits_dir / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file missing: {path}")
    ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [i for i in ids if i]


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not readable: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask not readable: {path}")
    return m


class RetinalDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        train: bool = True,
        patch_size: tuple[int, int] = (256, 256),
        samples_per_epoch: int | None = None,
        vessel_sampling_prob: float = 0.0,
        pad_multiple: int = 32,
        seed: int = 0,
        use_paper_augs: bool = True,
    ) -> None:
        super().__init__()
        self.paths = DatasetPaths.from_root(root)
        self.split = split
        self.train = train
        self.patch_size = tuple(patch_size)
        self.vessel_sampling_prob = float(vessel_sampling_prob)
        self.pad_multiple = int(pad_multiple)
        self.use_paper_augs = bool(use_paper_augs)
        self.ids = _read_ids(self.paths.splits, split)
        if not self.ids:
            raise RuntimeError(f"Empty split '{split}' at {self.paths.splits}")
        self.samples_per_epoch = (
            int(samples_per_epoch) if (train and samples_per_epoch) else len(self.ids)
        )
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _image_path(self, sid: str) -> Path:
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            p = self.paths.images / f"{sid}{ext}"
            if p.exists():
                return p
        return self.paths.images / f"{sid}.png"

    def _load_sample(self, sid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = _load_image(self._image_path(sid))
        mask = binarize(_load_mask(self.paths.masks / f"{sid}.png"))
        fov = binarize(_load_mask(self.paths.fov_masks / f"{sid}.png"))
        return img, mask, fov

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.train:
            sid = self.ids[int(self._rng.integers(0, len(self.ids)))]
        else:
            sid = self.ids[idx % len(self.ids)]
        img, mask, fov = self._load_sample(sid)

        if self.train:
            img, mask, fov = augment_train(img, mask, fov, self._rng, use_paper_augs=self.use_paper_augs)
            ph, pw = self.patch_size
            coords = None
            if self.vessel_sampling_prob > 0.0 and self._rng.random() < self.vessel_sampling_prob:
                coords = vessel_centered_patch_coords(mask, ph, pw, self._rng)
            if coords is None:
                coords = random_patch_coords(img.shape[0], img.shape[1], ph, pw, self._rng)
            y, x = coords
            img, mask, fov = crop(img, mask, fov, y, x, ph, pw)
        else:
            img, mask, fov = pad_to_multiple(img, mask, fov, multiple=self.pad_multiple)

        img = normalize_image(img)
        sample = to_tensor_sample(img, mask, fov)
        sample["id"] = sid
        return sample


class DriveDataset(RetinalDataset):
    pass


class StareDataset(RetinalDataset):
    pass


class ChaseDB1Dataset(RetinalDataset):
    pass


class HRFDataset(RetinalDataset):
    pass
