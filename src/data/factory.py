from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from src.data.dataset import (
    ChaseDB1Dataset,
    DriveDataset,
    HRFDataset,
    RetinalDataset,
    StareDataset,
)


_DATASET_REGISTRY = {
    "drive": DriveDataset,
    "stare": StareDataset,
    "chase": ChaseDB1Dataset,
    "chase_db1": ChaseDB1Dataset,
    "chasedb1": ChaseDB1Dataset,
    "hrf": HRFDataset,
}


def build_dataset(cfg: dict[str, Any], split: str, train: bool) -> RetinalDataset:
    name = str(cfg["name"]).lower()
    cls = _DATASET_REGISTRY.get(name, RetinalDataset)
    patch_size = tuple(cfg.get("patch_size", (256, 256)))
    samples_per_epoch = cfg.get("samples_per_epoch")
    vessel_prob = float(cfg.get("vessel_sampling_prob", 0.0))
    pad_multiple = int(cfg.get("pad_multiple", 32))
    seed = int(cfg.get("seed", 0))
    use_paper_augs = bool(cfg.get("use_paper_augs", True))
    return cls(
        root=cfg["root"],
        split=split,
        train=train,
        patch_size=patch_size,
        samples_per_epoch=samples_per_epoch,
        vessel_sampling_prob=vessel_prob,
        pad_multiple=pad_multiple,
        seed=seed,
        use_paper_augs=use_paper_augs,
    )


def build_dataloader(cfg: dict[str, Any], split: str, train: bool) -> DataLoader:
    dataset = build_dataset(cfg, split=split, train=train)
    batch_size = int(cfg.get("batch_size", 2)) if train else int(cfg.get("eval_batch_size", 1))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
