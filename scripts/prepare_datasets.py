"""Normalize raw retinal datasets into unified layout.

Reads from data/raw/ (deepdyn tarball + HRF zip already extracted),
writes to data/{DRIVE,STARE,CHASE_DB1,HRF}/{images,masks,fov_masks,splits}.

Unified layout per dataset:
    images/{sid}.png         uint8 RGB
    masks/{sid}.png          uint8 binary (0/255)
    fov_masks/{sid}.png      uint8 binary (0/255)
    splits/train.txt         newline sids
    splits/test.txt
    ids_map.json             original -> sid
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEEPDYN = RAW / "deepdyn-master" / "data"
HRF_RAW = RAW / "HRF"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clear_dataset_dir(ds_root: Path) -> None:
    for sub in ("images", "masks", "fov_masks", "splits"):
        d = ds_root / sub
        if d.exists():
            shutil.rmtree(d)
        _ensure_dir(d)


def _load_any(path: Path) -> np.ndarray:
    """Load gif/tif/jpg/ppm/pgm as RGB or grayscale numpy array (uint8)."""
    if path.suffix.lower() in {".gif", ".ppm", ".pgm"}:
        # cv2 handles ppm/pgm; PIL handles gif best.
        try:
            img = np.array(Image.open(path).convert("RGB" if _is_rgb(path) else "L"))
            return img
        except Exception:
            pass
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(f"Cannot read {path}")
    if data.ndim == 3 and data.shape[2] >= 3:
        data = cv2.cvtColor(data[..., :3], cv2.COLOR_BGR2RGB)
    return data


def _is_rgb(path: Path) -> bool:
    # Heuristic: mask/fov files typically grayscale. Let caller force mode if needed.
    return True


def _save_png(path: Path, arr: np.ndarray) -> None:
    if arr.ndim == 2:
        cv2.imwrite(str(path), arr)
    else:
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _binarize(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return (arr > 127).astype(np.uint8) * 255


def _generate_fov(rgb: np.ndarray, thresh: int = 10) -> np.ndarray:
    """FOV mask from luminance + largest-connected-component + fill holes."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fov = (gray > thresh).astype(np.uint8)
    # Morphological close to seal seams.
    k = max(3, min(gray.shape) // 50)
    k = k if k % 2 == 1 else k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, kernel)
    # Largest CC.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fov, connectivity=8)
    if num <= 1:
        return fov * 255
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fov = (labels == best).astype(np.uint8)
    # Fill internal holes.
    inv = 1 - fov
    num_h, labels_h, _, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num_h > 1:
        h, w = fov.shape
        border = set()
        border.update(labels_h[0, :].tolist())
        border.update(labels_h[-1, :].tolist())
        border.update(labels_h[:, 0].tolist())
        border.update(labels_h[:, -1].tolist())
        for lbl in range(1, num_h):
            if lbl not in border:
                fov[labels_h == lbl] = 1
    return fov * 255


def _write_splits(ds_root: Path, train_ids: list[str], test_ids: list[str]) -> None:
    (ds_root / "splits" / "train.txt").write_text("\n".join(train_ids) + "\n")
    (ds_root / "splits" / "test.txt").write_text("\n".join(test_ids) + "\n")


def _sid(i: int) -> str:
    return f"img_{i:03d}"


def prepare_drive() -> dict:
    src = DEEPDYN / "DRIVE"
    dst = ROOT / "data" / "DRIVE"
    _clear_dataset_dir(dst)
    mapping = {}
    train_ids, test_ids = [], []
    idx = 0
    # DRIVE canonical split: 21-40 train, 01-20 test.
    names = sorted(p.name for p in (src / "images").iterdir())
    for name in names:
        m = re.match(r"(\d+)_(test|training)\.tif", name)
        if not m:
            continue
        num = int(m.group(1))
        split = m.group(2)
        sid = _sid(idx)
        idx += 1
        mapping[sid] = name
        img = _load_any(src / "images" / name)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        _save_png(dst / "images" / f"{sid}.png", img)
        mask = _load_any(src / "manual" / f"{num:02d}_manual1.gif")
        _save_png(dst / "masks" / f"{sid}.png", _binarize(mask))
        fov = _load_any(src / "mask" / f"{num:02d}_mask.gif")
        _save_png(dst / "fov_masks" / f"{sid}.png", _binarize(fov))
        (test_ids if split == "test" else train_ids).append(sid)
    _write_splits(dst, train_ids, test_ids)
    (dst / "ids_map.json").write_text(json.dumps(mapping, indent=2))
    return {"name": "DRIVE", "n_train": len(train_ids), "n_test": len(test_ids), "n_total": idx}


def prepare_stare() -> dict:
    src = DEEPDYN / "STARE"
    dst = ROOT / "data" / "STARE"
    _clear_dataset_dir(dst)
    mapping = {}
    idx = 0
    ids = []
    names = sorted(p.name for p in (src / "stare-images").iterdir() if p.suffix == ".ppm")
    for name in names:
        stem = Path(name).stem  # im0001
        sid = _sid(idx)
        idx += 1
        mapping[sid] = name
        ids.append(sid)
        img = _load_any(src / "stare-images" / name)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        _save_png(dst / "images" / f"{sid}.png", img)
        mask = _load_any(src / "labels-ah" / f"{stem}.ah.pgm")
        _save_png(dst / "masks" / f"{sid}.png", _binarize(mask))
        fov = _generate_fov(img)
        _save_png(dst / "fov_masks" / f"{sid}.png", fov)
    # Literature split: first 10 train, last 10 test.
    half = len(ids) // 2
    _write_splits(dst, ids[:half], ids[half:])
    (dst / "ids_map.json").write_text(json.dumps(mapping, indent=2))
    return {"name": "STARE", "n_train": half, "n_test": len(ids) - half, "n_total": idx}


def prepare_chase() -> dict:
    src = DEEPDYN / "CHASEDB"
    dst = ROOT / "data" / "CHASE_DB1"
    _clear_dataset_dir(dst)
    mapping = {}
    idx = 0
    ids = []
    names = sorted(p.name for p in (src / "images").iterdir() if p.suffix == ".jpg")
    for name in names:
        stem = Path(name).stem  # Image_01L
        sid = _sid(idx)
        idx += 1
        mapping[sid] = name
        ids.append(sid)
        img = _load_any(src / "images" / name)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        _save_png(dst / "images" / f"{sid}.png", img)
        # Use 1stHO as primary ground truth.
        mask = _load_any(src / "manual" / f"{stem}_1stHO.png")
        _save_png(dst / "masks" / f"{sid}.png", _binarize(mask))
        fov = _generate_fov(img)
        _save_png(dst / "fov_masks" / f"{sid}.png", fov)
    # Literature split: first 20 train (Image_01L..Image_10R), last 8 test.
    _write_splits(dst, ids[:20], ids[20:])
    (dst / "ids_map.json").write_text(json.dumps(mapping, indent=2))
    return {"name": "CHASE_DB1", "n_train": 20, "n_test": len(ids) - 20, "n_total": idx}


def prepare_hrf() -> dict:
    src = HRF_RAW
    dst = ROOT / "data" / "HRF"
    _clear_dataset_dir(dst)
    mapping = {}
    idx = 0
    ids_by_cat = {"h": [], "dr": [], "g": []}
    names = sorted(p.name for p in (src / "images").iterdir())
    for name in names:
        stem = Path(name).stem  # 01_h
        m = re.match(r"(\d+)_(h|dr|g)", stem)
        if not m:
            continue
        cat = m.group(2)
        sid = _sid(idx)
        idx += 1
        mapping[sid] = name
        ids_by_cat[cat].append(sid)
        img = _load_any(src / "images" / name)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        _save_png(dst / "images" / f"{sid}.png", img)
        mask = _load_any(src / "manual1" / f"{stem}.tif")
        _save_png(dst / "masks" / f"{sid}.png", _binarize(mask))
        fov = _load_any(src / "mask" / f"{stem}_mask.tif")
        _save_png(dst / "fov_masks" / f"{sid}.png", _binarize(fov))
    # Stratified split: per category, first 10 train, last 5 test (total 30 train / 15 test).
    train_ids, test_ids = [], []
    for cat in ("h", "dr", "g"):
        cat_ids = sorted(ids_by_cat[cat])
        train_ids.extend(cat_ids[:10])
        test_ids.extend(cat_ids[10:])
    _write_splits(dst, train_ids, test_ids)
    (dst / "ids_map.json").write_text(json.dumps(mapping, indent=2))
    return {"name": "HRF", "n_train": len(train_ids), "n_test": len(test_ids), "n_total": idx}


def main() -> None:
    summary = []
    for fn in (prepare_drive, prepare_stare, prepare_chase, prepare_hrf):
        print(f"Preparing {fn.__name__}...")
        summary.append(fn())
        print(f"  done: {summary[-1]}")
    (ROOT / "results" / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    print("Summary:")
    for s in summary:
        print(f"  {s['name']:10s} train={s['n_train']:3d} test={s['n_test']:3d} total={s['n_total']:3d}")


if __name__ == "__main__":
    main()
