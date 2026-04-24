"""QC sweep over prepared datasets.

Checks per sample:
- image/mask/fov exist and load
- matching spatial shape
- mask binary (unique in {0,255})
- fov binary
- mask ⊂ fov (no vessel pixels outside FOV)
- vessel pixel fraction inside FOV within sane bounds [0.01, 0.35]
- image not all-zero / corrupt

Cross-sample:
- duplicate SHA256 of images within + across datasets

Writes results/data_qc.json + prints summary. Samples failing hard checks
are listed in the 'issues' array (training code may filter them).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["DRIVE", "STARE", "CHASE_DB1", "HRF"]
VESSEL_FRAC_BOUNDS = (0.01, 0.35)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"cv2 failed on {path}")
    return arr


def _load_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"cv2 failed on {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def qc_dataset(name: str) -> dict:
    root = ROOT / "data" / name
    ids = sorted(p.stem for p in (root / "images").glob("*.png"))
    per_sample = []
    issues = []
    hashes: dict[str, str] = {}

    for sid in ids:
        entry = {"sid": sid, "ok": True, "checks": {}}
        try:
            img = _load_rgb(root / "images" / f"{sid}.png")
            mask = _load_gray(root / "masks" / f"{sid}.png")
            fov = _load_gray(root / "fov_masks" / f"{sid}.png")
        except Exception as e:
            entry["ok"] = False
            entry["checks"]["load_error"] = str(e)
            issues.append(entry)
            per_sample.append(entry)
            continue

        H, W = img.shape[:2]
        entry["checks"]["shape"] = [H, W]
        entry["checks"]["img_mean"] = float(img.mean())
        if img.mean() < 1.0:
            entry["ok"] = False
            entry["checks"]["black_image"] = True

        if mask.shape != (H, W):
            entry["ok"] = False
            entry["checks"]["mask_shape_mismatch"] = list(mask.shape)
        if fov.shape != (H, W):
            entry["ok"] = False
            entry["checks"]["fov_shape_mismatch"] = list(fov.shape)

        mask_uniq = np.unique(mask).tolist()
        fov_uniq = np.unique(fov).tolist()
        entry["checks"]["mask_unique_count"] = len(mask_uniq)
        entry["checks"]["fov_unique_count"] = len(fov_uniq)
        # Accept near-binary (png compression can intro 0,255 + small noise).
        m_bin = (mask > 127).astype(np.uint8)
        f_bin = (fov > 127).astype(np.uint8)
        non_bin_mask = int(((mask > 0) & (mask < 255) & (mask != m_bin * 255)).sum())
        entry["checks"]["mask_non_binary_px"] = non_bin_mask

        outside = int(((m_bin == 1) & (f_bin == 0)).sum())
        fov_area = int(f_bin.sum())
        entry["checks"]["mask_px_outside_fov"] = outside
        entry["checks"]["fov_area"] = fov_area
        entry["checks"]["fov_frac"] = fov_area / (H * W) if H * W else 0.0

        vessel_fov = int(((m_bin == 1) & (f_bin == 1)).sum())
        vessel_frac = vessel_fov / fov_area if fov_area else 0.0
        entry["checks"]["vessel_frac_in_fov"] = vessel_frac
        if not (VESSEL_FRAC_BOUNDS[0] <= vessel_frac <= VESSEL_FRAC_BOUNDS[1]):
            entry["ok"] = False
            entry["checks"]["vessel_frac_out_of_bounds"] = True

        hashes[sid] = _sha256(root / "images" / f"{sid}.png")
        if not entry["ok"]:
            issues.append(entry)
        per_sample.append(entry)

    # Intra-set dup detection.
    dup_groups: dict[str, list[str]] = defaultdict(list)
    for sid, h in hashes.items():
        dup_groups[h].append(sid)
    intra_dups = [v for v in dup_groups.values() if len(v) > 1]

    return {
        "name": name,
        "n_samples": len(ids),
        "n_issues": len(issues),
        "intra_dups": intra_dups,
        "sample_hashes": hashes,
        "per_sample": per_sample,
        "issues": issues,
    }


def main() -> None:
    reports = [qc_dataset(n) for n in DATASETS]
    # Cross-dataset dup check.
    global_hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rep in reports:
        for sid, h in rep["sample_hashes"].items():
            global_hashes[h].append((rep["name"], sid))
    cross_dups = [v for v in global_hashes.values() if len(v) > 1]

    out = {
        "datasets": [
            {k: v for k, v in r.items() if k != "sample_hashes"} for r in reports
        ],
        "cross_dataset_dups": cross_dups,
    }
    (ROOT / "results" / "data_qc.json").write_text(json.dumps(out, indent=2))

    print("QC Summary:")
    for r in reports:
        print(
            f"  {r['name']:10s} n={r['n_samples']:3d}  issues={r['n_issues']:3d}  "
            f"intra_dups={len(r['intra_dups'])}"
        )
    print(f"  Cross-dataset dups: {len(cross_dups)}")
    # Flag worst samples.
    for r in reports:
        if r["issues"]:
            print(f"  [!] {r['name']} failing samples:")
            for it in r["issues"][:5]:
                print(f"      {it['sid']} -> {list(it['checks'].keys())[-1]}")


if __name__ == "__main__":
    main()
