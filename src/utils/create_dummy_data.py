from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def draw_vessels(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(rng.integers(18, 30)):
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x2, y2 = int(rng.integers(0, w)), int(rng.integers(0, h))
        thickness = int(rng.integers(1, 4))
        cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=thickness)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return (mask > 40).astype(np.uint8) * 255


def make_fov(h: int, w: int) -> np.ndarray:
    fov = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(fov, (w // 2, h // 2), int(min(h, w) * 0.45), 255, -1)
    return fov


def write_split(root: Path, train_ids: list[str], test_ids: list[str]) -> None:
    (root / "splits").mkdir(parents=True, exist_ok=True)
    (root / "splits" / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (root / "splits" / "test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")


def build_dataset(root: Path, n_train: int, n_test: int, size: tuple[int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    for d in ["images", "masks", "fov_masks"]:
        (root / d).mkdir(parents=True, exist_ok=True)

    h, w = size
    fov = make_fov(h, w)
    ids = []
    for i in range(n_train + n_test):
        sid = f"img_{i:03d}"
        ids.append(sid)
        vessel = draw_vessels(h, w, rng)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 1] = (vessel * 0.7).astype(np.uint8)
        img[..., 0] = rng.integers(0, 30, size=(h, w), dtype=np.uint8)
        img[..., 2] = rng.integers(0, 30, size=(h, w), dtype=np.uint8)
        img = cv2.add(img, np.stack([fov // 8, fov // 6, fov // 10], axis=-1))

        cv2.imwrite(str(root / "images" / f"{sid}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(root / "masks" / f"{sid}.png"), vessel)
        cv2.imwrite(str(root / "fov_masks" / f"{sid}.png"), fov)

    write_split(root, ids[:n_train], ids[n_train:])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    build_dataset(data_root / "DRIVE", n_train=6, n_test=2, size=(584, 565), seed=args.seed)
    build_dataset(data_root / "STARE", n_train=6, n_test=2, size=(700, 605), seed=args.seed + 1)
    build_dataset(data_root / "CHASE_DB1", n_train=6, n_test=2, size=(999, 960), seed=args.seed + 2)
    print(f"Dummy datasets written to {data_root.resolve()}")


if __name__ == "__main__":
    main()
