"""Ensemble evaluation: avg sigmoid probs of N checkpoints + TTA + threshold sweep.

Usage:
  python scripts/ensemble_eval.py \
      --config configs/drive_rsf_paper_laptop_safe.yaml \
      --checkpoints outputs/rsf_paper_laptop_safe/best.pt outputs/rsf_paper_a100_seed43/best.pt \
      --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.factory import build_dataloader
from src.metrics.vessel_metrics import compute_metrics_from_probs
from src.models.factory import build_model
from src.patch_eval import predict_patches_overlap
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble eval with TTA + threshold sweep")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--save-probs", type=str, default=None)
    p.add_argument("--save-metrics", type=str, default=None)
    return p.parse_args()


def _probs_for_model(
    model: torch.nn.Module,
    images: torch.Tensor,
    tta: bool,
    patch_overlap: bool,
    patch_size: int,
    stride: int,
) -> np.ndarray:
    if patch_overlap:
        probs = predict_patches_overlap(
            model, images, patch_size=patch_size, stride=stride, tta=tta
        )
        return probs.cpu().numpy()[:, 0]
    logits = model(images)
    probs = torch.sigmoid(logits)
    if not tta:
        return probs.cpu().numpy()[:, 0]
    images_h = torch.flip(images, dims=[3])
    images_v = torch.flip(images, dims=[2])
    probs_h = torch.flip(torch.sigmoid(model(images_h)), dims=[3])
    probs_v = torch.flip(torch.sigmoid(model(images_v)), dims=[2])
    probs_avg = (probs + probs_h + probs_v) / 3.0
    return probs_avg.cpu().numpy()[:, 0]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tta = bool(cfg["eval"].get("tta", True))
    patch_overlap = bool(cfg["eval"].get("patch_overlap", False))
    patch_size = int(cfg["eval"].get("patch_size", 256))
    stride = int(cfg["eval"].get("stride", 128))
    thresholds = cfg["eval"].get("threshold_candidates") or [0.5]

    loader = build_dataloader(cfg["data"], split=args.split, train=False)
    batches = list(loader)
    targets = np.concatenate([b["mask"].numpy()[:, 0] for b in batches], axis=0)
    fovs = np.concatenate([b["fov_mask"].numpy()[:, 0] for b in batches], axis=0)

    probs_sum = None
    n_ckpts = len(args.checkpoints)
    per_ckpt_metrics: list[dict] = []

    for ckpt_path in args.checkpoints:
        print(f"[ensemble] loading {ckpt_path}")
        model = build_model(cfg["model"]).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        probs_batches: list[np.ndarray] = []
        with torch.no_grad():
            for b in batches:
                images = b["image"].to(device)
                probs = _probs_for_model(
                    model,
                    images,
                    tta=tta,
                    patch_overlap=patch_overlap,
                    patch_size=patch_size,
                    stride=stride,
                )
                probs_batches.append(probs)
        probs_np = np.concatenate(probs_batches, axis=0)

        for th in thresholds:
            m = compute_metrics_from_probs(probs_np, targets, fovs, threshold=float(th))
            per_ckpt_metrics.append(
                {"ckpt": ckpt_path, "threshold": float(th), **m}
            )
        if probs_sum is None:
            probs_sum = probs_np.astype(np.float64)
        else:
            probs_sum = probs_sum + probs_np.astype(np.float64)

    probs_avg = (probs_sum / n_ckpts).astype(np.float32)

    ensemble_metrics: list[dict] = []
    best = None
    for th in thresholds:
        m = compute_metrics_from_probs(probs_avg, targets, fovs, threshold=float(th))
        row = {"threshold": float(th), **m}
        ensemble_metrics.append(row)
        if best is None or m["F1"] > best["F1"]:
            best = row
        print(f"[ensemble th={th:.2f}] " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))

    print("[ensemble BEST-F1]", best)

    if args.save_probs:
        np.save(args.save_probs, probs_avg)
        print(f"[ensemble] probs saved -> {args.save_probs}")
    if args.save_metrics:
        payload = {
            "checkpoints": args.checkpoints,
            "per_checkpoint": per_ckpt_metrics,
            "ensemble_per_threshold": ensemble_metrics,
            "ensemble_best_by_f1": best,
            "tta": tta,
            "patch_overlap": patch_overlap,
        }
        Path(args.save_metrics).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_metrics, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[ensemble] metrics saved -> {args.save_metrics}")


if __name__ == "__main__":
    main()
