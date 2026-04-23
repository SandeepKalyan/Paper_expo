from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from src.data.factory import build_dataloader
from src.metrics.vessel_metrics import compute_metrics_from_probs
from src.models.factory import build_model
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retinal vessel segmentation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    return parser.parse_args()


def _predict_probs(model: torch.nn.Module, images: torch.Tensor, tta: bool) -> np.ndarray:
    logits = model(images)
    probs = torch.sigmoid(logits)
    if not tta:
        return probs.cpu().numpy()[:, 0]

    # Simple TTA: horizontal and vertical flips.
    images_h = torch.flip(images, dims=[3])
    images_v = torch.flip(images, dims=[2])
    probs_h = torch.flip(torch.sigmoid(model(images_h)), dims=[3])
    probs_v = torch.flip(torch.sigmoid(model(images_v)), dims=[2])
    probs_avg = (probs + probs_h + probs_v) / 3.0
    return probs_avg.cpu().numpy()[:, 0]


def _apply_postprocess(probs: np.ndarray, use_post: bool) -> np.ndarray:
    if not use_post:
        return probs
    out = []
    kernel = np.ones((3, 3), np.uint8)
    for p in probs:
        b = (p > 0.5).astype(np.uint8) * 255
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
        out.append((b > 127).astype(np.float32))
    return np.stack(out, axis=0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = build_dataloader(cfg["data"], split=args.split, train=False)
    probs_all, target_all, fov_all = [], [], []
    tta = bool(cfg["eval"].get("tta", False))
    postprocess = bool(cfg["eval"].get("postprocess", False))

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            probs = _predict_probs(model, images, tta=tta)
            probs_all.append(probs)
            target_all.append(batch["mask"].numpy()[:, 0])
            fov_all.append(batch["fov_mask"].numpy()[:, 0])

    probs_np = np.concatenate(probs_all, axis=0)
    target_np = np.concatenate(target_all, axis=0)
    fov_np = np.concatenate(fov_all, axis=0)
    probs_np = _apply_postprocess(probs_np, use_post=postprocess)
    thresholds = cfg["eval"].get("threshold_candidates")
    if thresholds:
        best = None
        for th in thresholds:
            metrics = compute_metrics_from_probs(probs_np, target_np, fov_np, threshold=float(th))
            payload = {"threshold": float(th), **{k: round(v, 6) for k, v in metrics.items()}}
            print(payload)
            if best is None or metrics["F1"] > best["F1"]:
                best = {"threshold": float(th), **metrics}
        print({"best_by_f1": {k: round(v, 6) for k, v in best.items()}})
    else:
        metrics = compute_metrics_from_probs(
            probs_np,
            target_np,
            fov_np,
            threshold=float(cfg["eval"].get("threshold", 0.5)),
        )
        print({k: round(v, 6) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
