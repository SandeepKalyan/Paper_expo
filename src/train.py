from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.factory import build_dataloader
from src.losses import build_loss
from src.metrics.vessel_metrics import compute_metrics_from_logits
from src.models.factory import build_model
from src.models.rsf_conv import count_parameters
from src.utils.config import load_config
from src.utils.io import ensure_dir, write_json
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train retinal vessel segmentation models.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    return parser.parse_args()


def evaluate(model: nn.Module, loader, device: torch.device, threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    logits_all, target_all, fov_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images).cpu().numpy()[:, 0]
            logits_all.append(logits)
            target_all.append(batch["mask"].numpy()[:, 0])
            fov_all.append(batch["fov_mask"].numpy()[:, 0])

    logits_np = np.concatenate(logits_all, axis=0)
    target_np = np.concatenate(target_all, axis=0)
    fov_np = np.concatenate(fov_all, axis=0)
    return compute_metrics_from_logits(logits_np, target_np, fov_np, threshold=threshold)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)

    train_loader = build_dataloader(cfg["data"], split=cfg["data"].get("train_split", "train"), train=True)
    val_loader = build_dataloader(cfg["data"], split=cfg["data"].get("val_split", "test"), train=False)

    loss_name = str(cfg["train"].get("loss", "bce"))
    criterion = build_loss(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["optim"].get("lr", 2e-4)))
    epochs = int(cfg["train"].get("epochs", 200))
    threshold = float(cfg["eval"].get("threshold", 0.5))

    out_dir = ensure_dir(Path(cfg["train"].get("output_dir", "outputs")) / args.experiment)
    write_json(
        out_dir / "model_info.json",
        {"trainable_params": count_parameters(model), "device": str(device), "loss": loss_name},
    )

    best_auc = -1.0
    best_ckpt = out_dir / "best.pt"
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = evaluate(model, val_loader, device=device, threshold=threshold)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        row = {"epoch": epoch, "train_loss": avg_loss, **metrics}
        history.append(row)
        print(
            f"[{epoch:03d}] loss={avg_loss:.4f} AUC={metrics['AUC']:.4f} "
            f"F1={metrics['F1']:.4f} Se={metrics['Se']:.4f} Sp={metrics['Sp']:.4f}"
        )

        if metrics["AUC"] > best_auc:
            best_auc = metrics["AUC"]
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "config": cfg, "metrics": metrics},
                best_ckpt,
            )

    write_json(out_dir / "history.json", {"history": history, "best_auc": best_auc})
    print(f"Training complete. Best AUC: {best_auc:.4f}. Checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
