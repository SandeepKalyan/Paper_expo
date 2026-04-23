# RSF-Conv Reproduction Starter

This repository bootstraps a reproducible PyTorch project to implement and evaluate **RSF-Conv** for retinal vessel segmentation based on the paper in `Base_paper.pdf`.

## Scope

- Baseline backbone: `U-Net`
- RSF integration: `RSF-Conv + U-Net`
- Metrics (inside FOV): `Se`, `Sp`, `F1`, `Acc`, `AUC`
- Primary starting dataset: `DRIVE`
- Planned OOD evaluation: `DRIVE -> STARE`, `DRIVE -> CHASE_DB1`

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Layout

Expected dataset root:

```text
data/
  DRIVE/
    images/
    masks/
    fov_masks/
    splits/
      train.txt
      test.txt
  STARE/
    images/
    masks/
    fov_masks/
    splits/
      train.txt
      test.txt
  CHASE_DB1/
    images/
    masks/
    fov_masks/
    splits/
      train.txt
      test.txt
```

`train.txt` / `test.txt` are newline-separated sample IDs (without extension).

## Training

Baseline U-Net:

```bash
python -m src.train --config configs/drive_baseline.yaml --experiment unet_drive
```

RSF-Conv + U-Net:

```bash
python -m src.train --config configs/drive_rsf.yaml --experiment rsf_unet_drive
```

## Evaluation

In-domain or OOD evaluation from a saved checkpoint:

```bash
python -m src.eval \
  --config configs/drive_rsf.yaml \
  --checkpoint outputs/rsf_unet_drive/best.pt \
  --split test
```

Example OOD evaluation:

```bash
python -m src.eval \
  --config configs/ood_drive_to_stare.yaml \
  --checkpoint outputs/rsf_unet_drive/best.pt \
  --split test
```

## Notes

- This codebase is intentionally modular for ablations (`configs/ablation/`).
- `scripts/` contains reproducible command runners.
- `results/` stores experiment reports.
