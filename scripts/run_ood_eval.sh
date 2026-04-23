#!/usr/bin/env bash
set -euo pipefail

CKPT=${1:-outputs/rsf_unet_drive/best.pt}

python -m src.eval --config configs/ood_drive_to_stare.yaml --checkpoint "$CKPT" --split test
python -m src.eval --config configs/ood_drive_to_chase.yaml --checkpoint "$CKPT" --split test
