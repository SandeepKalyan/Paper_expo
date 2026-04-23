#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/drive_baseline.yaml --experiment unet_drive
python -m src.train --config configs/drive_rsf.yaml --experiment rsf_unet_drive
