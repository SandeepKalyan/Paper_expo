#!/usr/bin/env bash
set -euo pipefail

# Smoke ablations on synthetic data.
.venv/bin/python -m src.train --config configs/ablation/bce_dice_smoke.yaml --experiment rsf_ablate_bce_dice_smoke
.venv/bin/python -m src.train --config configs/ablation/bce_focal_smoke.yaml --experiment rsf_ablate_bce_focal_smoke
.venv/bin/python -m src.train --config configs/ablation/vessel_sampling_smoke.yaml --experiment rsf_ablate_vessel_sampling_smoke

.venv/bin/python -m src.eval --config configs/ablation/bce_dice_smoke.yaml --checkpoint outputs/rsf_ablate_bce_dice_smoke/best.pt --split test
.venv/bin/python -m src.eval --config configs/ablation/bce_focal_smoke.yaml --checkpoint outputs/rsf_ablate_bce_focal_smoke/best.pt --split test
.venv/bin/python -m src.eval --config configs/ablation/vessel_sampling_smoke.yaml --checkpoint outputs/rsf_ablate_vessel_sampling_smoke/best.pt --split test

# TTA + threshold sweep + postprocess on best baseline smoke checkpoint.
.venv/bin/python -m src.eval --config configs/ablation/tta_postprocess_smoke.yaml --checkpoint outputs/rsf_unet_drive_smoke/best.pt --split test
