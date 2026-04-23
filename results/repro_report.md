# Reproduction Report (Kickoff)

This report documents the initial end-to-end execution of the project pipeline.

## Execution Context

- Environment: local `.venv`
- Data used in this run: **synthetic smoke datasets** generated via `src/utils/create_dummy_data.py`
- Purpose: validate the full training/evaluation flow and artifact generation before running on real DRIVE/STARE/CHASE_DB1

## Commands Executed

```bash
.venv/bin/python -m src.utils.create_dummy_data --data-root data
.venv/bin/python -m src.train --config configs/drive_baseline_smoke.yaml --experiment unet_drive_smoke
.venv/bin/python -m src.train --config configs/drive_rsf_smoke.yaml --experiment rsf_unet_drive_smoke
.venv/bin/python -m src.eval --config configs/drive_baseline_smoke.yaml --checkpoint outputs/unet_drive_smoke/best.pt --split test
.venv/bin/python -m src.eval --config configs/drive_rsf_smoke.yaml --checkpoint outputs/rsf_unet_drive_smoke/best.pt --split test
.venv/bin/python -m src.eval --config configs/ood_drive_to_stare_smoke.yaml --checkpoint outputs/rsf_unet_drive_smoke/best.pt --split test
.venv/bin/python -m src.eval --config configs/ood_drive_to_chase_smoke.yaml --checkpoint outputs/rsf_unet_drive_smoke/best.pt --split test
```

## Smoke Metrics (Synthetic Data)

- In-domain (DRIVE smoke)
  - `U-Net`: `AUC=0.926767`, `F1=0.000000`, `Se=0.000000`, `Sp=1.000000`, `Acc=0.847291`
  - `RSF-U-Net`: `AUC=0.697349`, `F1=0.000000`, `Se=0.000000`, `Sp=1.000000`, `Acc=0.847291`
- Out-of-domain (train DRIVE smoke RSF model, test on synthetic target)
  - `DRIVE -> STARE`: `AUC=0.694652`, `F1=0.000000`, `Se=0.000000`, `Sp=1.000000`, `Acc=0.908171`
  - `DRIVE -> CHASE_DB1`: `AUC=0.728572`, `F1=0.000000`, `Se=0.000000`, `Sp=1.000000`, `Acc=0.929849`

## Artifacts

- Checkpoints:
  - `outputs/unet_drive_smoke/best.pt`
  - `outputs/rsf_unet_drive_smoke/best.pt`
- Histories:
  - `outputs/unet_drive_smoke/history.json`
  - `outputs/rsf_unet_drive_smoke/history.json`

## Next Reproduction Step

Run the same flow with real DRIVE/STARE/CHASE_DB1 data and full configs:

```bash
python -m src.train --config configs/drive_baseline.yaml --experiment unet_drive
python -m src.train --config configs/drive_rsf.yaml --experiment rsf_unet_drive
python -m src.eval --config configs/ood_drive_to_stare.yaml --checkpoint outputs/rsf_unet_drive/best.pt --split test
python -m src.eval --config configs/ood_drive_to_chase.yaml --checkpoint outputs/rsf_unet_drive/best.pt --split test
```
