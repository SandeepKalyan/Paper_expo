# Setup on a Fresh Machine (e.g. 3080 desktop)

This repo excludes raw + prepared data from git (too large). Everything is
re-generated from public sources by running two scripts.

## Prereqs

- Python 3.9+
- CUDA 11.8 or 12.x (matches `requirements.txt` torch build)
- ~1 GB free disk for datasets + outputs

## Steps

```bash
git clone https://github.com/SandeepKalyan/Paper_expo.git
cd Paper_expo

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install matplotlib scikit-image scipy pypdf  # for EDA + report
```

## Fetch + prepare data

```bash
bash scripts/fetch_datasets.sh     # ~2 min, downloads deepdyn + HRF
python scripts/prepare_datasets.py # ~1 min, normalizes to unified PNG layout
python scripts/qc_datasets.py      # QC sweep; prints 0 issues
python scripts/eda.py              # optional: regen EDA report + figs
```

After this you will have:

```
data/
  DRIVE/    (40 images: 20 train / 20 test)
  STARE/    (20 images: 10 train / 10 test)
  CHASE_DB1/(28 images: 20 train /  8 test)
  HRF/      (45 images: 30 train / 15 test)
```

## Train

Baseline U-Net on DRIVE:
```bash
python -m src.train --config configs/drive_baseline.yaml --experiment unet_drive
```

RSF-Conv + U-Net on DRIVE:
```bash
python -m src.train --config configs/drive_rsf.yaml --experiment rsf_unet_drive
```

## OOD evaluation

```bash
python -m src.eval --config configs/ood_drive_to_stare.yaml \
  --checkpoint outputs/rsf_unet_drive/best.pt --split test

python -m src.eval --config configs/ood_drive_to_chase.yaml \
  --checkpoint outputs/rsf_unet_drive/best.pt --split test
```

## Troubleshooting

- `No module named src`: run commands from repo root (not inside `src/`).
- `data/DRIVE/splits/train.txt` missing → `prepare_datasets.py` not run yet.
- GPU OOM on HRF eval → use sliding-window eval helper (TBD).
