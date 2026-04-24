#!/usr/bin/env bash
# Sequential queue: wait for seed43 to finish, train seed44, run all evals,
# run ensemble, run OOD (DRIVE -> STARE, DRIVE -> CHASE) on both + ensemble.
#
# Expected to be started AFTER seed43 training already running.

set -euo pipefail
cd /workspace/Paper_expo
source .venv/bin/activate
mkdir -p outputs/logs outputs/eval

stamp() { date -Is; }

echo "[$(stamp)] QUEUE_START"

# 1. Wait for seed43 training to finish
while pgrep -af "rsf_paper_a100_seed43" > /dev/null; do
  echo "[$(stamp)] waiting for seed43..."
  sleep 60
done
echo "[$(stamp)] seed43 finished"

# 2. Launch seed44 training
echo "[$(stamp)] launching seed44 training"
python -m src.train \
  --config configs/a100/drive_rsf_paper_a100_seed44.yaml \
  --experiment rsf_paper_a100_seed44 \
  > outputs/logs/rsf_paper_a100_seed44.log 2>&1

echo "[$(stamp)] seed44 finished"

# 3. Per-checkpoint eval (TTA + threshold sweep) for both
echo "[$(stamp)] eval seed43"
python -m src.eval \
  --config configs/a100/drive_rsf_paper_a100_seed43.yaml \
  --checkpoint outputs/rsf_paper_a100_seed43/best.pt \
  --split test \
  > outputs/eval/eval_seed43.log 2>&1 || true

echo "[$(stamp)] eval seed44"
python -m src.eval \
  --config configs/a100/drive_rsf_paper_a100_seed44.yaml \
  --checkpoint outputs/rsf_paper_a100_seed44/best.pt \
  --split test \
  > outputs/eval/eval_seed44.log 2>&1 || true

# 4. Ensemble seed43 + seed44 (+ laptop if present)
ENS_CKPTS="outputs/rsf_paper_a100_seed43/best.pt outputs/rsf_paper_a100_seed44/best.pt"
if [ -f outputs/rsf_paper_laptop_safe/best.pt ]; then
  ENS_CKPTS="$ENS_CKPTS outputs/rsf_paper_laptop_safe/best.pt"
fi
echo "[$(stamp)] ensemble DRIVE test: $ENS_CKPTS"
python scripts/ensemble_eval.py \
  --config configs/a100/drive_rsf_paper_a100_seed43.yaml \
  --checkpoints $ENS_CKPTS \
  --split test \
  --save-metrics outputs/eval/ensemble_drive.json \
  > outputs/eval/ensemble_drive.log 2>&1 || true

# 5. OOD: DRIVE -> STARE (each ckpt, then ensemble)
for exp in rsf_paper_a100_seed43 rsf_paper_a100_seed44; do
  echo "[$(stamp)] OOD DRIVE->STARE $exp"
  python -m src.eval \
    --config configs/ood_drive_to_stare_paper.yaml \
    --checkpoint outputs/${exp}/best.pt \
    --split test \
    > outputs/eval/ood_stare_${exp}.log 2>&1 || true
done

echo "[$(stamp)] ensemble OOD DRIVE->STARE"
python scripts/ensemble_eval.py \
  --config configs/ood_drive_to_stare_paper.yaml \
  --checkpoints $ENS_CKPTS \
  --split test \
  --save-metrics outputs/eval/ensemble_ood_stare.json \
  > outputs/eval/ensemble_ood_stare.log 2>&1 || true

# 6. OOD: DRIVE -> CHASE (each ckpt, then ensemble)
for exp in rsf_paper_a100_seed43 rsf_paper_a100_seed44; do
  echo "[$(stamp)] OOD DRIVE->CHASE $exp"
  python -m src.eval \
    --config configs/ood_drive_to_chase_paper.yaml \
    --checkpoint outputs/${exp}/best.pt \
    --split test \
    > outputs/eval/ood_chase_${exp}.log 2>&1 || true
done

echo "[$(stamp)] ensemble OOD DRIVE->CHASE"
python scripts/ensemble_eval.py \
  --config configs/ood_drive_to_chase_paper.yaml \
  --checkpoints $ENS_CKPTS \
  --split test \
  --save-metrics outputs/eval/ensemble_ood_chase.json \
  > outputs/eval/ensemble_ood_chase.log 2>&1 || true

echo "[$(stamp)] QUEUE_DONE"
