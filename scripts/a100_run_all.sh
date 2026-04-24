#!/usr/bin/env bash
# Sequential A100 job queue. Each run writes structured logs + checkpoints.
# Intended: run this after a100_bootstrap.sh finishes.
# ~7h total on A100 40GB with bf16 + paper augs + patch_overlap eval.

set -euo pipefail
. .venv/bin/activate
mkdir -p outputs/logs

run() {
  local cfg="$1"; local exp="$2"
  local log="outputs/logs/${exp}.log"
  local t0=$(date +%s)
  echo "=== START $(date -Is) exp=${exp} cfg=${cfg} ==="
  python -m src.train --config "${cfg}" --experiment "${exp}" 2>&1 | tee "${log}"
  local dt=$(($(date +%s) - t0))
  echo "=== END   $(date -Is) exp=${exp} dur=${dt}s ==="
  python - <<PY
import json
with open("outputs/${exp}/history.json") as f:
    h = json.load(f)
print(f"[${exp}] best_auc={h['best_auc']:.4f} epochs={len(h['history'])}")
PY
}

eval_cross() {
  local cfg="$1"; local ckpt="$2"; local tag="$3"
  local log="outputs/logs/eval_${tag}.log"
  echo "=== EVAL $(date -Is) tag=${tag} ==="
  python -m src.eval --config "${cfg}" --checkpoint "${ckpt}" --split test 2>&1 | tee "${log}"
}

# 1. Paper-faithful RSF on DRIVE
run configs/a100/drive_rsf_paper.yaml   rsf_unet_paper

# 2. RSF "plus" on DRIVE (BCE+Dice + bigger batch + vessel sampling + TTA)
run configs/a100/drive_rsf_plus.yaml    rsf_unet_plus

# 3. Baseline UNet on DRIVE (sanity ref; laptop already has this too)
run configs/a100/drive_baseline.yaml    unet_drive_a100

# 4. In-domain STARE
run configs/a100/stare_rsf.yaml         rsf_stare

# 5. In-domain CHASE_DB1
run configs/a100/chase_rsf.yaml         rsf_chase_db1

# 6. OOD evals from DRIVE-trained RSF
eval_cross configs/ood_drive_to_stare.yaml outputs/rsf_unet_paper/best.pt drive_to_stare_paper
eval_cross configs/ood_drive_to_chase.yaml outputs/rsf_unet_paper/best.pt drive_to_chase_paper
eval_cross configs/ood_drive_to_stare.yaml outputs/rsf_unet_plus/best.pt  drive_to_stare_plus
eval_cross configs/ood_drive_to_chase.yaml outputs/rsf_unet_plus/best.pt  drive_to_chase_plus

echo "ALL DONE $(date -Is)"
