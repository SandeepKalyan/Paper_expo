#!/usr/bin/env bash
# A100 queue with proper paper RSFConv port.
# Intended to run AFTER rsf_unet_plus finishes on A100.

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
}

eval_cross() {
  local cfg="$1"; local ckpt="$2"; local tag="$3"
  local log="outputs/logs/eval_${tag}.log"
  echo "=== EVAL $(date -Is) tag=${tag} ==="
  python -m src.eval --config "${cfg}" --checkpoint "${ckpt}" --split test 2>&1 | tee "${log}"
}

# Proper paper RSFConv port, in-domain DRIVE
run configs/a100/drive_rsf_paper_proper.yaml rsf_paper_proper

# OOD from proper RSF
eval_cross configs/ood_drive_to_stare.yaml outputs/rsf_paper_proper/best.pt drive_to_stare_proper
eval_cross configs/ood_drive_to_chase.yaml outputs/rsf_paper_proper/best.pt drive_to_chase_proper

echo "PROPER DONE $(date -Is)"
