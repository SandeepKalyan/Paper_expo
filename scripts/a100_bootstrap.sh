#!/usr/bin/env bash
# One-shot A100 pod bootstrap. Run after SSH into the pod.
# Clones repo, sets up env, fetches datasets, prepares splits.
# Expected: ~5-7 min total.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/SandeepKalyan/Paper_expo.git}"
BRANCH="${BRANCH:-laptop/src-data-and-rsf-opts}"
WORKDIR="${WORKDIR:-/workspace/Paper_expo}"

echo "[1/6] Install system deps..."
apt-get update -qq
apt-get install -y -qq git curl unzip python3-venv python3-pip >/dev/null

echo "[2/6] Clone repo (branch: ${BRANCH})..."
if [ ! -d "${WORKDIR}/.git" ]; then
  git clone --branch "${BRANCH}" "${REPO_URL}" "${WORKDIR}"
else
  (cd "${WORKDIR}" && git fetch origin && git checkout "${BRANCH}" && git pull)
fi
cd "${WORKDIR}"

echo "[3/6] Create venv + install python deps..."
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt pypdf matplotlib

echo "[4/6] Verify CUDA torch..."
python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
PY

echo "[5/6] Fetch datasets..."
bash scripts/fetch_datasets.sh

echo "[6/6] Prepare datasets..."
python scripts/prepare_datasets.py

mkdir -p outputs/logs
echo "Bootstrap complete. Workdir: ${WORKDIR}"
echo "Kick off first run:"
echo "  python -m src.train --config configs/a100/drive_rsf_paper.yaml --experiment rsf_unet_paper 2>&1 | tee outputs/logs/rsf_paper.log"
