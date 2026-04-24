#!/usr/bin/env bash
# Fetch raw retinal datasets into data/raw/.
# Idempotent: re-running skips already-extracted dirs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${ROOT}/data/raw"
mkdir -p "${RAW}"

DEEPDYN_URL="https://codeload.github.com/sraashis/deepdyn/tar.gz/master"
HRF_URL="https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip"

# 1. deepdyn tarball (bundles DRIVE + STARE + CHASE_DB1 + splits + FOV masks for DRIVE).
if [ ! -d "${RAW}/deepdyn-master/data" ]; then
  echo "[1/2] Fetching deepdyn tarball (DRIVE + STARE + CHASE)..."
  curl -sSfL --retry 3 --max-time 600 -o "${RAW}/deepdyn.tar.gz" "${DEEPDYN_URL}"
  tar -xzf "${RAW}/deepdyn.tar.gz" -C "${RAW}" deepdyn-master/data
  echo "     done: ${RAW}/deepdyn-master/data"
else
  echo "[1/2] deepdyn already extracted, skipping."
fi

# 2. HRF.
if [ ! -d "${RAW}/HRF/images" ]; then
  echo "[2/2] Fetching HRF..."
  mkdir -p "${RAW}/HRF"
  curl -sSfL --retry 3 --max-time 900 -o "${RAW}/HRF/all.zip" "${HRF_URL}"
  (cd "${RAW}/HRF" && unzip -q all.zip)
  echo "     done: ${RAW}/HRF"
else
  echo "[2/2] HRF already extracted, skipping."
fi

echo "Raw data ready under ${RAW}"
echo "Next: python scripts/prepare_datasets.py"
