#!/usr/bin/env bash
# Poll a remote training pod. Usage:
#   SSH_TARGET="root@ssh1.vast.ai -p 12345" bash scripts/track_remote.sh
# Or pass explicit:
#   bash scripts/track_remote.sh "root@1.2.3.4 -p 12345"

set -euo pipefail
TARGET="${1:-${SSH_TARGET:-}}"
if [ -z "${TARGET}" ]; then
  echo "usage: $0 \"user@host -p PORT\"" >&2
  exit 1
fi

echo "=== remote time ==="
ssh -o StrictHostKeyChecking=no ${TARGET} "date -Is"

echo ""
echo "=== GPU ==="
ssh -o StrictHostKeyChecking=no ${TARGET} "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv"

echo ""
echo "=== last 5 epochs per experiment ==="
ssh -o StrictHostKeyChecking=no ${TARGET} "cd /workspace/Paper_expo && for f in outputs/logs/*.log; do
  echo '--- '\$f' ---'
  tr '\\r' '\\n' < \$f | grep -E '^\[[0-9]' | tail -5
done"

echo ""
echo "=== latest metrics.jsonl per experiment ==="
ssh -o StrictHostKeyChecking=no ${TARGET} "cd /workspace/Paper_expo && for f in outputs/*/metrics.jsonl; do
  echo '--- '\$f' ---'
  tail -3 \$f
done"
