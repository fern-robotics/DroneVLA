#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-/workspace/venv311/bin/python}
ROOT=${ROOT:-/workspace/DroneVLA-uavflow}
PID=$(ss -ltnp 2>/dev/null | awk '/:8000/ {match($0,/pid=([0-9]+)/,a); print a[1]; exit}')
[ -n "$PID" ] && kill "$PID" 2>/dev/null || true
sleep 3
cd "$ROOT"
PYTHONPATH=. "$PY" scripts/train_b0_ppo.py \
  --groups /workspace/runs/b0_grpo_100/groups.json \
  --out /workspace/runs/b0_ppo_smoke --max-groups 1 --lr 1e-5 \
  > /workspace/b0_ppo_smoke.log 2>&1
touch /workspace/runs/b0_ppo_smoke/DONE
