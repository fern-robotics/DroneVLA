#!/usr/bin/env bash
set -euo pipefail
OUT=${OUT:-/workspace/runs/eval_b0_baseline}
PY=${PY:-/workspace/venv-uavflow/bin/python}
SERVER_PY=${SERVER_PY:-/workspace/venv311/bin/python}
ROOT=${ROOT:-/workspace/DroneVLA-uavflow}
TEST=${TEST:-/workspace/UAV-Flow/UAV-Flow-Eval/test_jsons}
mkdir -p "$OUT"
PID=$(ss -ltnp 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
[ -n "$PID" ] && kill "$PID" 2>/dev/null || true
sleep 3
cd "$ROOT"
PYTHONPATH=. nohup "$SERVER_PY" scripts/serve_openvla_actions.py \
  --vla-path /workspace/openvla-uav --unnorm-key sim \
  --prompt-style openvla_uav --absolute-output --batch-size 3 --port 8000 \
  > /workspace/b0_baseline_server.log 2>&1 &
for _ in $(seq 1 60); do ss -ltn | grep -q ':8000' && break; sleep 2; done
"$PY" "$ROOT/scripts/eval_uavflow_test.py" --test-dir "$TEST" --out "$OUT" \
  --workers 3 --steps 20 --rollouts 1
echo DONE > "$OUT/EVAL_DONE"
