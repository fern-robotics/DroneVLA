#!/usr/bin/env bash
set -euo pipefail
OUT=${OUT:-/workspace/runs/eval_b0_baseline}
PY=${PY:-/workspace/venv-uavflow/bin/python}
SERVER_PY=${SERVER_PY:-/workspace/venv311/bin/python}
ROOT=${ROOT:-/workspace/DroneVLA-uavflow}
TEST=${TEST:-/workspace/UAV-Flow/UAV-Flow-Eval/test_jsons}
mkdir -p "$OUT"
cd "$ROOT"
PYTHONPATH=. nohup "$SERVER_PY" scripts/serve_openvla_actions.py \
  --vla-path /workspace/openvla-uav --unnorm-key sim \
  --prompt-style openvla_uav --absolute-output --batch-size 3 --port 8001 \
  > /workspace/b0_baseline_server_8001.log 2>&1 &
PYTHONPATH=. nohup /workspace/venv-uavflow/bin/python integrations/uavflow_sim/policy_proxy.py \
  --upstream http://127.0.0.1:8001 --port 5008 > /workspace/b0_baseline_proxy.log 2>&1 &
for _ in $(seq 1 60); do ss -ltn | grep -q ':8001' && ss -ltn | grep -q ':5008' && break; sleep 2; done
"$PY" scripts/eval_uavflow_test.py --test-dir "$TEST" --out "$OUT" \
  --workers 1 --steps 20 --rollouts 1 --proxy http://127.0.0.1:5008/predict
echo DONE > "$OUT/EVAL_DONE"
