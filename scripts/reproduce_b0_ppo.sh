#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/workspace/DroneVLA-uavflow}
TRAIN=${TRAIN:-/workspace/uavflow_train_tasks}
TEST=${TEST:-/workspace/UAV-Flow/UAV-Flow-Eval/test_jsons}
RUN=${RUN:-/workspace/runs/b0_ppo_100}
PY=${PY:-/workspace/venv-uavflow/bin/python}
mkdir -p "$RUN"
"$PY" "$ROOT/scripts/validate_uavflow_pool.py" --train-dir "$TRAIN" --test-dir "$TEST"
"$PY" "$ROOT/scripts/sample_uavflow_tasks.py" --train-dir "$TRAIN" --test-dir "$TEST" \
  --out "$RUN/manifest.json" --count 100 --seed 7
# B0 token server must already be running on :5007.
"$PY" "$ROOT/scripts/collect_uavflow_parallel.py" --manifest "$RUN/manifest.json" \
  --out "$RUN" --workers 3 --steps 10 --timeout 60
"$PY" "$ROOT/scripts/train_b0_ppo.py" --groups "$RUN/groups.json" \
  --out "$RUN/adapter" --max-groups 100 --lr 1e-5
# Serve $RUN/adapter, then run scripts/eval_uavflow_test.py --workers 3 --rollouts 1
