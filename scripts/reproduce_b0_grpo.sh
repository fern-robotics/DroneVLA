#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/workspace/DroneVLA-uavflow}
TRAIN=${TRAIN:-/workspace/uavflow_train_tasks}
TEST=${TEST:-/workspace/UAV-Flow/UAV-Flow-Eval/test_jsons}
RUN=${RUN:-/workspace/runs/b0_grpo_full}
PY=${PY:-/workspace/venv-uavflow/bin/python}
mkdir -p "$RUN"
"$PY" "$ROOT/scripts/validate_uavflow_pool.py" --train-dir "$TRAIN" --test-dir "$TEST"
"$PY" "$ROOT/scripts/sample_uavflow_tasks.py" --train-dir "$TRAIN" --test-dir "$TEST" \
  --out "$RUN/manifest.json" --count 100 --seed 7
# Requires the B0 token server on :5007.
while read -r task; do
  id=$(basename "$task" .json)
  "$PY" "$ROOT/scripts/collect_uavflow_groups.py" --task "$task" \
    --out "$RUN/$id" --groups 1 --group-size 4 --steps 20
done < <("$PY" -c "import json;print('\\n'.join(json.load(open('$RUN/manifest.json'))['tasks']))")
"$PY" "$ROOT/scripts/aggregate_uavflow_groups.py" --run "$RUN" --manifest "$RUN/manifest.json" --out "$RUN/groups.json"
"$PY" "$ROOT/scripts/train_b0_grpo.py" --groups "$RUN/groups.json" --out "$RUN/adapter" --max-groups 100
