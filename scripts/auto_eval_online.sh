#!/bin/bash
set -uo pipefail
ROOT=/home/digilab/ws/skripsi-ryan-adidaru
REPO=$ROOT/repo
DATA=/mnt/data/skripsi-ryan-adidaru
TAG=online150
ADAPTER=$DATA/runs/${TAG}_adapter_r15
OUT=$DATA/runs/eval_${TAG}_244
LOG=$DATA/eval_${TAG}_244.log
while true; do
  if [ -f "$ADAPTER/adapter_model.safetensors" ] && grep -q "ONLINE_DONE" "$DATA/${TAG}.launch.log" 2>/dev/null; then break; fi
  sleep 60
done
pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true
sleep 2
(cd "$REPO" && PYTHONPATH=$REPO/stubs:$ROOT/UAV-Flow/OpenVLA-UAV:. nohup $ROOT/envs/openvla/bin/python scripts/serve_openvla_actions.py --vla-path $DATA/openvla-uav --adapter "$ADAPTER" --unnorm-key sim --prompt-style openvla_uav --absolute-output --batch-size 1 --port 8010 >> "$LOG" 2>&1 &)
for i in $(seq 1 120); do ss -ltn | grep -q ":8010" && break; sleep 1; done
if ! ss -ltn | grep -q ":8010"; then echo SERVER_FAIL >> "$LOG"; exit 1; fi
if ! ss -ltn | grep -q ":5007"; then (cd "$REPO" && PYTHONPATH=. nohup $ROOT/envs/uavflow/bin/python integrations/uavflow_sim/policy_proxy.py --upstream http://127.0.0.1:8010 --port 5007 >> "$LOG" 2>&1 &); sleep 2; fi
rm -rf "$OUT"
(cd "$REPO" && PYTHONUNBUFFERED=1 PYTHONPATH=$REPO:$ROOT/UAV-Flow/UAV-Flow-Eval $ROOT/envs/uavflow/bin/python scripts/eval_uavflow_test_5090.py --test-dir $ROOT/UAV-Flow/UAV-Flow-Eval/test_jsons --out "$OUT" --workers 1 --retries 2 --rollouts 1 --limit 244 --proxy http://127.0.0.1:5007/predict >> "$LOG" 2>&1)
pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true
echo AUTO_EVAL_DONE >> "$LOG"
