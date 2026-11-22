#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/digilab/ws/skripsi-ryan-adidaru
REPO=$ROOT/repo
DATA=/mnt/data/skripsi-ryan-adidaru
EVAL_PY=$ROOT/envs/uavflow/bin/python
SERVE_PY=$ROOT/envs/openvla/bin/python
LOG=$DATA/schedules_queue.log
CSV=$DATA/runs/ablation_results.csv
: > "$LOG"
eval_arm(){
  local exp=$1 id=$2 adapter=$3
  local eout=$DATA/runs/ablation_eval/$id
  rm -rf "$eout"
  pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true
  (cd "$REPO" && PYTHONPATH=$REPO/stubs:$ROOT/UAV-Flow/OpenVLA-UAV:. nohup $SERVE_PY scripts/serve_openvla_actions.py --vla-path "$DATA/openvla-uav" --adapter "$adapter" --unnorm-key sim --prompt-style openvla_uav --absolute-output --batch-size 1 --port 8010 >> "$LOG" 2>&1 &)
  for i in $(seq 1 120); do ss -ltn | grep -q ":8010" && break; sleep 1; done
  (cd "$REPO" && PYTHONUNBUFFERED=1 PYTHONPATH=$REPO:$ROOT/UAV-Flow/UAV-Flow-Eval $EVAL_PY scripts/eval_uavflow_test_5090.py --test-dir $ROOT/UAV-Flow/UAV-Flow-Eval/test_jsons --out "$eout" --workers 1 --retries 2 --rollouts 1 --limit 50 --proxy http://127.0.0.1:5007/predict >> "$LOG" 2>&1)
  pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true; pkill -9 -x Collection 2>/dev/null || true
  python3 - "$eout/summary.json" "$CSV" "$exp" "$id" "0.2" "2.0" "0.01" "1e-5" "$adapter" <<"PY"
import json,sys
s=json.load(open(sys.argv[1])); row=[sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],s["tasks"],s["success_rate"]*s["tasks"],s["success_rate"],s["mean_ndtW"],sys.argv[9]]
with open(sys.argv[2],"a") as f: f.write(",".join(map(str,row))+"\n")
print("EVAL",row)
PY
}
for spec in "5x30 5 30 online_sched_5x30" "30x5 30 5 online_sched_30x5"; do
  set -- $spec; name=$1; rounds=$2; per=$3; tag=$4
  echo "=== schedule $name ===" | tee -a "$LOG"
  env TAG=$tag MANIFEST=$DATA/manifest_150.json N_ROUNDS=$rounds PER=$per bash "$ROOT/run_online_eggrpo.sh" >> "$LOG" 2>&1
  adapter=$DATA/runs/${tag}_adapter_r$(printf %02d "$rounds")
  if [ -f "$adapter/adapter_model.safetensors" ]; then
    eval_arm 8 "$tag" "$adapter"
  else
    echo "SCHEDULE_FAIL $name" | tee -a "$LOG"
  fi
done
echo SCHEDULES_DONE | tee -a "$LOG"
