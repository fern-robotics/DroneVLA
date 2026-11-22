#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/digilab/ws/skripsi-ryan-adidaru
REPO=$ROOT/repo
DATA=/mnt/data/skripsi-ryan-adidaru
EVAL_PY=$ROOT/envs/uavflow/bin/python
SERVE_PY=$ROOT/envs/openvla/bin/python
LOG=$DATA/bc244_queue.log
CSV=$DATA/runs/bc244_results.csv
: > "$LOG"
echo "id,tasks,success,success_rate,ndtW,adapter" > "$CSV"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo BC244_FAIL >> "$LOG"; fi' EXIT
pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true
pkill -9 -x Collection 2>/dev/null || true
if ! ss -ltn | grep -q ":5007"; then
  (cd "$REPO" && PYTHONPATH=. nohup $EVAL_PY integrations/uavflow_sim/policy_proxy.py --upstream http://127.0.0.1:8010 --port 5007 >> "$LOG" 2>&1 &)
  sleep 2
fi
eval_one(){
  local id=$1 adapter=$2
  local eout=$DATA/runs/eval_${id}_244
  rm -rf "$eout"
  pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true
  (cd "$REPO" && PYTHONPATH=$REPO/stubs:$ROOT/UAV-Flow/OpenVLA-UAV:. nohup $SERVE_PY scripts/serve_openvla_actions.py --vla-path "$DATA/openvla-uav" --adapter "$adapter" --unnorm-key sim --prompt-style openvla_uav --absolute-output --batch-size 1 --port 8010 >> "$LOG" 2>&1 &)
  for i in $(seq 1 120); do ss -ltn | grep -q ":8010" && break; sleep 1; done
  if ! ss -ltn | grep -q ":8010"; then echo "SERVER_FAIL $id" | tee -a "$LOG"; return 1; fi
  (cd "$REPO" && PYTHONUNBUFFERED=1 PYTHONPATH=$REPO:$ROOT/UAV-Flow/UAV-Flow-Eval $EVAL_PY scripts/eval_uavflow_test_5090.py --test-dir $ROOT/UAV-Flow/UAV-Flow-Eval/test_jsons --out "$eout" --workers 1 --retries 2 --rollouts 1 --limit 244 --proxy http://127.0.0.1:5007/predict >> "$LOG" 2>&1)
  pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true; pkill -9 -x Collection 2>/dev/null || true
  python3 - "$eout/summary.json" "$CSV" "$id" "$adapter" <<"PY"
import json,sys
s=json.load(open(sys.argv[1])); row=[sys.argv[3],s["tasks"],s["success_rate"]*s["tasks"],s["success_rate"],s["mean_ndtW"],sys.argv[4]]
with open(sys.argv[2],"a") as f: f.write(",".join(map(str,row))+"\n")
print("FULL244",row)
PY
}
eval_one ab_bc0 "$DATA/runs/ablation/ab_bc0" && eval_one ab_bc05 "$DATA/runs/ablation/ab_bc05"
echo BC244_DONE | tee -a "$LOG"
