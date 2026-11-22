#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/digilab/ws/skripsi-ryan-adidaru
REPO=$ROOT/repo
DATA=/mnt/data/skripsi-ryan-adidaru
TRAIN_PY=$ROOT/envs/openvla/bin/python
EVAL_PY=$ROOT/envs/uavflow/bin/python
SERVE_PY=$ROOT/envs/openvla/bin/python
BASE=$DATA/openvla-uav
GROUP_FILE=$DATA/groups_offline_150.json
CACHE=$DATA/expert_cache_150
OUTDIR=$DATA/runs/ablation
EVALDIR=$DATA/runs/ablation_eval
LOG=$DATA/ablation_queue.log
CSV=$DATA/runs/ablation_results.csv
mkdir -p "$OUTDIR" "$EVALDIR"
: > "$LOG"
echo "experiment,id,bc,anchor,beta,lr,tasks,success,success_rate,ndtW,adapter" > "$CSV"
pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true
pkill -9 -x Collection 2>/dev/null || true
if ! ss -ltn | grep -q ":5007"; then
  (cd "$REPO" && PYTHONPATH=. nohup $EVAL_PY integrations/uavflow_sim/policy_proxy.py --upstream http://127.0.0.1:8010 --port 5007 >> "$LOG" 2>&1 &)
  sleep 2
fi
start_server(){
  pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true
  (cd "$REPO" && PYTHONPATH=$REPO/stubs:$ROOT/UAV-Flow/OpenVLA-UAV:. nohup $SERVE_PY scripts/serve_openvla_actions.py --vla-path "$BASE" --adapter "$1" --unnorm-key sim --prompt-style openvla_uav --absolute-output --batch-size 1 --port 8010 >> "$LOG" 2>&1 &)
  for i in $(seq 1 120); do ss -ltn | grep -q ":8010" && return 0; sleep 1; done
  return 1
}
stop_all(){ pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true; pkill -9 -x Collection 2>/dev/null || true; }
eval_arm(){
  local id=$1 adapter=$2 exp=$3 bc=$4 anchor=$5 beta=$6 lr=$7
  local eout=$EVALDIR/$id
  rm -rf "$eout"
  if ! start_server "$adapter"; then echo "server_fail $id" | tee -a "$LOG"; return 1; fi
  (cd "$REPO" && PYTHONUNBUFFERED=1 PYTHONPATH=$REPO:$ROOT/UAV-Flow/UAV-Flow-Eval $EVAL_PY scripts/eval_uavflow_test_5090.py --test-dir $ROOT/UAV-Flow/UAV-Flow-Eval/test_jsons --out "$eout" --workers 1 --retries 2 --rollouts 1 --limit 50 --proxy http://127.0.0.1:5007/predict >> "$LOG" 2>&1)
  stop_all
  python3 - "$eout/summary.json" "$CSV" "$exp" "$id" "$bc" "$anchor" "$beta" "$lr" "$adapter" <<"PY"
import json,sys
s=json.load(open(sys.argv[1])); row=[sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],s["tasks"],s["success_rate"]*s["tasks"],s["success_rate"],s["mean_ndtW"],sys.argv[9]]
with open(sys.argv[2],"a") as f: f.write(",".join(map(str,row))+"\n")
print("EVAL",row)
PY
}
run_offline(){
  local exp=$1 id=$2 bc=$3 anchor=$4 beta=$5 lr=$6
  local out=$OUTDIR/$id
  echo "=== $id bc=$bc anchor=$anchor beta=$beta lr=$lr ===" | tee -a "$LOG"
  if [ ! -f "$out/adapter_model.safetensors" ]; then
    (cd "$REPO" && PYTHONPATH=$ROOT/UAV-Flow/OpenVLA-UAV:. $TRAIN_PY scripts/train_b0_eggrpo.py --groups "$GROUP_FILE" --base "$BASE" --out "$out" --max-groups 150 --save-every 150 --expert-cache "$CACHE" --expert-bc-weight "$bc" --expert-bc-steps 4 --expert-reward "$anchor" --beta "$beta" --lr "$lr" --recompute-old >> "$LOG" 2>&1)
  fi
  if [ ! -f "$out/adapter_model.safetensors" ]; then echo "TRAIN_FAIL $id" | tee -a "$LOG"; return 1; fi
  eval_arm "$id" "$out" "$exp" "$bc" "$anchor" "$beta" "$lr"
}
# Baseline BC=0.2 is the existing offline150 adapter; reuse its full-244 first 50 instead of retraining.
python3 - "$DATA/runs/eval_offline150_244_gpu/results" "$CSV" <<"PY"
import json,glob,os,sys,statistics
files=sorted(glob.glob(os.path.join(sys.argv[1],"*.json")))[:50]
r=[json.load(open(p)) for p in files]
row=["3","baseline_bc02","0.2","2.0","0.01","1e-5",len(r),sum(x["success"] for x in r),sum(x["success"] for x in r)/len(r),statistics.mean(x["ndtW"] for x in r),"/mnt/data/skripsi-ryan-adidaru/runs/adapter_offline150"]
with open(sys.argv[2],"a") as f: f.write(",".join(map(str,row))+"\n")
print("EVAL",row)
PY
run_offline 3 ab_bc0 0.0 2.0 0.01 1e-5
run_offline 3 ab_bc05 0.5 2.0 0.01 1e-5
run_offline 4 ab_anchor_off 0.2 0.0 0.01 1e-5
run_offline 5 ab_kl0 0.2 2.0 0.0 1e-5
run_offline 6 ab_lr5e6 0.2 2.0 0.01 5e-6
run_offline 6 ab_lr2e5 0.2 2.0 0.01 2e-5
echo ABLATION_QUEUE_DONE | tee -a "$LOG"
