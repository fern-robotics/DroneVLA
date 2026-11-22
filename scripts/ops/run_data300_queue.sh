#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/digilab/ws/skripsi-ryan-adidaru
REPO=$ROOT/repo
DATA=/mnt/data/skripsi-ryan-adidaru
TRAIN_PY=$ROOT/envs/openvla/bin/python
EVAL_PY=$ROOT/envs/uavflow/bin/python
SERVE_PY=$ROOT/envs/openvla/bin/python
LOG=$DATA/data300_queue.log
CSV=$DATA/runs/ablation_results.csv
mkdir -p "$DATA/runs"
: > "$LOG"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo DATA300_FAIL >> "$LOG"; fi' EXIT
pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true
pkill -9 -x Collection 2>/dev/null || true
# 1. Collect the extra offline groups with the offline150 policy.
if [ ! -f "$DATA/runs/offline_extra150/groups.json" ]; then
  bash "$ROOT/collect_extra_groups.sh" "$DATA/runs/adapter_offline150" "$DATA/manifest_extra150.json" "$DATA/runs/offline_extra150" "$DATA/groups_offline_150.json" "$DATA/groups_offline_300.json" >> "$LOG" 2>&1
fi
# 1b. If the merge did not happen, do it explicitly.
python3 - "$DATA/groups_offline_150.json" "$DATA/runs/offline_extra150/groups.json" "$DATA/groups_offline_300.json" <<"PY"
import json,sys
base=json.load(open(sys.argv[1])); extra=json.load(open(sys.argv[2]))
out=[]; seen=set()
for g in base+extra:
    if g["task"] in seen: continue
    seen.add(g["task"]); g=dict(g); g["group"]=len(out); out.append(g)
json.dump(out,open(sys.argv[3],"w"),indent=2); print(json.dumps({"merged":len(out)}))
if len(out) < 300: raise SystemExit("offline300 needs 300 groups")
PY
if [ $? -ne 0 ]; then echo OFFLINE300_GROUPS_FAIL | tee -a "$LOG"; exit 1; fi
# 2. Offline300 training.
if [ ! -f "$DATA/runs/adapter_offline300/adapter_model.safetensors" ]; then
  (cd "$REPO" && PYTHONPATH=$ROOT/UAV-Flow/OpenVLA-UAV:. $TRAIN_PY scripts/train_b0_eggrpo.py --groups "$DATA/groups_offline_300.json" --base "$DATA/openvla-uav" --out "$DATA/runs/adapter_offline300" --max-groups 300 --save-every 300 --expert-cache "$DATA/expert_cache_300" --expert-bc-weight 0.2 --expert-bc-steps 4 --expert-reward 2.0 --beta 0.01 --lr 1e-5 --recompute-old >> "$LOG" 2>&1)
fi
# 3. Online groups: merge the existing 15 rounds, then collect 150 extra with r15.
python3 - "$DATA/runs/online150" "$DATA/groups_online150.json" <<"PY"
import json,glob,sys
out=[]; seen=set()
for p in sorted(glob.glob(sys.argv[1]+"/roundr*/groups.json")):
    for g in json.load(open(p)):
        if g["task"] in seen: continue
        seen.add(g["task"]); g=dict(g); g["group"]=len(out); out.append(g)
json.dump(out,open(sys.argv[2],"w"),indent=2); print(json.dumps({"online150_groups":len(out)}))
PY
if [ ! -f "$DATA/runs/online_extra150/groups.json" ]; then
  bash "$ROOT/collect_extra_groups.sh" "$DATA/runs/online150_adapter_r15" "$DATA/manifest_extra150.json" "$DATA/runs/online_extra150" "$DATA/groups_online150.json" "$DATA/groups_online_300.json" >> "$LOG" 2>&1
fi
python3 - "$DATA/groups_online_300.json" <<"PY"
import json,sys
n=len(json.load(open(sys.argv[1])))
print(json.dumps({"online300_groups":n}))
if n < 300: raise SystemExit("online300 needs 300 groups")
PY
if [ $? -ne 0 ]; then echo ONLINE300_GROUPS_FAIL | tee -a "$LOG"; exit 1; fi
# 4. Online300 training from online r15.
if [ ! -f "$DATA/runs/adapter_online300/adapter_model.safetensors" ]; then
  (cd "$REPO" && PYTHONPATH=$ROOT/UAV-Flow/OpenVLA-UAV:. $TRAIN_PY scripts/train_b0_eggrpo.py --groups "$DATA/groups_online_300.json" --base "$DATA/openvla-uav" --init-adapter "$DATA/runs/online150_adapter_r15" --out "$DATA/runs/adapter_online300" --max-groups 300 --save-every 300 --expert-cache "$DATA/expert_cache_300" --expert-bc-weight 0.2 --expert-bc-steps 4 --expert-reward 2.0 --beta 0.01 --lr 1e-5 --recompute-old >> "$LOG" 2>&1)
fi

eval_adapter(){
  local exp=$1 id=$2 adapter=$3 bc=$4 anchor=$5 beta=$6 lr=$7
  local eout=$DATA/runs/ablation_eval/$id
  rm -rf "$eout"
  pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true
  (cd "$REPO" && PYTHONPATH=$REPO/stubs:$ROOT/UAV-Flow/OpenVLA-UAV:. nohup $SERVE_PY scripts/serve_openvla_actions.py --vla-path "$DATA/openvla-uav" --adapter "$adapter" --unnorm-key sim --prompt-style openvla_uav --absolute-output --batch-size 1 --port 8010 >> "$LOG" 2>&1 &)
  for i in $(seq 1 120); do ss -ltn | grep -q ":8010" && break; sleep 1; done
  (cd "$REPO" && PYTHONUNBUFFERED=1 PYTHONPATH=$REPO:$ROOT/UAV-Flow/UAV-Flow-Eval $EVAL_PY scripts/eval_uavflow_test_5090.py --test-dir $ROOT/UAV-Flow/UAV-Flow-Eval/test_jsons --out "$eout" --workers 1 --retries 2 --rollouts 1 --limit 50 --proxy http://127.0.0.1:5007/predict >> "$LOG" 2>&1)
  pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true; pkill -9 -x Collection 2>/dev/null || true
  python3 - "$eout/summary.json" "$CSV" "$exp" "$id" "$bc" "$anchor" "$beta" "$lr" "$adapter" <<"PY2"
import json,sys
s=json.load(open(sys.argv[1])); row=[sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],s["tasks"],s["success_rate"]*s["tasks"],s["success_rate"],s["mean_ndtW"],sys.argv[9]]
with open(sys.argv[2],"a") as f: f.write(",".join(map(str,row))+"\n")
print("EVAL",row)
PY2
}
eval_adapter 7 offline300 "$DATA/runs/adapter_offline300" 0.2 2.0 0.01 1e-5
eval_adapter 7 online300 "$DATA/runs/adapter_online300" 0.2 2.0 0.01 1e-5

echo DATA300_DONE | tee -a "$LOG"
