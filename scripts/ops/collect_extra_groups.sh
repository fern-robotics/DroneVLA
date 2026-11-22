#!/usr/bin/env bash
# Collect extra EG-GRPO groups and merge them with an existing groups file.
# Usage: collect_extra_groups.sh <adapter> <manifest> <out_dir> <base_groups> <merged_out>
set -uo pipefail
ROOT=/home/digilab/ws/skripsi-ryan-adidaru
REPO=$ROOT/repo
DATA=/mnt/data/skripsi-ryan-adidaru
EVAL_PY=$ROOT/envs/uavflow/bin/python
SERVE_PY=$ROOT/envs/openvla/bin/python
ADAPTER=$1; MANIFEST=$2; OUT=$3; BASE_GROUPS=$4; MERGED=$5
LOG=$DATA/collect_extra.log
mkdir -p "$OUT"
pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true
pkill -9 -x Collection 2>/dev/null || true
if ! ss -ltn | grep -q ":5007"; then
  (cd "$REPO" && PYTHONPATH=. nohup $EVAL_PY integrations/uavflow_sim/policy_proxy.py --upstream http://127.0.0.1:8010 --port 5007 >> "$LOG" 2>&1 &)
  sleep 2
fi
(cd "$REPO" && PYTHONPATH=$REPO/stubs:$ROOT/UAV-Flow/OpenVLA-UAV:. nohup $SERVE_PY scripts/serve_openvla_actions.py --vla-path "$DATA/openvla-uav" --adapter "$ADAPTER" --unnorm-key sim --prompt-style openvla_uav --absolute-output --batch-size 1 --port 8010 >> "$LOG" 2>&1 &)
for i in $(seq 1 120); do ss -ltn | grep -q ":8010" && break; sleep 1; done
if ! ss -ltn | grep -q ":8010"; then echo SERVER_FAIL | tee -a "$LOG"; exit 1; fi
(cd "$REPO" && $EVAL_PY scripts/collect_uavflow_queue.py --manifest "$MANIFEST" --out "$OUT" --workers 1 --retries 2 --timeout 1800 --python "$EVAL_PY" --runner "$REPO/scripts/eval/uavflow_native_policy.py" --root "$ROOT/UAV-Flow/UAV-Flow-Eval" --unreal-env "$DATA/UnrealEnv" >> "$LOG" 2>&1)
pkill -f "[s]erve_openvla_actions.py.*--port 8010" 2>/dev/null || true
pkill -9 -x Collection 2>/dev/null || true
python3 - "$BASE_GROUPS" "$OUT/groups.json" "$MERGED" <<"PY"
import json,sys
base=json.load(open(sys.argv[1])); extra=json.load(open(sys.argv[2]))
out=[]; seen=set()
for g in base+extra:
    task=g["task"]
    if task in seen: continue
    seen.add(task); g=dict(g); g["group"]=len(out); out.append(g)
json.dump(out,open(sys.argv[3],"w"),indent=2)
print(json.dumps({"base":len(base),"extra":len(extra),"merged":len(out)}))
PY
