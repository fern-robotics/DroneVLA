#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/digilab/ws/skripsi-ryan-adidaru
REPO=$ROOT/repo
DATA=/mnt/data/skripsi-ryan-adidaru
TAG=${TAG:-online150}
N_ROUNDS=${N_ROUNDS:-15}
PER=${PER:-10}
EVAL_PY=$ROOT/envs/uavflow/bin/python
TRAIN_PY=$ROOT/envs/openvla/bin/python
SERVE_PY=$ROOT/envs/openvla/bin/python
BASE=$DATA/openvla-uav
MANIFEST=$DATA/manifest_150.json
ADAPTER=$DATA/runs/adapter_offline150
OUT=$DATA/runs/$TAG
LOG=$DATA/${TAG}.log
mkdir -p "$OUT"
: > "$LOG"
python3 - "$MANIFEST" "$OUT" "$N_ROUNDS" "$PER" <<"PY"
import json,sys,pathlib
m=json.load(open(sys.argv[1]))["tasks"]; out=pathlib.Path(sys.argv[2]); n=int(sys.argv[3]); per=int(sys.argv[4])
for r in range(n):
    tasks=m[r*per:(r+1)*per]
    (out/f"manifest_r{r+1:02d}.json").write_text(json.dumps({"tasks":tasks},indent=2))
    print(len(tasks), out/f"manifest_r{r+1:02d}.json")
PY
SERVER_PID=""
cleanup(){
  [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true
  pkill -9 -x Collection 2>/dev/null || true
}
trap cleanup EXIT
if ! ss -ltn | grep -q ":5007"; then
  (cd "$REPO" && PYTHONPATH=. nohup $EVAL_PY integrations/uavflow_sim/policy_proxy.py --upstream http://127.0.0.1:8010 --port 5007 >> "$LOG" 2>&1 &)
  sleep 2
fi
for r in $(seq 1 "$N_ROUNDS"); do
  RR=$(printf "r%02d" "$r")
  echo "=== ROUND $r adapter=$ADAPTER ===" | tee -a "$LOG"
  pkill -f "serve_openvla_actions.py.*--port 8010" 2>/dev/null || true
  (cd "$REPO" && PYTHONPATH=$REPO/stubs:$ROOT/UAV-Flow/OpenVLA-UAV:. nohup $SERVE_PY scripts/serve_openvla_actions.py --vla-path "$BASE" --adapter "$ADAPTER" --unnorm-key sim --prompt-style openvla_uav --absolute-output --batch-size 1 --port 8010 >> "$LOG" 2>&1 & echo $! > /tmp/online_server.pid)
  SERVER_PID=$(cat /tmp/online_server.pid)
  for i in $(seq 1 120); do ss -ltn | grep -q ":8010" && break; sleep 1; done
  if ! ss -ltn | grep -q ":8010"; then echo "server failed" | tee -a "$LOG"; exit 1; fi
  RDIR=$OUT/round$RR
  mkdir -p "$RDIR"
  (cd "$REPO" && $EVAL_PY scripts/collect_uavflow_queue.py --manifest "$OUT/manifest_$RR.json" --out "$RDIR" --workers 1 --retries 2 --timeout 1800 --python "$EVAL_PY" --runner "$REPO/scripts/eval/uavflow_native_policy.py" --root "$ROOT/UAV-Flow/UAV-Flow-Eval" --unreal-env "$DATA/UnrealEnv" >> "$LOG" 2>&1)
  kill "$SERVER_PID" 2>/dev/null || true; pkill -f "serve_openvla_actions.py.*--port 8010" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""
  pkill -9 -x Collection 2>/dev/null || true
  G=$(python3 -c "import json;print(len(json.load(open(\"$RDIR/groups.json\"))))" 2>/dev/null || echo 0)
  echo "collected $G groups" | tee -a "$LOG"
  if [ "$G" -lt 1 ]; then echo "no groups" | tee -a "$LOG"; exit 1; fi
  NEW=$DATA/runs/${TAG}_adapter_$RR
  (cd "$REPO" && PYTHONPATH=$ROOT/UAV-Flow/OpenVLA-UAV:. $TRAIN_PY scripts/train_b0_eggrpo.py --groups "$RDIR/groups.json" --base "$BASE" --init-adapter "$ADAPTER" --expert-cache "$DATA/expert_cache_150" --out "$NEW" --max-groups "$G" --save-every "$G" --expert-bc-weight 0.2 --expert-bc-steps 4 --recompute-old >> "$LOG" 2>&1)
  if [ ! -f "$NEW/adapter_model.safetensors" ]; then echo "training failed $RR" | tee -a "$LOG"; exit 1; fi
  ADAPTER=$NEW
  echo "round $r done -> $ADAPTER" | tee -a "$LOG"
done
echo "ONLINE_DONE $ADAPTER" | tee -a "$LOG"
