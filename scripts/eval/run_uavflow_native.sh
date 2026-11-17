#!/usr/bin/env bash
# Native UAV-Flow-Sim / UnrealZoo run with screen recording.
set -euo pipefail
UAVFLOW_DIR="${UAVFLOW_DIR:-/workspace/UAV-Flow/UAV-Flow-Eval}"
SCENE_ROOT="${SCENE_ROOT:-/workspace/unrealzoo}"
OUT="${OUT:-/workspace/runs/uavflow_native}"
SERVER_PORT="${SERVER_PORT:-5007}"
MAX_STEPS="${MAX_STEPS:-100}"
DISPLAY_NUM="${DISPLAY_NUM:-:99}"
mkdir -p "$OUT"
EXE=$(find "$SCENE_ROOT" -type f -path '*/Binaries/Linux/*' -perm -111 2>/dev/null | head -1)
if [ -z "$EXE" ]; then echo "no Linux UnrealZoo executable under $SCENE_ROOT"; exit 1; fi
echo "scene exe: $EXE" | tee "$OUT/scene_exe.txt"
python3 - "$UAVFLOW_DIR/gym_unrealcv/envs/setting/Track/DowntownWest.json" "$EXE" <<'PY'
import json, sys
path, exe = sys.argv[1], sys.argv[2]
d = json.load(open(path)); d["env_bin"] = exe
json.dump(d, open(path, "w"), indent=4)
print("patched", path, "->", exe)
PY
export DISPLAY="$DISPLAY_NUM"
ffmpeg -y -loglevel error -f x11grab -video_size 1280x720 -framerate 10 -i "$DISPLAY" \
  -t 1200 -c:v libx264 -pix_fmt yuv420p "$OUT/native.mp4" > "$OUT/ffmpeg.log" 2>&1 &
FF=$!
cd "$UAVFLOW_DIR"
set +e
timeout 1800 python3 batch_run_act_all.py \
  --env_id UnrealTrack-DowntownWest-ContinuousColor-v0 \
  --server_port "$SERVER_PORT" \
  --json_folder ./test_jsons \
  --images_dir "$OUT/images" \
  --max_steps "$MAX_STEPS" > "$OUT/run.log" 2>&1
STATUS=$?
set -e
kill "$FF" 2>/dev/null || true
wait "$FF" 2>/dev/null || true
echo "status=$STATUS"
exit "$STATUS"
