#!/usr/bin/env bash
# Logged live AirSim evaluation: absolute XYZ, VLA chunk, tokens/logprobs, MP4.
set -euo pipefail
OUT="${OUT:-/workspace/runs/logged_eval}"
VIDEO="${VIDEO:-$OUT/episode.mp4}"
EPISODE="${EPISODE:-id-approach-01}"
POLICY_URL="${POLICY_URL:-http://127.0.0.1:8000}"
PYTHON="${PYTHON:-/workspace/venv-airsim/bin/python}"
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
PYTHONPATH=. "$PYTHON" -m scripts.eval.evaluate_vla \
  --env airsim --airsim-ip 127.0.0.1 --airsim-port 41451 \
  --policy http --policy-url "$POLICY_URL" \
  --capture-image --video "$VIDEO" \
  --episode-id "$EPISODE" --out "$OUT"
echo "log:   $OUT/episodes/$EPISODE.jsonl"
echo "video: $VIDEO"
