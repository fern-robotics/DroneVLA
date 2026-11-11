#!/usr/bin/env bash
# Run a real OpenVLA/pi0 HTTP policy in AirSimNH and record an MP4.
# Assumes AirSimNH is running and the policy server is already serving.
set -euo pipefail
POLICY_URL="${POLICY_URL:-http://127.0.0.1:8000}"
OUT="${OUT:-/workspace/runs/real_vla}"
VIDEO="${VIDEO:-$OUT/real_vla.mp4}"
EPISODE="${EPISODE:-id-approach-01}"
PYTHON="${PYTHON:-/workspace/venv-airsim/bin/python}"
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
PYTHONPATH=. "$PYTHON" -m scripts.eval.evaluate_vla \
  --env airsim --airsim-ip 127.0.0.1 --airsim-port 41451 \
  --policy http --policy-url "$POLICY_URL" \
  --capture-image --video "$VIDEO" \
  --episode-id "$EPISODE" --out "$OUT"
echo "video: $VIDEO"
