#!/usr/bin/env bash
# Serve a trained UAV pi0-FAST LoRA checkpoint with token log-probs.
# Usage: scripts/run_uav_pi0_server.sh /path/to/checkpoint/599
set -euo pipefail
OPENPI_DIR="${OPENPI_DIR:-/workspace/openpi}"
CHECKPOINT="${1:?pass the checkpoint directory, e.g. .../uav_pilot/10000}"
CONFIG="${CONFIG:-pi0_fast_uav_low_mem}"
PORT="${PORT:-8000}"
export PATH="$HOME/.local/bin:$PATH"
cd "$OPENPI_DIR"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
exec uv run python "$SCRIPT_DIR/serve_pi0_logprobs.py" \
  --config "$CONFIG" \
  --dir "$CHECKPOINT" \
  --port "$PORT"
