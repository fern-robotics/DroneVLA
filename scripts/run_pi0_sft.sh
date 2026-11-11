#!/usr/bin/env bash
# Official OpenPI pi0-FAST LoRA-SFT on the exported UAV LeRobot split.
set -euo pipefail
OPENPI_DIR="${OPENPI_DIR:-/workspace/openpi}"
REPO_ID="${UAV_REPO_ID:-uav_flow_sim_train}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/lerobot}"
export UAV_REPO_ID="$REPO_ID"
cd "$OPENPI_DIR"
uv run scripts/compute_norm_stats.py --config-name pi0_fast_uav_low_mem
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM:-0.9}" uv run scripts/train.py pi0_fast_uav_low_mem \
  --exp-name "${EXP_NAME:-uav_sft}" --overwrite
