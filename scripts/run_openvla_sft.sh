#!/usr/bin/env bash
# Official OpenVLA LoRA-SFT on the exported UAV RLDS dataset.
set -euo pipefail
OPENVLA_DIR="${OPENVLA_DIR:-/workspace/openvla}"
RLDS_DIR="${RLDS_DIR:-/workspace/rlds}"
RUN_DIR="${RUN_DIR:-/workspace/runs/openvla_sft}"
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
GPU_COUNT="${GPU_COUNT:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$OPENVLA_DIR"
python3 "$SCRIPT_DIR/../integrations/openvla/patch_lora_config.py" "$OPENVLA_DIR"
torchrun --standalone --nnodes 1 --nproc-per-node "$GPU_COUNT" vla-scripts/finetune.py \
  --vla_path "$VLA_PATH" \
  --data_root_dir "$RLDS_DIR" \
  --dataset_name uav_flow_sim \
  --run_root_dir "$RUN_DIR" \
  --adapter_tmp_dir "$RUN_DIR/adapter-tmp" \
  --lora_rank 32 \
  --batch_size "${BATCH_SIZE:-8}" \
  --max_steps "$MAX_STEPS" \
  --grad_accumulation_steps "${GRAD_ACCUM:-2}" \
  --learning_rate 2e-4 \
  --image_aug False \
  --save_steps "${SAVE_STEPS:-1000}" \
  --save_latest_checkpoint_only True
