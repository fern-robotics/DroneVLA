#!/usr/bin/env bash
# One-GPU verification: OpenVLA RLDS + one-step official LoRA-SFT.
# Verified on A100 80GB; OpenVLA needs Python 3.10/3.11, not 3.12.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl ffmpeg >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh >/tmp/uv-install.log 2>&1
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv venv /workspace/venv311 --python 3.11
P=/workspace/venv311/bin/python
cd /workspace
[ -d openvla ] || git clone --depth 1 https://github.com/openvla/openvla.git openvla
uv pip install --python "$P" -e /workspace/openvla
uv pip install --python "$P" "numpy<2" "tensorflow-metadata==1.14.0" "protobuf==3.20.3" "wandb==0.16.6" pyarrow pillow
uv pip install --python "$P" --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"
uv pip install --python "$P" --no-deps tensorflow-graphics
mkdir -p /workspace/data
$P /workspace/scripts/build_dataset_online.py --shards 0 --out /workspace/data/uav_smoke --fraction 0.20 --formats jsonl
cp /workspace/scripts/dataset/rlds_builder.py /workspace/data/uav_smoke/uav_flow_sim_dataset_builder.py
cd /workspace/data/uav_smoke
UAV_EXPORT_DIR=/workspace/data/uav_smoke $P - <<'PY'
import importlib.util, tensorflow_datasets as tfds
spec = importlib.util.spec_from_file_location("uav_flow_sim_dataset_builder", "uav_flow_sim_dataset_builder.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
b = mod.Builder(data_dir="/workspace/data/uav_smoke/rlds")
b.download_and_prepare(download_dir="/workspace/data/uav_smoke/rlds")
print("RLDS", b.info.splits)
PY
$P /workspace/integrations/openvla/patch_openvla_uav.py /workspace/openvla
mkdir -p /workspace/runs/openvla_smoke
cd /workspace/openvla
WANDB_MODE=disabled MASTER_ADDR=127.0.0.1 MASTER_PORT=29525 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
  $P vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /workspace/data/uav_smoke/rlds/uav_flow_sim \
  --dataset_name uav_flow_sim \
  --run_root_dir /workspace/runs/openvla_smoke \
  --adapter_tmp_dir /workspace/runs/openvla_smoke/adapter-tmp \
  --lora_rank 32 --batch_size 1 --grad_accumulation_steps 1 \
  --max_steps 1 --save_steps 1 --learning_rate 5e-4 --image_aug False \
  --shuffle_buffer_size 100 --save_latest_checkpoint_only True \
  --wandb_project dronevl --wandb_entity unused
echo GPU_VERIFY_OK
