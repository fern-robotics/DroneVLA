#!/usr/bin/env bash
# One-GPU verification: pi0-FAST config, OpenVLA RLDS + 1-step LoRA-SFT.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl python3-venv ffmpeg >/dev/null 2>&1
python3 -m venv /workspace/venv
P=/workspace/venv/bin/python
$P -m pip install -q --upgrade pip
$P -m pip install -q pyarrow pillow numpy
cd /workspace
if [ ! -d openvla ]; then git clone --depth 1 https://github.com/openvla/openvla.git openvla; fi
$P -m pip install -q --no-deps -e /workspace/openvla
$P -m pip install -q accelerate draccus==0.8.0 einops huggingface_hub json-numpy jsonlines matplotlib peft==0.11.1 protobuf rich sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 transformers==4.40.1 wandb tensorflow-cpu tensorflow-datasets
$P -m pip install -q "dlimp @ git+https://github.com/moojink/dlimp_openvla" || echo "dlimp install failed (RLDS loader may need it)"
mkdir -p /workspace/data
$P /workspace/scripts/build_dataset_online.py --shards 0 --out /workspace/data/uav_smoke --fraction 0.20 --formats jsonl
cp /workspace/scripts/dataset/rlds_builder.py /workspace/data/uav_smoke/uav_flow_sim_dataset_builder.py
cd /workspace/data/uav_smoke
UAV_EXPORT_DIR=/workspace/data/uav_smoke $P - <<'PY'
import importlib.util, tensorflow_datasets as tfds
spec=importlib.util.spec_from_file_location("uav_flow_sim_dataset_builder","uav_flow_sim_dataset_builder.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
b=mod.Builder(data_dir="/workspace/data/uav_smoke/rlds")
b.download_and_prepare(download_dir="/workspace/data/uav_smoke/rlds")
print("RLDS", b.info.splits)
PY
$P /workspace/integrations/openvla/patch_openvla_uav.py /workspace/openvla
mkdir -p /workspace/runs/openvla_smoke
cd /workspace/openvla
WANDB_MODE=disabled $P -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /workspace/data/uav_smoke/rlds \
  --dataset_name uav_flow_sim \
  --run_root_dir /workspace/runs/openvla_smoke \
  --adapter_tmp_dir /workspace/runs/openvla_smoke/adapter-tmp \
  --lora_rank 32 --batch_size 1 --grad_accumulation_steps 1 \
  --max_steps 1 --save_steps 1 --learning_rate 5e-4 --image_aug False \
  --shuffle_buffer_size 100 --save_latest_checkpoint_only True \
  --wandb_project dronevl --wandb_entity unused
echo GPU_VERIFY_OK
