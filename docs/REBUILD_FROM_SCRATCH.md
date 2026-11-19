# DroneVL native UAV-Flow rebuild from scratch

## 1. Pod and storage

- A40 44 GB, 250 GB container disk
- optional network volume mounted at `/results`

## 2. Repositories

```bash
cd /workspace
git clone --branch feature/uavflow-sim-native \
  https://github.com/fern-robotics/DroneVLA.git DroneVLA-uavflow
git clone --depth 1 https://github.com/buaa-colalab/UAV-Flow.git UAV-Flow
```

## 3. Data and checkpoint

```bash
cd /workspace
/workspace/venv311/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("wangxiangyu0814/OpenVLA-UAV", local_dir="/workspace/openvla-uav")
snapshot_download("wangxiangyu0814/UAV-Flow-Sim", repo_type="dataset",
                  local_dir="/workspace/uavflow_sim_full")
PY
```

## 4. Linux Unreal scene

```bash
cd /workspace
wget -c -O Collection_v4_LinuxNoEditor.zip \
  "https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE4/resolve/master/Collection_v4_LinuxNoEditor.zip"
mkdir -p /workspace/UnrealEnv
unzip -q Collection_v4_LinuxNoEditor.zip -d /workspace/UnrealEnv
useradd -m -s /bin/bash ue4
chown -R ue4:ue4 /workspace/UnrealEnv
```

Set the engine path:

```bash
python - <<'PY'
import json
p="/workspace/UAV-Flow/UAV-Flow-Eval/gym_unrealcv/envs/setting/Track/DowntownWest.json"
d=json.load(open(p)); d["env_bin"]="/workspace/UnrealEnv/Collection_v4_LinuxNoEditor/Collection/Binaries/Linux/Collection"
json.dump(d,open(p,"w"),indent=4)
PY
```

## 5. Python environments

OpenVLA:

```bash
uv venv --python 3.11 /workspace/venv311
uv pip install --python /workspace/venv311/bin/python --index-url \
  https://download.pytorch.org/whl/cu121 torch==2.2.0 torchvision==0.17.0
uv pip install --python /workspace/venv311/bin/python \
  transformers==4.40.2 peft==0.10.0 accelerate==0.30.0 \
  bitsandbytes==0.43.3 draccus tensorflow pillow flask safetensors pyarrow
```

Simulator:

```bash
uv venv --python 3.11 /workspace/venv-uavflow
uv pip install --python /workspace/venv-uavflow/bin/python \
  -e /workspace/UAV-Flow/UAV-Flow-Eval setuptools==68.2.2 \
  unrealcv==1.1.5 opencv-python simple_pid
```

## 6. Tasks and expert cache

```bash
/workspace/venv311/bin/python /workspace/DroneVLA-uavflow/scripts/build_uavflow_tasks.py \
  --data /workspace/uavflow_sim_full --out /workspace/uavflow_train_tasks
/workspace/venv311/bin/python /workspace/DroneVLA-uavflow/scripts/sample_uavflow_tasks.py \
  --train-dir /workspace/uavflow_train_tasks \
  --test-dir /workspace/UAV-Flow/UAV-Flow-Eval/test_jsons \
  --out /workspace/runs/b0_eggrpo/manifest_300.json --count 300 --seed 7
/workspace/venv311/bin/python /workspace/DroneVLA-uavflow/scripts/build_expert_cache.py \
  --data /workspace/uavflow_sim_full \
  --tasks /workspace/runs/b0_eggrpo/manifest_300.json \
  --out /workspace/runs/b0_eggrpo/expert_cache --limit 300
```

## 7. Server and display

```bash
nohup Xvfb :99 -screen 0 1280x720x24 -ac > /workspace/xvfb.log 2>&1 &
cd /workspace/DroneVLA-uavflow
PYTHONPATH=/workspace/UAV-Flow/OpenVLA-UAV:. \
  nohup /workspace/venv311/bin/python scripts/serve_openvla_actions.py \
  --vla-path /workspace/openvla-uav --prompt-style openvla_uav \
  --absolute-output --batch-size 4 --port 8000 > /workspace/eg_server.log 2>&1 &
PYTHONPATH=. nohup /workspace/venv-uavflow/bin/python \
  integrations/uavflow_sim/policy_proxy.py --upstream http://127.0.0.1:8000 \
  --port 5007 > /workspace/eg_proxy.log 2>&1 &
```

## 8. Collection

Eight policy rollouts per task: two waves of four engines.

```bash
/workspace/venv-uavflow/bin/python \
  /workspace/DroneVLA-uavflow/scripts/collect_uavflow_parallel.py \
  --manifest /workspace/runs/b0_eggrpo/manifest_300.json \
  --out /workspace/runs/b0_eggrpo --workers 4 --waves 2 --timeout 1800
```

## 9. EG-GRPO training

```bash
cd /workspace/DroneVLA-uavflow
PYTHONPATH=. /workspace/venv311/bin/python scripts/train_b0_eggrpo.py \
  --groups /workspace/runs/b0_eggrpo/groups.json \
  --expert-cache /workspace/runs/b0_eggrpo/expert_cache \
  --out /workspace/runs/b0_eggrpo/adapter --max-groups 300 --lr 1e-5
```

## 10. Evaluation

Serve the adapter on `:8000`, then:

```bash
/workspace/venv-uavflow/bin/python \
  /workspace/DroneVLA-uavflow/scripts/eval_uavflow_test.py \
  --test-dir /workspace/UAV-Flow/UAV-Flow-Eval/test_jsons \
  --out /workspace/runs/eval_b0_eggrpo \
  --workers 2 --retries 2 --rollouts 1 --limit 244 --seed 7
```

## Troubleshooting: `ue4` cannot execute venv Python

If the venv interpreter lives under `/root/.local/...`, `ue4` may fail with
`Permission denied`. Relocate the interpreter into `/workspace`:

```bash
cp -a /root/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu /workspace/python311
ln -sf /workspace/python311/bin/python3.11 /workspace/venv-uavflow/bin/python
ln -sf /workspace/python311/bin/python3.11 /workspace/venv-uavflow/bin/python3
sed -i 's#^home = .*#home = /workspace/python311/bin#' /workspace/venv-uavflow/pyvenv.cfg
runuser -u ue4 -- /workspace/venv-uavflow/bin/python -c 'import gym, unrealcv; print("ok")'
```
