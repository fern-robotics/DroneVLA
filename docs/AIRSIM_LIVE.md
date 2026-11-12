# Live AirSimNH evaluation

Verified on an A6000 with AirSimNH 1.8.1 offscreen.

## Setup

```bash
scripts/eval/airsim_setup.sh
```

The container's NVIDIA driver did not expose a Vulkan ICD, so the run uses
`-opengl` with Mesa llvmpipe. It is slow but functional for smoke tests.

## Smoke result

```bash
cd /workspace/DroneVLA
PYTHONPATH=. /workspace/venv-airsim/bin/python -m scripts.eval.evaluate_vla \
  --env airsim --policy mock --episode-id id-approach-01 \
  --out /workspace/runs/airsim_smoke --airsim-ip 127.0.0.1
```

Result:

- 100 steps
- collision: false
- intent score: 0.086
- success: false (software-rendered scripted policy did not reach the marker)

The live path exercised marker spawn/move/clear, vehicle reset, action stepping,
collision reporting and the intent evaluator.

## Fixes found

- AirSim vehicle is `SimpleFlight`; auto-detect it instead of assuming `Drone0`.
- `msgpack<1.0` is required by `msgpack-rpc-python`.
- Ignore the one-frame collision latch after a teleport.
- Image capture is optional; scripted smoke tests can skip the slow software
  framebuffer.

## Real OpenVLA-UAV result

The released `wangxiangyu0814/OpenVLA-UAV` checkpoint was served over HTTP and
run through the same manifest/episode:

- policy: OpenVLA-UAV, `unnorm_key="sim"`
- **100 steps**
- **success: 1.0**
- **intent score: 1.0**
- **collisions: 0**
- video: `real_vla_airsim.mp4`, 256×144, 10 s

The video is recorded directly from the AirSim camera stream while the policy is
acting. The A100 pod used for this run has been deleted.
