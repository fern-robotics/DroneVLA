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
