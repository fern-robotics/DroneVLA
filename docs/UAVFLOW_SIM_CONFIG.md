# Native UAV-Flow simulator config

The canonical config is `configs/uavflow_sim_env.yaml`.

## Parallel engines

`scripts/run_parallel_rollouts.py` starts N native engines with a stagger. The
UnrealCV launcher checks the shared `unrealcv.ini` and increments the port from
9000 when it is occupied, so each engine gets its own server port.

```bash
python scripts/run_parallel_rollouts.py \
  --task /workspace/uavflow_pass/2025-05-12_21-48-30.json \
  --out /workspace/runs/parallel4 --workers 4 --steps 10 --stagger 10
```

Outputs:

```text
runs/parallel4/w0/episode.jsonl
runs/parallel4/w1/episode.jsonl
runs/parallel4/parallel.json
```

## Rendering

The engine is launched with Vulkan by default. On this pod no NVIDIA Vulkan ICD
is present, so Unreal renders through llvmpipe (CPU); OpenVLA still uses the GPU.
To use NVIDIA rendering, install a matching `nvidia_icd.json` from the host
driver and verify with `vulkaninfo --summary`.

## Reproducibility

- `scripts/sim_env_report.py` prints the scene path, binary, display, ports,
  Vulkan ICDs, and GPU availability.
- `scripts/validate_uavflow_pool.py` verifies 10,109 train / 273 test, zero
  overlap, and valid reference paths.
- `configs/b0_grpo_full.yaml` fixes prompt/state/action contract and seed.
