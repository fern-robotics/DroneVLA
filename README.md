# DroneVL

Instruction-conditioned UAV navigation with **OpenVLA-7B** and **π₀-FAST**: LoRA-SFT on UAV-Flow-Sim, then GRPO/PPO, evaluated on AirSimNH flight intents.

## Status

| Area | State |
|---|---|
| Data | 10% UAV-Flow-Sim subset built and validated (936 episodes, 765/90/81 splits, 81 OOD) |
| π₀-FAST SFT | Official OpenPI path; 2-step smoke passed; token log-prob path verified on A40 |
| OpenVLA SFT | Official repo; 1-step LoRA-SFT verified on A100 (484 MB adapter) |
| GRPO / PPO | Tested RL core; model-specific adapters pending |
| Evaluation | 30-episode AirSimNH suite (22 ID / 8 OOD); mock tested, live AirSim pending |
| Rust crate | Surrogate protocol harness, not the thesis model |

## Quick start

```bash
# Data: single-pass 10% subset, RLDS-ready
python scripts/build_dataset_online.py --shards all \
  --out data/uav_flow_10pct --fraction 0.10 --formats jsonl

# π₀-FAST SFT (official OpenPI)
bash scripts/run_pi0_sft.sh

# OpenVLA SFT (official OpenVLA; apply RLDS patch first)
python integrations/openvla/patch_openvla_uav.py /path/to/openvla
bash scripts/run_openvla_sft.sh

# Evaluation
python -m scripts.eval.evaluate --mock
python -m scripts.eval.evaluate_vla --env airsim --policy websocket --port 8000

# Surrogate harness
cargo test
```

## Layout

- `scripts/dataset/` — selection, splits, image export, LeRobot/RLDS writers.
- `scripts/build_dataset_online.py` — one-shard-at-a-time 10% builder.
- `integrations/openvla/` — RLDS registration and official LoRA-SFT launcher.
- `integrations/openpi/` — UAV policy adapter; π₀-FAST token/log-prob patch.
- `integrations/verl/` — GRPO/PPO integration notes for OpenVLA.
- `training/rl/` — tested GRPO/PPO losses, GAE and rollout collector.
- `scripts/eval/` — 30-episode manifest, predicates, mock/AirSim evaluators.
- `src/` — Rust surrogate simulator, policies, analysis.

## Documentation

- [EXPERIMENT.md](EXPERIMENT.md) — conditions, data, metrics, budget.
- [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) — detailed protocol.
- [docs/DATA_BUILD.md](docs/DATA_BUILD.md) — 10% build result.
- [docs/RESULTS.md](docs/RESULTS.md) — surrogate results.

## Limits

- Rust policy is a linear surrogate, not OpenVLA/π₀-FAST.
- The released OpenVLA-UAV checkpoint is a third-party UAV-Flow baseline.
- No official π₀-FAST UAV checkpoint exists; it must be trained here.
- Live AirSim evaluation requires a separately running AirSimNH pod.
