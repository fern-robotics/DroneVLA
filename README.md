# DroneVLA-RS

Rust implementation of the revised DroneVL thesis protocol: instruction-conditioned UAV flight intents evaluated in front of visual landmarks.

The protocol follows the rewritten thesis:

- intent families: primitive motion, landmark interaction (approach / pass-side / orbit / hold-at), compositional sequences;
- conditions: M0, M1 (SFT), continued SFT, M2 (GRPO after M1);
- SFT demonstrations from **UAV-Flow-Sim**;
- GRPO rollouts and evaluation on the **AirSimNH** split;
- metrics: mission success, intent-adherence score, collisions, safety interventions, path efficiency, paired bootstrap intervals.

## What is in this repository

- `src/` — deterministic kinematic simulator, software RGB/depth renderer, marker perception, safety gate, intent predicates, planner, linear policy, SFT/GRPO training, paired analysis, report/SVG generation.
- `scripts/prepare_uavflow.py` — converts a UAV-Flow-Sim parquet shard into compact compact DroneVL JSONL using the official OpenVLA-UAV action convention (`[dx,dy,dz,dyaw]`, metres/radians).
- `scripts/prepare_all_uavflow.py` — converts all 21 shards on a pod.
- `scripts/airsimnh_eval_uavflow.py` — evaluates a UAV-Flow-trained checkpoint in actual AirSimNH.
- `scripts/openvla_airsim_test.py`, `scripts/openvla_airsim_intent.py` — OpenVLA-UAV closed-loop bridge and intent mission in actual AirSimNH.

## Build and test

```bash
cargo test
cargo build --release
```

## Convert UAV-Flow-Sim

```bash
python3 -m venv .venv && .venv/bin/pip install pyarrow numpy
.venv/bin/python scripts/prepare_uavflow.py path/to/train-00000-of-00021.parquet data
```

## Train M1 and run the surrogate experiment

```bash
./target/release/dronevla m1 --uavflow data/uavflow_sim.jsonl --out runs/m1.json
./target/release/dronevla experiment --out runs/experiment --seeds 0,1,2 --epochs 25 --groups 80
```

`runs/experiment/report.md` contains the condition table, paired bootstrap differences, and learning curves.

## Actual AirSimNH

`scripts/airsimnh_eval_uavflow.py` expects a running AirSimNH RPC endpoint on `127.0.0.1:41451` and the reference Python harness in `/workspace/thesis`. It flies approach / pass / orbit / hold intents and scores them from the actual trajectory.

## Status and limits

- The Rust policy is a compact **goal-conditioned linear surrogate**, not OpenVLA-7B or π₀-FAST. It exists to exercise the protocol, the AirSimNH bridge, and the analysis end to end.
- SFT data are the real UAV-Flow-Sim trajectories.
- The OpenVLA-UAV checkpoint is the third-party UAV-Flow baseline (`wangxiangyu0814/OpenVLA-UAV`), not a checkpoint trained in this repository.
- π₀-FAST UAV integration is not available in the reference artifacts.
- The exact UAV-Flow-Eval UE4 `Collection` environment is Windows-only; actual AirSimNH 1.8.1 is used on Linux.

## Results

See [docs/RESULTS.md](docs/RESULTS.md).
