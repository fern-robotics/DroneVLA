# Rust control plane

The VLA model and PyTorch trainer stay Python. Rust owns the experiment control
plane: evaluation metrics, paired reports, process supervision, the HTTP policy
gateway, UnrealCV/engine lifecycle, expert-cache materialization, and the
machine-readable plan for experiments 1–8.

## Commands

```bash
cargo build
./target/debug/dronevlctl

# Metrics + paired report from existing eval JSONs
dronevlctl eval-summary --dir runs/eval_online244/results --out runs/report.json
dronevlctl eval-compare \
  --baseline runs/eval_offline244/results \
  --candidate runs/eval_online244/results \
  --out runs/paired.json

# Experiment 1–8 plan (21 runs)
dronevlctl plan --out-dir runs/plan

# HTTP gateway in front of the Python OpenVLA server
dronevlctl proxy --bind 127.0.0.1:5007 \
  --upstream http://127.0.0.1:8010/predict

# Supervised command with timeout and process-group cleanup
dronevlctl run --timeout-s 3600 --retries 1 -- python scripts/train_b0_eggrpo.py ...

# Expert cache
dronevlctl cache --src /mnt/data/expert_cache_150 --out /mnt/data/expert_cache_150_copy

# UnrealCV smoke probe
dronevlctl unrealcv-probe --addr 127.0.0.1:9000
```

## Modules

| Module | Responsibility |
|---|---|
| `src/eval.rs` | load result JSON, success/NDTW, horizon bins, McNemar, paired bootstrap |
| `src/reward.rs` | reference-trajectory rollout reward (progress, cross-track, terminal) |
| `src/orchestrator.rs` | single-writer lock, retries, timeout, process-group kill, server readiness |
| `src/proxy.rs` | Python-compatible HTTP gateway: base64 image decode, 224×224 resize, payload translation, `/reset` |
| `src/unrealcv.rs` | UnrealCV framed client (text + raw PNG payloads) + engine supervisor |
| `src/cache.rs` | content-addressed expert/dataset cache |
| `src/experiment_plan.rs` | experiments 1–8 as runnable specifications |

## Experiment plan

The planner emits both `experiment_plan.csv` and `experiment_plan.json`:

- experiment 1: M0 / offline150 / online150 fixed-244
- experiment 2: paired offline vs online
- experiment 3: BC 0.0 / 0.2 / 0.5
- experiment 4: expert reward anchor on/off
- experiment 5: KL 0.0 / 0.01
- experiment 6: LR 5e-6 / 1e-5 / 2e-5
- experiment 7: offline/online 150 vs 300 groups (`MANIFEST` must contain at least 300 tasks for the online-300 arm)
- experiment 8: 5×30 / 15×10 / 30×5 online schedules

## Smoke test

```bash
cargo test
python3 scripts/smoke_rust_control_plane.py
```

The smoke test uses mock HTTP, mock UnrealCV (framed protocol), a real 1×1 PNG
resized to 224×224, synthetic eval JSONs, a temp expert cache, and a timeout
command. It requires no GPU, Unreal, or model weights.

## Compatibility notes

- The proxy matches `policy_proxy.py`: `image` → `image_png_b64`, and
  `actions/tokens/logprobs` → `action/tokens/logprobs/done`.
- PIL bicubic is approximated with `CatmullRom`; images already 224×224 are
  passed through unchanged, as in Python.
- UnrealCV request framing is `<magic 0x9E2B83C1><len><id>:<command>` and image
  responses are raw PNG bytes, matching the Python `unrealcv` package.
