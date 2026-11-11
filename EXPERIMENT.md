# DroneVL Experiment

**Thesis title:** *DroneVL: Evaluating GRPO and PPO after LoRA-SFT for Instruction-Conditioned UAV Navigation*  
**Indonesian title:** *DroneVL: Evaluasi GRPO dan PPO setelah LoRA-SFT untuk Navigasi UAV Berbasis Instruksi*

**Status:** data pipeline, official SFT integrations, GRPO/PPO RL core and 30-episode eval suite implemented; full model-specific RL adapters and 7B smoke tests still pending.  
**Budget cap:** USD 20–30.  
**Last updated:** 2026-11-11

---

## 1. Research question

Within each VLA family:

> After LoRA-SFT on UAV demonstrations, does GRPO improve mission success and intent adherence relative to the SFT checkpoint, continued SFT, and a PPO control?

Primary comparison:

1. **GRPO − M1** (does RL add value after SFT?)
2. **GRPO − SFT+** (is the gain from RL, or just more optimization?)
3. **GRPO − PPO** (is any gain specific to GRPO's group-relative objective?)

The main thesis claim is the GRPO-vs-SFT comparison. GRPO-vs-PPO is an **algorithm ablation**, not the primary hypothesis.

---

## 2. Conditions

Each model family (OpenVLA-7B, π₀-FAST) has five conditions:

| Condition | Initialization | Training |
|---|---|---|
| **M0** | Base VLA weights | none; calibrated 4-D UAV decoder |
| **M1** | M0 | LoRA-SFT on UAV-Flow-Sim |
| **SFT+** | M1 | continued SFT, same data, extra steps |
| **GRPO** | M1 | GRPO on AirSimNH rollouts |
| **PPO** | M1 | PPO on AirSimNH rollouts |

M1, SFT+, GRPO and PPO share the same base weights, action interface, observation builder, controller, safety gate and evaluator.

---

## 3. Data

### SFT corpus

- Source: **UAV-Flow-Sim** (`wangxiangyu0814/UAV-Flow-Sim`).
- Full release: **~312,240 frames, ~10,400 episodes, ~55 GB, 21 shards**.
- Action convention: `[dx, dy, dz, dyaw]` in metres/radians at 10 Hz, body FRD frame, computed with the official OpenVLA-UAV loader.
- Pilot subset: **300 episodes / 9,437 frames** (one shard).
- Recommended partial corpus: **10% (~1,040 episodes / ~31k frames)**, stratified by instruction type and episode length.
- Splits: episode-level train/development; test episodes are never used for SFT or RL.

### RL and evaluation environment

- **Actual AirSimNH** (Microsoft AirSim 1.8.1, UE4/Vulkan, Linux GPU pod).
- Frozen intent manifests: approach, pass-left/right, orbit-CW/CCW, hold-at-distance, and held-out compositions.
- Policy observation: front RGB, instruction, UAV state, verified progress count.
- Evaluator-only data: landmark IDs/poses, intent predicates, terminal rules.

### Test split

- Target: **30 frozen episodes — 22 ID, 8 OOD**.
- Every condition is evaluated on the same episode, initial state and budget.
- Pilot limitation: one seed unless budget allows three.

---

## 4. Tasks and metrics

### Intent families

- Primitive: move forward/back/left/right, turn, climb/descend, hold position.
- Landmark interaction: approach within distance, pass specified side, orbit specified direction/angle, hold at distance.
- Composition: ordered combinations, including held-out sequences.

### Primary metrics

| Metric | Definition |
|---|---|
| Mission success | All mandatory intent predicates satisfied before terminal limit |
| Intent adherence `I` | Mean per-intent completion, in [0,1] |
| Ordered completion `P` | Completed intents / total intents |
| Collision rate | Episodes with at least one mission collision / episodes |
| Safety interventions | Clipped or rejected actions / parsed actions |
| Invalid actions | Parse failures / decisions |
| Path efficiency | Reference path / actual path, successful episodes only |
| Latency | Median/p95 inference and end-to-end decision latency |

Report success with Wilson intervals and intent score with bootstrap intervals. Paired differences use the same episodes across conditions; ID and OOD are reported separately. With one seed and 30 episodes, results are exploratory.

---

## 5. Training protocol

### 5.1 LoRA-SFT

#### OpenVLA-7B

- LoRA rank **32**, alpha 64, q/v projections (OpenVLA reference recipe).
- 256-bin action tokens × 4 dimensions.
- Batch 8–16 on A100 80GB; batch 1–2 on A40.
- Pilot budget: **5,000 steps** (paper-scale is 200,000 steps on 8×A100).
- Released `OpenVLA-UAV` checkpoint can be used as a ready M1 if training budget is tight; this must be reported as a third-party full-data checkpoint, not our M1.

#### π₀-FAST

- OpenPI recipe: LoRA rank **16**, alpha 16, attention + FFN (`gemma_2b_lora`).
- `action_horizon=10`, `max_token_len=180`, no extra delta transform.
- Base checkpoint: `gs://openpi-assets/checkpoints/pi0_fast_base`.
- Batch 4 on A40; official batch 32.
- Pilot budget: **10,000 steps** (official LIBERO recipe: 30,000).

### 5.2 GRPO

- Reference policy: frozen M1.
- G = 4 rollouts per group, same episode and initial state within a group.
- 100 groups per model (pilot).
- Reward: `R = 1.0·S + 0.5·P + 1.0·I − 1.0·C − 0.25·V − 0.1·J`.
- Group-normalized advantage, PPO-style ratio clipping, KL to M1.
- Token-level log-probabilities over the model's action tokens.

### 5.3 PPO

- Same rollouts and reward as GRPO, for a controlled algorithm comparison.
- Actor-critic with GAE; clipped surrogate objective; multiple epochs per rollout batch.
- Value function: small critic head or separate compact critic to avoid doubling 7B memory.
- PPO is an **ablation**; it must use the same rollout budget, base checkpoint and evaluation episodes as GRPO.

### Critical dependency

**Neither GRPO nor PPO is implemented yet.** Only SFT is implemented. Before a 100-group run, the trainer must pass a 5-group smoke test that produces valid token log-probabilities, reference KL, advantages and a saved checkpoint.

---

## 6. Experiment matrix

| Model | M0 | M1 | SFT+ | GRPO | PPO |
|---|---|---|---|---|---|
| OpenVLA-7B | ✓ | ✓ | ✓ | 100 groups | 100 groups |
| π₀-FAST | ✓ | ✓ | ✓ | 100 groups | 100 groups |

Total policies to evaluate: **10**. With 30 test episodes each: **300 AirSim episodes**, excluding training rollouts.

---

## 7. Compute and cost

Assumptions: A40 **$0.59/h**, A100 80GB community **$1.19/h**, one AirSim A40 **$0.59/h**.

| Stage | Model | GPU | Hours | Cost |
|---|---|---|---:|---:|
| SFT | π₀-FAST, 10k | A40 | 3 | $1.77 |
| SFT | OpenVLA, 5k | A100 80GB | 2.5 | $2.98 |
| GRPO | π₀-FAST | A40 + A40 sim | 3 | $3.54 |
| GRPO | OpenVLA | A100 + A40 sim | 3 | $5.34 |
| PPO | π₀-FAST | A40 + A40 sim | 4 | $4.72 |
| PPO | OpenVLA | A100 + A40 sim | 4 | $7.12 |
| Eval | both | A40 sim | 2 | $1.18 |
| **Total** | | | **~21.5 h** | **~$26.65** |

### Budget variants

| Plan | Cost | Coverage |
|---|---:|---|
| Both SFT + both GRPO, no PPO | **~$14.8** | Main GRPO-vs-SFT question |
| Both SFT + both GRPO + PPO on π₀ only | **~$19.5** | Fits $20 |
| Full matrix at 50 groups | **~$19–20** | Weaker statistics |
| Full matrix at 100 groups | **~$26.7** | Over $20 |

### GPU allocation

- OpenVLA SFT/RL: **A100 80GB** for batch size and memory.
- π₀-FAST SFT/RL: **A40 46GB** is sufficient.
- AirSim: separate **A40 46GB**; parallelize 4–8 AirSim instances for RL throughput.
- Do not co-locate OpenVLA-7B training and AirSim rendering on one 46GB A40.
- H100/B200 do not help GRPO much: rollouts dominate.

Stop rules: cap OpenVLA SFT at 5k; cap RL at 100 groups; reduce to 50 groups if throughput is low; stop and report SFT-only if the RL trainer fails its smoke test.

---

## 8. Current evidence

| Item | Status |
|---|---|
| Actual AirSimNH on RunPod | validated: out-and-back, square, visual grounding 2/2 |
| OpenVLA-UAV released checkpoint | runs closed-loop; approach intent 3/3 success, 10 m → 3.2–3.4 m |
| π₀-FAST SFT | 600-step checkpoint trained (loss 3.67 → 2.81); 10k run started but pod terminated by budget, no final checkpoint |
| GRPO trainer | not implemented |
| PPO trainer | not implemented |
| Rust surrogate harness | 912 paired episodes; M1 > M0; GRPO mixed; **not thesis model results** |
| AirSimNH pod | stopped |
| Compute balance | exhausted |

These are development results. None of them is a substitute for the missing full M1/GRPO/PPO matrix.

---

## 9. Limitations

- One seed and 30 test episodes: exploratory, not a powered benchmark.
- π₀-FAST uses a 300-episode subset while the released OpenVLA-UAV checkpoint is full-data; a fair comparison requires re-training OpenVLA on the same subset or scaling π₀-FAST to the same corpus.
- No official π₀-FAST UAV checkpoint exists; our adapter must be trained.
- Exact UAV-Flow-Eval UE4 scene is Windows-only; AirSimNH is the Linux evaluation environment.
- PPO on 7B may require an A100/H100 or a compact critic.
- Aviation-safety claims are out of scope.

---

## 9b. Implemented code (2026-11-11)

- `scripts/build_dataset_online.py` — single-pass 10% UAV-Flow-Sim subset; stratified episode selection, ID/OOD splits, image/JSONL export, stats/validation.
- `scripts/dataset/lerobot_writer.py` — split-specific LeRobot datasets for pi0-FAST (`uav_flow_sim_train/dev/test`).
- `scripts/dataset/rlds_builder.py` + `integrations/openvla/patch_openvla_uav.py` — RLDS export and OpenVLA registry patch (state 4→6, action 4→7 zero-padding).
- `scripts/run_openvla_sft.sh`, `scripts/run_pi0_sft.sh` — official-repo SFT launchers.
- `training/rl/` — tested GRPO/PPO losses, GAE, group advantages, rollout collector (7 CPU tests).
- `scripts/collect_rollouts.py` — UAV adapter → RL group collection; requires a policy server returning tokens/logprobs.
- `scripts/eval/` — frozen 30-episode manifest (22 ID/8 OOD), geometric predicates, mock + AirSim evaluator, VLA policy bridge (15 tests).
- `scripts/airsim_rollout.py` — thin observation/action/reward/token JSONL adapter (mock tested).

Known gaps before the full thesis matrix:

1. OpenVLA SFT has not been smoke-tested on a GPU after the RLDS build.
2. The OpenVLA and pi0-FAST servers currently expose actions; RL needs token ids and log-probabilities from the model servers.
3. GRPO/PPO losses are tested, but model-specific optimizer loops are not wired to OpenVLA/pi0-FAST.
4. AirSim `--airsim` eval and 100-group RL runs require a live AirSimNH pod.

## 10. Reproduction

```bash
cargo test
cargo run --release --bin dronevla -- m1 --uavflow data/uavflow_sim.jsonl --out runs/m1.json
```

Dataset conversion, OpenPI patches and AirSim policy clients:

- `scripts/prepare_uavflow.py`
- `scripts/prepare_all_uavflow.py`
- `integrations/openpi/uav_policy.py`
- `integrations/openpi/patch_config.py`
- `scripts/openvla_airsim_intent.py`
- `scripts/pi0_airsim_intent.py`

Full protocol and budget notes: [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md).
