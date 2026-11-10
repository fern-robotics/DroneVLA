# DroneVL experiment plan — OpenVLA-7B and π₀-FAST

**Budget cap:** USD 20 for compute, storage and evaluation  
**Status:** π₀-FAST SFT running; OpenVLA SFT and both GRPO stages still to build  
**Last updated:** 2026-11-10

---

## 1. Objective

Test the thesis hypothesis within each model family:

> Does GRPO continued from an LoRA-SFT checkpoint improve instruction-conditioned UAV mission success and intent adherence compared with M1 and a continued-SFT control?

Two model families, four conditions each:

| Condition | Definition |
|---|---|
| **M0** | Base VLA weights with a calibrated 4-D UAV action decoder |
| **M1** | M0 + LoRA-SFT on UAV-Flow-Sim demonstrations |
| **SFT+** | Continued SFT from M1 (same demonstrations, extra optimizer steps) |
| **M2** | GRPO from the same M1 checkpoint, using AirSimNH rollouts |

Primary contrast: **M2 − M1** within each model. Secondary: **M2 − SFT+**, **M1 − M0**.

This is a **budget-constrained pilot**, not a reproduction of the UAV-Flow 200k-step OpenVLA training run or the OpenPI 30k-step LIBERO recipe. The plan states every deviation explicitly.

---

## 2. Data and splits

### SFT data

- Source: **UAV-Flow-Sim** (`wangxiangyu0814/UAV-Flow-Sim`), one shard for the pilot.
- Conversion: official OpenVLA-UAV local-frame convention, `[dx, dy, dz, dyaw]` in metres/radians, 10 Hz.
- Pilot subset: **300 episodes, 9,437 frames, 2,359 action-chunk samples** (action horizon 10).
- Splits: episode-level train/development; the test split is never used for SFT or GRPO.

### GRPO and evaluation environment

- Environment: **actual AirSimNH** (Microsoft AirSim 1.8.1, Linux, UE4/Vulkan) on a GPU pod.
- Episodes: frozen intent manifests — approach, pass-left/right, orbit-CW/CCW, hold-at-distance, and held-out compositions.
- Policy inputs: front RGB, instruction, UAV state, verified progress count. Depth is safety-gate only.
- Evaluator inputs: landmark identities/poses, intent predicates, terminal rules. Never given to the policy.

### Test split

- **30 frozen AirSimNH episodes**: 22 ID and 8 held-out intent compositions.
- One training seed per model in this budget. The thesis requires multiple seeds; that is a stated limitation.
- Pairing: every condition sees the same episode, initial state, controller, safety gate and budget.

---

## 3. Model and training configuration

### π₀-FAST (OpenPI)

Follow the OpenPI π₀-FAST recipe as closely as the hardware allows:

| Item | OpenPI reference | This pilot |
|---|---|---|
| LoRA | rank 16, alpha 16, attention + FFN | **same** |
| Action horizon | 10 | **10** |
| Max token length | 180 | **180** |
| Extra delta transform | none for delta actions | **none** |
| Batch | 32 | **4** (A40 46 GB) |
| Steps | 30,000 (LIBERO) | **10,000** |
| Action dim | 7 (LIBERO) | **4** (UAV) |

- Base checkpoint: `gs://openpi-assets/checkpoints/pi0_fast_base`.
- Optimizer/schedule: OpenPI defaults.
- 10k steps × batch 4 = 40k samples ≈ 17 epochs on 2,359 samples.

### OpenVLA-7B

| Item | OpenVLA / UAV-Flow reference | This pilot |
|---|---|---|
| LoRA | rank 32, alpha 64, q/v projections | **same** |
| Action representation | 256 bins × 4 dimensions | **same** |
| Batch | 32 (8×A100) | **1–2** (A40 46 GB) |
| Steps | 200,000 (UAV-Flow) | **3,000–5,000** |
| Data | UAV-Flow-Sim | same shard |

- If A40 memory forces OOM at batch 2, use batch 1 with gradient accumulation (if supported) or cap at 3k steps.
- If an A100 80 GB is cheaper per completed step, use it for this stage only.

### GRPO (both models)

- Reference policy: frozen M1 checkpoint.
- G = **4 rollouts per group**, same episode and initial state.
- **100 groups per model** (pilot), ~400 AirSim episodes per model.
- Reward from the thesis:
  `R = 1.0·S + 0.5·P + 1.0·I − 1.0·C − 0.25·V − 0.1·J`
  where S = success, P = ordered subgoal completion, I = intent-adherence score, C = collision, V = invalid-action rate, J = safety-intervention rate.
- Group-normalized advantage; PPO-style ratio clipping; KL penalty to M1.
- Group members share the same initial conditions; policy sampling seeds differ, perception noise may be fixed or varied and must be logged.

**Critical engineering gap:** neither OpenVLA GRPO nor π₀-FAST GRPO is implemented yet. Only SFT is implemented. The GRPO stage should not start until the rollout logger, token log-probability path, reference policy, and optimizer loop are tested on a 5-group smoke run.

---

## 4. Compute and budget plan

Assumed prices: A40 **$0.59/h**, A100 80 GB secure **$1.79/h**, one AirSim pod **$0.59/h**.

| Stage | Hardware | Estimated wall time | Estimated cost |
|---|---|---:|---:|
| π₀-FAST SFT, 10k | A40 | 3 h | **$1.8** |
| OpenVLA SFT, 5k | A40 | 4–6 h | **$2.4–3.5** |
| π₀-FAST GRPO, 100 groups | A40 train + A40 sim | 3 h | **$3.5** |
| OpenVLA GRPO, 100 groups | A40 train + A40 sim | 3–4 h | **$3.5–4.7** |
| Frozen test evaluation | A40 sim | 1–2 h | **$0.6–1.2** |
| Storage, retries, buffer | — | — | **$3.0** |
| **Total** | | **~15–18 h** | **~$15–18** |

The plan fits **$20** only if:

1. AirSim is stopped whenever it is not rolling out or evaluating.
2. OpenVLA SFT is capped at **5k steps**; 10k is not affordable at A40 batch 1–2.
3. Checkpoints are pruned: 8 GB per π₀-FAST checkpoint and ~15 GB per OpenVLA checkpoint; the 50 GB pod cannot hold many.
4. GRPO stays at **100 groups**; 100 is a pilot, not a powered comparison.
5. No more than one training pod and one AirSim pod run concurrently.

Current balance is below the plan total; a top-up of roughly **$15** is required.

### Stop rules

- OpenVLA SFT > 6 h or repeated OOM → stop at 3k steps and record the deviation.
- GRPO smoke test fails or rollout throughput < 20 episodes/h → skip GRPO, finish SFT-only results, preserve budget.
- π₀-FAST inference > 5 s/decision → reduce groups to 50 or use fixed perception noise.
- Any stage exceeds its estimate by 50% → stop and reassess before continuing.

---

## 5. Evaluation protocol

Per episode, record:

- success (all mandatory intent predicates satisfied);
- intent-adherence score `I`;
- ordered subgoal completion;
- collision events and episodes with collision;
- safety-gate accepts, clips, rejections;
- invalid actions / parse failures;
- path length, reference path length, success-only efficiency;
- inference and end-to-end decision latency;
- terminal reason and failure category.

Analysis:

- success rate with Wilson intervals;
- mean intent score with bootstrap intervals;
- paired differences M1−M0, M2−M1, M2−SFT+;
- report ID and OOD separately;
- per-family breakdown (primitive / interaction / composition);
- exploratory only: n = 30 episodes, one seed.

No claim of statistical significance will be made from 100 GRPO groups and one seed.

---

## 6. Stage plan and go/no-go

| Stage | Deliverable | Gate |
|---|---|---|
| 0. Environment | AirSimNH + VNC + RPC; intent oracle validation | oracle succeeds on all test episodes |
| 1. π₀-FAST SFT | 10k-step checkpoint | dev success above M0; latency measured |
| 2. OpenVLA SFT | 3k–5k-step checkpoint | dev success above M0; fits A40 |
| 3. π₀-FAST GRPO | 100-group M2 | 5-group smoke run produces valid advantages and a checkpoint |
| 4. OpenVLA GRPO | 100-group M2 | same |
| 5. Final evaluation | 8 policies × 30 test episodes | frozen test manifest, paired logs |
| 6. Analysis/report | tables, paired CIs, failure analysis | raw logs traceable to episodes |

Current position: **stage 1 running** (π₀-FAST SFT, ~615/10,000 at the time of writing). AirSim pod is stopped to preserve budget. OpenVLA SFT setup and both GRPO trainers are the critical path.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| OpenVLA 7B OOM on A40 | batch 1, gradient checkpointing, shorter context; fall back to A100 for SFT only |
| GRPO implementation missing | build and smoke-test on 5 groups before spending on 100 |
| AirSim crashes during marker spawn | restart simulator; reuse validated marker manifests; log cleanup |
| π₀-FAST inference 2–15 s | warm up JAX; disable websocket keepalive; reduce groups if needed |
| Overfitting 2.3k samples at 17 epochs | monitor dev success, not training loss; stop early if dev degrades |
| Checkpoint disk exhaustion | save every 2k steps, keep only latest; delete base cache after load |
| Budget overrun | stop rules above; AirSim pod off when idle |
| One seed only | report as pilot; do not claim generalization |

---

## 8. Artifacts

- Frozen AirSimNH manifests and their hashes.
- SFT dataset conversion script and dataset statistics.
- LoRA configs, ranks, alpha, target modules, batch, steps, seed.
- GRPO group logs: episode IDs, sampled actions/tokens, old/current/reference log-probs, rewards, advantages, skipped groups.
- Per-episode decision/trajectory logs, safety-gate records, intent events.
- Checkpoints and checkpoint metadata (step, config hash, parent M1).
- Analysis notebook/script that recomputes every table from raw logs.

---

## 9. Deviations from the thesis

- π₀-FAST: 10k SFT steps instead of the 30k OpenPI recipe; batch 4 instead of 32.
- OpenVLA: 3k–5k SFT steps instead of 200k; batch 1–2 instead of 32.
- GRPO: 100 groups instead of a powered budget; one seed instead of several.
- UAV-Flow-Sim: one shard (300 episodes) instead of the full 55 GB dataset.
- No π₀-FAST UAV checkpoint exists in the reference release, so the UAV adapter is trained here rather than loaded.
