# DroneVL experiment plan

Aligned with Chapter III of the thesis.

## Conditions

Each model (OpenVLA-7B, π₀-FAST) has five conditions:

| Condition | Start | Training |
|---|---|---|
| M0 | base VLA + calibrated UAV decoder | none |
| M1 | M0 | LoRA-SFT on UAV-Flow-Sim train |
| SFT+ | M1 | continued SFT, same train data |
| GRPO | M1 | GRPO on AirSimNH train rollouts |
| PPO | M1 | PPO on AirSimNH train rollouts, same rollout budget |

Primary hypothesis: **GRPO > M1**. Secondary: **GRPO vs SFT+** and **GRPO vs PPO**. PPO is an algorithm ablation, not a replacement for the main hypothesis.

## Data

- UAV-Flow-Sim 10% stratified subset: 936 episodes, 765/90/81 train/dev/test.
- Action: delta4 FRD, metres/radians, q01/q99 normalization.
- AirSimNH train rollouts for GRPO/PPO.
- Frozen evaluation manifest: 30 episodes, 22 ID + 8 OOD.
- Test manifest is not used for SFT/RL or checkpoint selection.

## SFT

### OpenVLA-7B

- LoRA rank 32, alpha 64, q_proj/v_proj.
- Batch 8 × grad-accum 2 = effective 16 on A100 80GB.
- LR 2e-4, grad clip 1, 5,000 steps, ~40k samples.
- Run `integrations/openvla/patch_lora_config.py` on the official repo first.

### π₀-FAST

- LoRA rank 16, alpha 16, attention + FFN.
- `action_horizon=10`, `max_token_len=180`.
- Batch 4 on A40 or 16 on A100 80GB; 10,000 steps, ~40k samples.

Both models use the same episodes and sample budget, not the same optimizer-step count.

## GRPO

- G = 4, 100 groups per model.
- Temperature 1, clip 0.2, β = 0.04, LR 1e-5, grad clip 1.
- Frozen M1 reference; old policy fixed within a group.
- Reward: `1.0·S + 0.5·P + 1.0·I − 1.0·C − 0.25·V − 0.1·J`.
- Skip and log zero-variance groups.
- 4–8 parallel AirSimNH instances.

## PPO

- Same M1 checkpoint, episodes, reward and rollout count as GRPO.
- Actor-critic, GAE γ=0.99, λ=0.95, clip 0.2, value 0.5, entropy 0.01.
- 2 epochs per rollout batch, LR 1e-5.
- Small critic head/compact critic; not a full 7B critic.
- If budget reduces the run, reduce GRPO and PPO together.

## Evaluation

- 30 frozen episodes, 22 ID / 8 OOD.
- Paired across conditions.
- Two seeds planned; third only if budget allows.
- Metrics: success, intent adherence, subgoal completion, collisions, invalid actions, safety interventions, path efficiency, latency.
- Report per family and ID/OOD separately; paired bootstrap intervals.

## Compute

- OpenVLA SFT/RL: A100 80GB.
- π₀-FAST SFT/RL: A40 46GB.
- AirSimNH: separate A40/L40S with 4–8 parallel instances.
- Full 100-group matrix exceeds $20; the budget-safe pilot uses 50 groups for both RL algorithms or PPO on one model.

## Current gaps

- OpenPI nnx update/loss adapters are wired and CPU-tested; the checkpoint-loading + rollout loop on GPU is still pending.
- π₀-FAST UAV full SFT/RL not run.
- M1/M2/SFT+/PPO matrix not run.
- Real-policy video exists for the third-party OpenVLA-UAV checkpoint, not for thesis M1.
