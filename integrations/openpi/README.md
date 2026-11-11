# OpenPI / pi0-FAST: official SFT + thin JAX RL

**Rule:** SFT uses the official OpenPI repository. The only additions here are a
UAV policy adapter and a training config; the model, FAST tokenizer, loss,
optimizer and checkpointing remain OpenPI code.

Official source: https://github.com/Physical-Intelligence/openpi

## SFT

Official config shape (see `patch_config.py` in this directory):

- model: `Pi0FASTConfig(action_dim=4, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora")`
- LoRA rank 16, alpha 16, attention + FFN
- weight loader: `gs://openpi-assets/checkpoints/pi0_fast_base/params`
- data: LeRobot UAV dataset with keys `image`, `state`, `actions`, `prompt`
- transforms: `uav_policy.UAVInputs` / `UAVOutputs`

```bash
uv sync
HF_LEROBOT_HOME=/workspace/lerobot uv run scripts/compute_norm_stats.py --config-name pi0_fast_uav_low_mem
HF_LEROBOT_HOME=/workspace/lerobot XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi0_fast_uav_low_mem --exp-name uav_pilot
```

## Rollout adapter

`scripts/pi0_airsim_intent.py` is the thin AirSim client. Keep it limited to
observation capture, policy-server calls, action execution and logging.

## GRPO / PPO

There is no standard π₀-FAST RL library path. Use a thin JAX loop around OpenPI
that:

1. collects rollouts with the same adapter,
2. keeps the frozen M1 reference,
3. computes FAST-token log-probabilities and group advantages,
4. applies clipped policy-gradient updates to the LoRA parameters,
5. saves checkpoints in OpenPI format.

Do not reimplement SFT or the FAST tokenizer. The RL loop is the only new training
code and must be smoke-tested on 5 groups before spending on 100.
