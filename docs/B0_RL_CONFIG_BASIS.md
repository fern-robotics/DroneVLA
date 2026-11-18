# Basis for B0 GRPO/PPO configs

## GRPO

| Parameter | Value | Basis |
|---|---:|---|
| group size | 8 | SimpleVLA-RL; Liu et al. use 8 trajectories/group |
| learning rate | 1e-5 | AeroDASH; RLinf safe setting |
| clip | 0.20 / 0.28 | AeroDASH; SimpleVLA-RL; Martingale GRPO |
| KL beta | 0.01 | AeroDASH |
| LoRA | rank 32, alpha 16, all-linear | AeroDASH rank 32/alpha 16; Liu et al. all-linear rank 32 |

Sources:
- [AeroDASH](https://arxiv.org/abs/2610.25407)
- [SimpleVLA-RL](https://arxiv.org/abs/2509.09674)
- [What Can RL Bring to VLA Generalization?](https://arxiv.org/abs/2505.19789)
- [RLinf-VLA](https://arxiv.org/abs/2510.06710)

## PPO

| Parameter | Value | Basis |
|---|---:|---|
| learning rate | 5e-6 | Martingale GRPO PPO table; VLA-RL uses 2e-5, but 5e-6 is the conservative VLA setting |
| clip | 0.20 / 0.28 | Martingale GRPO; SimpleVLA-RL |
| epochs | 1 | Liu et al. show >1 epoch gives no gain |
| entropy coef | 0.0 | Martingale GRPO table |
| gamma / lambda | 0.99 / 0.95 | standard PPO/GAE; VLA-RL |
| value head | 3-layer MLP, first action-token feature | Liu et al. shared actor-critic recipe |
| LoRA | rank 32, alpha 16, all-linear | Liu et al. rank 32 all-linear; AeroDASH alpha 16 |

Sources:
- [Martingale GRPO](https://arxiv.org/abs/2610.32431)
- [What Can RL Bring to VLA Generalization?](https://arxiv.org/abs/2505.19789)
- [VLA-RL](https://arxiv.org/abs/2505.18719)

## Deliberate deviations

- `group_size: 3` in PPO config: the native machine is limited to 3 stable engines.
- Value head uses the last prompt token hidden state as a proxy for the first action-token feature.
- KL beta in GRPO uses the sampling policy as reference in the current trainer, not a separately frozen M0 copy.
- Eval workers: 2, with two retries per task.

## Evaluation horizon

The fixed 20-step cap was removed. Both eval and rollout collection now use:

```text
max_steps_task = min(400, max(50, ceil(1.5 * len(reference_path_raw))))
```

This is not a paper hyperparameter; it is a reference-derived horizon that
prevents truncating the median (~38-step) and long (~379-step) UAV-Flow tasks.
`--no-adaptive-steps` restores the old fixed `--steps` behavior for smoke tests.
