# DroneVL RL core

A small, self-contained PyTorch implementation of the token-level GRPO/PPO
losses and generic rollout collection used by the DroneVL protocol.

**Status: the algorithms and rollout plumbing are unit-tested on CPU with
synthetic tensors and a mock environment. This is not tested against OpenVLA,
pi0-FAST, veRL, TRL, or AirSim, and it does not load any model checkpoint.**
Model-specific log-probability extraction is deliberately out of scope.

## Files

| File | Contents |
| --- | --- |
| `types.py` | `TokenActionChunk`, `Transition`, `EpisodeRollout`, `GroupRollout` |
| `grpo.py` | `group_advantages`, clipped `grpo_loss` with masked KL |
| `ppo.py` | clipped `ppo_loss`, `compute_gae` |
| `rollout.py` | `RolloutCollector` (no AirSim import) |
| `../../tests/test_rl_core.py` | CPU tests |

## Conventions

- Log-probabilities are natural logs, one value per generated token.
- `TokenActionChunk.tokens` and `.logprobs` are 1-D `(L,)`; `.actions` is the
  action vector or `(H, A)` chunk for that decision.
- Loss inputs are batched `(N, L)` token tensors. A per-rollout advantage
  `(N,)` is broadcast across that rollout's response tokens.
- `response_mask == 1` marks generated tokens. Prompt tokens, padding, and
  failed/skipped token positions are `0`. All loss means divide by the number
  of unmasked tokens (`clamp_min(1)`), so an empty mask gives zero, not NaN.
- `group_advantages` returns `(R - group_mean) / (group_std + eps)` using the
  population standard deviation. Zero-variance and singleton groups give zero
  advantages.

### GRPO

```python
out = grpo_loss(current_logprobs, old_logprobs, advantages, response_mask,
                clip_range=0.2, beta=0.01, ref_logprobs=ref_logprobs)
out.loss.backward()
```

`grpo_loss` minimizes

```text
L = -mean[min(r * A, clip(r, 1-eps, 1+eps) * A)] + beta * KL
r = exp(current_logprob - old_logprob)
```

If `ref_logprobs` is given, KL is the per-token k3 estimate
`exp(d) - d - 1`, `d = log ref - log current`, masked and averaged. Without a
reference, `beta > 0` uses the old/sampling policy as the reference, i.e.
KL(old || current). The returned `GrpoLoss` exposes `loss`, `policy_loss`,
`kl`, `ratio_mean`, and `clip_fraction`.

### PPO

```python
out = ppo_loss(current_logprobs, old_logprobs, advantages, response_mask,
               values, returns, entropy=token_entropy,
               value_coef=0.5, entropy_coef=0.01)
out.loss.backward()

advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
```

`ppo_loss` combines the clipped actor surrogate, a (optionally clipped) value
loss, and an entropy bonus. `compute_gae` handles 1-D or `(T, N)` tensors, an
optional `T+1` bootstrap value, and blocks bootstrapping across `done`.
`PpoLoss` exposes `loss`, `policy_loss`, `value_loss`, and `entropy`.

### Rollout collection

```python
collector = RolloutCollector(env, policy, group_size=4, max_steps=100)
group = collector.collect_group(episode_seed=17, group_id=0)
rewards = group.rewards                  # (G,)
advantages = group.advantages            # normalized within the group
for rollout in group.rollouts:
    for transition in rollout.transitions:
        ...  # observation, instruction, action chunk with tokens/logprobs, reward
```

The environment must provide `reset()` and `step(action)`; `reset(seed=...)`
and Gymnasium-style 5-tuple steps are supported. The collector calls
`step()` once per policy decision, forwarding the returned action array
(chunked execution is the environment adapter's job). The policy must
provide `act(observation, instruction) -> {"actions", "tokens", "logprobs"}`
(optionally `"value"`). Every decision is logged as a `Transition`, and each
rollout in a group is reset with the same episode seed. `group.advantages` is
populated by `group_advantages` from the per-rollout total rewards.

## Install and test

```bash
python3 -m venv .venv-rl
.venv-rl/bin/pip install -r requirements-rl.txt pytest
.venv-rl/bin/python -m pytest tests/test_rl_core.py -v
```

The tests are pure CPU PyTorch and cover group-advantage normalization, GRPO
ratio clipping and masking, PPO loss shapes, GAE against a hand-computed
example, and rollout group consistency/logging with a mock env and policy.
If an inherited ROS `PYTHONPATH` breaks pytest plugin autoloading, run the
same command with `PYTHONPATH=` or `env -u PYTHONPATH`.

## Model integration boundary

This core stops at tensors. Real VLA integration needs an adapter that:

1. runs the model and collects the response-token log-probabilities,
2. records the token ids and response mask for each action chunk,
3. computes rewards from the DroneVL/AirSim intent environment,
4. feeds `(current, old, ref) logprobs + mask + group advantages` here.

Those adapters are separate from this package and are **not implemented or
tested here**. Production OpenVLA/pi0 adapters must use veRL or TRL where
possible for the training loop, distributed execution, and checkpoint
handling; use this core only where a dependency-free, auditable loss/rollout
reference is needed. A model adapter is not "integrated"
until its own log-prob parity and rollout tests pass.
