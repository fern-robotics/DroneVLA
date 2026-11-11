# veRL integration for OpenVLA GRPO and PPO

**Status: template / not yet tested in this repository.** The working path is
SimpleVLA-RL's veRL integration for OpenVLA-OFT.

Reference: https://github.com/chemark/ai-agent-book/tree/main/chapter7/SimpleVLA-RL

## Role of this directory

veRL owns:

- rollout workers and batching,
- token-level log-probabilities,
- advantage estimation (GRPO group baseline or PPO/GAE),
- clipping, KL, optimizer and distributed training.

This repository owns only:

- the AirSim intent environment and reward,
- the observation/action adapter,
- JSONL logging of every token/action/reward,
- frozen episode manifests and evaluation.

## GRPO sketch

- `adv_estimator=grpo`
- `n_samples=4` rollouts per episode/group
- same episode and initial state within a group
- reward from `scripts/airsim_rollout.py`
- no critic

## PPO sketch

- `adv_estimator=gae`
- critic/value model: small head or compact critic (a full 7B critic is expensive)
- multiple PPO epochs per rollout batch
- same rollout budget and evaluation episodes as GRPO

## Fairness rules

- M1, GRPO and PPO share the same base checkpoint and evaluation episodes.
- Compare **samples seen** or environment steps, not raw optimizer steps.
- Log skipped groups, zero-variance groups, rollout failures and safety interventions.
- Run a 5-group smoke test before a 100-group run.

This directory intentionally contains no reimplementation of PPO/GRPO. If veRL
cannot load the OpenVLA worker, use TRL behind a thin VLA wrapper, not a from-scratch
RL implementation.
