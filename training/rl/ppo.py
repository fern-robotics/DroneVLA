"""PPO for token-level VLA action chunks, plus GAE.

The actor is optimized with the usual clipped surrogate over response tokens.
The critic is per decision (one value per action-chunk timestep), which is the
granularity at which rewards are assigned.  Entropy, when supplied by the
policy adapter, is a per-token tensor and is maximized over response tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .grpo import _align_like, _masked_mean


@dataclass
class PpoLoss:
    """Scalar terms returned by :func:`ppo_loss`."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor


def ppo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    *,
    entropy: torch.Tensor | None = None,
    old_values: torch.Tensor | None = None,
    value_mask: torch.Tensor | None = None,
    clip_range: float = 0.2,
    value_clip_range: float | None = None,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> PpoLoss:
    """Clipped PPO actor loss + clipped value loss + entropy bonus.

    Args:
        current_logprobs: ``(N, L)`` (or any ``(..., L)``) current log-probs.
        old_logprobs: Sampling-policy log-probs, shaped like ``current_logprobs``.
        advantages: ``(N,)`` per-rollout advantages or token-level values.
        response_mask: 1 for response tokens, 0 for prompt/padding tokens.
        values: Per-decision critic values, e.g. ``(N,)``.
        returns: Value targets, shaped like ``values``.
        entropy: Optional per-token entropy, shaped like ``current_logprobs``.
        old_values: Optional behavior-critic values for value clipping.
        value_mask: Optional mask over ``values`` (defaults to all ones).
        clip_range: Actor ratio clip range.
        value_clip_range: Value clip range; only used with ``old_values``.
        value_coef: Weight of the (0.5 * squared-error) value loss.
        entropy_coef: Weight of the entropy bonus.

    Returns:
        :class:`PpoLoss`; ``.loss`` is the scalar tensor to backpropagate.
    """
    if clip_range < 0.0:
        raise ValueError("clip_range must be non-negative")
    if value_coef < 0.0 or entropy_coef < 0.0:
        raise ValueError("value_coef and entropy_coef must be non-negative")

    current = torch.as_tensor(current_logprobs, dtype=torch.float32)
    old = _align_like(old_logprobs, current, "old_logprobs")
    advantage = _align_like(advantages, current, "advantages")
    mask = (_align_like(response_mask, current, "response_mask") > 0).to(current.dtype)

    ratio = torch.exp(torch.clamp(current - old, min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    policy_loss = -(surrogate * mask).sum() / mask.sum().clamp_min(1.0)

    values = torch.as_tensor(values, dtype=torch.float32)
    returns = _align_like(returns, values, "returns")
    if value_mask is None:
        value_mask = torch.ones_like(values)
    else:
        value_mask = _align_like(value_mask, values, "value_mask")
    squared_error = (values - returns) ** 2
    if old_values is not None and value_clip_range is not None:
        old_values = _align_like(old_values, values, "old_values")
        clipped_values = old_values + (values - old_values).clamp(-value_clip_range, value_clip_range)
        squared_error = torch.maximum(squared_error, (clipped_values - returns) ** 2)
    value_loss = 0.5 * _masked_mean(squared_error, value_mask)

    if entropy is None:
        entropy_mean = torch.zeros((), dtype=current.dtype, device=current.device)
    else:
        entropy_mean = _masked_mean(_align_like(entropy, current, "entropy"), mask)

    return PpoLoss(
        loss=policy_loss + value_coef * value_loss - entropy_coef * entropy_mean,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy_mean,
    )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation.

    Args:
        rewards: ``(T,)`` or ``(T, N)`` rewards.
        values: ``(T,)``/``(T, N)`` critic values, or ``(T+1, ...)`` when a
            bootstrap value for the state after the last transition is given.
        dones: Episode-end flags shaped like ``rewards`` (1 after a terminal
            transition, 0 otherwise).  No bootstrap crosses a done.
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        ``(advantages, returns)`` with the same shape as ``rewards``;
        ``returns = advantages + values[:T]``.
    """
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32)
    dones = torch.as_tensor(dones, dtype=torch.float32)
    if rewards.shape != dones.shape:
        raise ValueError("rewards and dones must have the same shape")
    steps = rewards.shape[0]
    if values.shape[0] not in (steps, steps + 1):
        raise ValueError("values must have T or T+1 entries along the time axis")
    if values.shape[1:] != rewards.shape[1:]:
        raise ValueError("values and rewards must agree on non-time dimensions")

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rewards[0])
    for t in reversed(range(steps)):
        next_value = values[t + 1] if t + 1 < values.shape[0] else torch.zeros_like(values[0])
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_advantage = delta + gamma * lam * nonterminal * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values[:steps]
