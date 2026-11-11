"""GRPO for token-level VLA action chunks.

The policy is treated as a token model: one log-probability per generated
token.  A rollout (sequence) has one scalar advantage, which is broadcast over
its response tokens.  The minimized loss is

    L = -mean[min(r * A, clip(r, 1-eps, 1+eps) * A)] + beta * KL

with ``r = exp(current_logprob - old_logprob)`` and the KL term computed from
a reference log-probability (or from the sampling/old policy when no reference
is supplied).  All means are taken over response-mask tokens only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GrpoLoss:
    """Scalar terms returned by :func:`grpo_loss`."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor
    ratio_mean: torch.Tensor
    clip_fraction: torch.Tensor


def _align_like(value, reference: torch.Tensor, name: str) -> torch.Tensor:
    """Convert ``value`` to a float tensor broadcastable to ``reference``."""
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.shape == reference.shape:
        return tensor
    if tensor.numel() == reference.numel():
        return tensor.reshape(reference.shape)
    if tensor.ndim == 1 and reference.ndim > 1 and tensor.shape[0] == reference.shape[0]:
        # Per-rollout values (advantages, masks) broadcast over response tokens.
        return tensor.reshape((tensor.shape[0],) + (1,) * (reference.ndim - 1)).expand_as(reference)
    try:
        return torch.broadcast_to(tensor, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} shape {tuple(tensor.shape)} is not broadcastable to {tuple(reference.shape)}"
        ) from exc


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over mask entries; returns 0 for an empty mask (never NaN)."""
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _kl_estimate(reference_logprobs: torch.Tensor, current_logprobs: torch.Tensor) -> torch.Tensor:
    """Stable k3 estimator of KL(reference || current), per token.

    ``exp(d) - d - 1`` with ``d = log ref - log current``.  ``expm1`` keeps the
    small-difference regime accurate; the expectation is non-negative.
    """
    # Bound the exponent for numerical safety on badly mismatched adapters.
    log_ratio = (reference_logprobs - current_logprobs).clamp(-20.0, 20.0)
    return torch.expm1(log_ratio) - log_ratio


def group_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return ``(R - group_mean) / (group_std + eps)`` for each reward.

    Args:
        rewards: ``(N,)`` scalar episode returns.
        group_ids: ``(N,)`` group label per reward; each group is normalized
            independently.
        eps: Added to the (population) standard deviation for stability.

    Groups with one member, or with zero variance, produce zero advantages.
    """
    rewards = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
    group_ids = torch.as_tensor(group_ids).reshape(-1)
    if rewards.shape != group_ids.shape:
        raise ValueError("rewards and group_ids must have the same shape")
    if rewards.numel() == 0:
        return torch.empty(0, dtype=torch.float32)

    advantages = torch.zeros_like(rewards)
    for group_id in torch.unique(group_ids):
        members = group_ids == group_id
        group_rewards = rewards[members]
        std = group_rewards.std(unbiased=False)
        advantages[members] = (group_rewards - group_rewards.mean()) / (std + eps)
    return advantages


def grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_range: float = 0.2,
    beta: float = 0.0,
    ref_logprobs: torch.Tensor | None = None,
) -> GrpoLoss:
    """Clipped GRPO loss for token-level action chunks.

    Args:
        current_logprobs: ``(..., L)`` log-probs under the policy being trained.
        old_logprobs: ``(..., L)`` log-probs from the sampling policy.
        advantages: ``(N,)`` group-normalized advantages (broadcast over
            tokens) or token-level values shaped like ``current_logprobs``.
        response_mask: 1 for generated/response tokens, 0 for ignored tokens
            (prompt or padding); shaped like ``current_logprobs``.
        clip_range: PPO/GRPO ratio clipping range ``eps``.
        beta: KL penalty coefficient.
        ref_logprobs: Optional reference-policy log-probs.  When omitted and
            ``beta > 0``, KL(old || current) is used.

    Returns:
        :class:`GrpoLoss`; ``.loss`` is the scalar tensor to backpropagate.
    """
    if clip_range < 0.0:
        raise ValueError("clip_range must be non-negative")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")

    current = torch.as_tensor(current_logprobs, dtype=torch.float32)
    old = _align_like(old_logprobs, current, "old_logprobs")
    advantage = _align_like(advantages, current, "advantages")
    mask = (_align_like(response_mask, current, "response_mask") > 0).to(current.dtype)

    log_ratio = torch.clamp(current - old, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
    surrogate = torch.minimum(unclipped, clipped)

    denominator = mask.sum().clamp_min(1.0)
    policy_loss = -(surrogate * mask).sum() / denominator
    if beta > 0.0:
        reference = old if ref_logprobs is None else _align_like(ref_logprobs, current, "ref_logprobs")
        kl = _masked_mean(_kl_estimate(reference, current), mask)
    else:
        kl = torch.zeros((), dtype=current.dtype, device=current.device)

    outside = ((ratio - 1.0).abs() > clip_range).to(current.dtype)
    return GrpoLoss(
        loss=policy_loss + beta * kl,
        policy_loss=policy_loss,
        kl=kl,
        ratio_mean=(ratio * mask).sum() / denominator,
        clip_fraction=(outside * mask).sum() / denominator,
    )
