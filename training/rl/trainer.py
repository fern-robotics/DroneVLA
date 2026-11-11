"""Generic GRPO/PPO optimizer loops over the tested RL core.

Model integration is deliberately external: a policy adapter provides
``score_tokens(tokens, contexts) -> logprobs`` (and ``values(contexts)`` for PPO).
OpenVLA/pi0-FAST adapters must keep their model-specific forward pass outside
this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch

from .grpo import GrpoLoss, grpo_loss
from .ppo import PpoLoss, ppo_loss
from .types import GroupRollout


class TokenPolicy(Protocol):
    def score_tokens(self, tokens: torch.Tensor, contexts: Sequence[Any]) -> torch.Tensor: ...
    def values(self, contexts: Sequence[Any]) -> torch.Tensor: ...


@dataclass
class TokenBatch:
    tokens: torch.Tensor          # (N, L)
    old_logprobs: torch.Tensor    # (N, L)
    response_mask: torch.Tensor   # (N, L)
    advantages: torch.Tensor      # (N,)
    contexts: list[Any]


def collate_groups(groups: Sequence[GroupRollout], pad_token: int = 0) -> TokenBatch:
    tokens, old, masks, adv, contexts = [], [], [], [], []
    for group in groups:
        if group.advantages is None:
            raise ValueError("group.advantages is required; call RolloutCollector with normalize_advantages=True")
        for i, rollout in enumerate(group.rollouts):
            data = rollout.token_data()
            if data["tokens"].numel() == 0:
                continue
            tokens.append(data["tokens"])
            old.append(data["logprobs"])
            masks.append(data["response_mask"])
            adv.append(group.advantages[i])
            contexts.append([t.observation for t in rollout.transitions])
    if not tokens:
        raise ValueError("no non-empty rollouts")
    length = max(int(t.shape[0]) for t in tokens)
    n = len(tokens)
    tok = torch.full((n, length), pad_token, dtype=torch.long)
    old_lp = torch.zeros((n, length), dtype=torch.float32)
    mask = torch.zeros((n, length), dtype=torch.float32)
    for i, (t, lp, m) in enumerate(zip(tokens, old, masks)):
        tok[i, : t.numel()] = t
        old_lp[i, : lp.numel()] = lp
        mask[i, : m.numel()] = m
    return TokenBatch(tok, old_lp, mask, torch.stack(adv), contexts)


def grpo_update(policy: TokenPolicy, optimizer: torch.optim.Optimizer, groups: Sequence[GroupRollout],
                *, clip_range: float = 0.2, beta: float = 0.0, max_grad_norm: float = 1.0) -> GrpoLoss:
    batch = collate_groups(groups)
    optimizer.zero_grad(set_to_none=True)
    current = policy.score_tokens(batch.tokens, batch.contexts)
    loss = grpo_loss(current, batch.old_logprobs, batch.advantages, batch.response_mask,
                     clip_range=clip_range, beta=beta)
    loss.loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in _parameters(policy)], max_grad_norm)
    optimizer.step()
    return loss


def ppo_update(policy: TokenPolicy, optimizer: torch.optim.Optimizer, groups: Sequence[GroupRollout],
               *, clip_range: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.0,
               max_grad_norm: float = 1.0) -> PpoLoss:
    batch = collate_groups(groups)
    optimizer.zero_grad(set_to_none=True)
    current = policy.score_tokens(batch.tokens, batch.contexts)
    values = policy.values(batch.contexts).reshape(-1)
    if values.numel() != batch.tokens.shape[0]:
        raise ValueError("PPO adapter must return one value per rollout")
    returns = batch.advantages.detach()
    loss = ppo_loss(current, batch.old_logprobs, batch.advantages, batch.response_mask,
                    values=values, returns=returns, clip_range=clip_range,
                    value_coef=value_coef, entropy_coef=entropy_coef)
    loss.loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in _parameters(policy)], max_grad_norm)
    optimizer.step()
    return loss


def _parameters(policy: Any):
    if hasattr(policy, "parameters"):
        return policy.parameters()
    raise TypeError("policy must expose .parameters() for the optimizer")
