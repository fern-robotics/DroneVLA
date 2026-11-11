"""Data structures shared by the DroneVL RL core.

Per-decision tensors are unbatched: ``tokens`` and ``logprobs`` are 1-D with
one entry per generated token, and ``actions`` is a 1-D action vector (or a
``(H, A)`` action chunk).  The loss functions accept batched ``(N, L)`` token
tensors as documented in :mod:`training.rl.grpo` and
:mod:`training.rl.ppo`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TokenActionChunk:
    """One policy decision: an action chunk and its generated tokens."""

    actions: torch.Tensor
    tokens: torch.Tensor
    logprobs: torch.Tensor

    def __post_init__(self) -> None:
        self.actions = torch.as_tensor(self.actions, dtype=torch.float32)
        self.tokens = torch.as_tensor(self.tokens, dtype=torch.long)
        self.logprobs = torch.as_tensor(self.logprobs, dtype=torch.float32)
        if self.tokens.ndim != 1:
            raise ValueError("tokens must be 1-D (one entry per token)")
        if self.logprobs.shape != self.tokens.shape:
            raise ValueError("logprobs and tokens must have the same shape")

    @property
    def num_tokens(self) -> int:
        return int(self.tokens.shape[0])

    def to(self, device: torch.device | str) -> "TokenActionChunk":
        return TokenActionChunk(
            self.actions.to(device),
            self.tokens.to(device),
            self.logprobs.to(device),
        )

    def detach(self) -> "TokenActionChunk":
        return TokenActionChunk(
            self.actions.detach(),
            self.tokens.detach(),
            self.logprobs.detach(),
        )


@dataclass
class Transition:
    """A single logged decision plus the environment response to it."""

    observation: Any
    instruction: str
    action_chunk: TokenActionChunk
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
    value: float | None = None
    step: int = 0


@dataclass
class EpisodeRollout:
    """Ordered decisions from one episode."""

    transitions: list[Transition] = field(default_factory=list)
    episode_seed: int | None = None
    group_id: int = 0
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        for step, transition in enumerate(self.transitions):
            transition.step = step

    @property
    def decisions(self) -> list[Transition]:
        return self.transitions

    @property
    def length(self) -> int:
        return len(self.transitions)

    @property
    def rewards(self) -> torch.Tensor:
        return torch.tensor([float(t.reward) for t in self.transitions], dtype=torch.float32)

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())

    @property
    def success(self) -> bool:
        return any(bool(t.info.get("success", False)) for t in self.transitions)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def token_data(self) -> dict[str, torch.Tensor]:
        """Concatenate every decision's tokens/logprobs.

        ``response_mask`` is all ones because every collected token is part of
        the generated response.  A model adapter that mixes prompt and
        response tokens must supply its own mask to the loss functions.
        """
        if not self.transitions:
            empty = torch.empty(0, dtype=torch.long)
            return {
                "tokens": empty,
                "logprobs": torch.empty(0, dtype=torch.float32),
                "response_mask": torch.empty(0, dtype=torch.float32),
            }
        chunks = [t.action_chunk for t in self.transitions]
        logprobs = torch.cat([c.logprobs for c in chunks])
        return {
            "tokens": torch.cat([c.tokens for c in chunks]),
            "logprobs": logprobs,
            "response_mask": torch.ones_like(logprobs),
        }


@dataclass
class GroupRollout:
    """G rollouts collected from the same episode seed (a GRPO group)."""

    group_id: int
    rollouts: list[EpisodeRollout] = field(default_factory=list)
    episode_seed: int | None = None
    advantages: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for rollout in self.rollouts:
            rollout.group_id = self.group_id
            if rollout.episode_seed is None:
                rollout.episode_seed = self.episode_seed

    @property
    def num_rollouts(self) -> int:
        return len(self.rollouts)

    @property
    def rewards(self) -> torch.Tensor:
        return torch.tensor([r.total_reward for r in self.rollouts], dtype=torch.float32)

    @property
    def mean_reward(self) -> float:
        if not self.rollouts:
            return 0.0
        return float(self.rewards.mean())
