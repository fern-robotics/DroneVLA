"""Generic, simulator-agnostic rollout collection for VLA policy groups.

The environment only needs ``reset()`` and ``step(action)``; the policy only
needs ``act(observation, instruction)`` returning ``actions``, ``tokens`` and
``logprobs``.  The same episode seed is used for every rollout in a group, so
a GRPO group shares its initial state while the stochastic policy varies.
"""

from __future__ import annotations

from typing import Any

import torch

from .grpo import group_advantages
from .types import EpisodeRollout, GroupRollout, TokenActionChunk, Transition


def _unpack_reset(result: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(result, tuple):
        observation = result[0]
        info = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
        return observation, dict(info)
    return result, {}


def _unpack_step(result: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    """Accept both 4-tuple and Gymnasium-style 5-tuple step results."""
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated), bool(truncated), dict(info)
    if len(result) == 4:
        observation, reward, done, info = result
        return observation, float(reward), bool(done), False, dict(info)
    raise ValueError("step() must return (obs, reward, done, info) or a 5-tuple")


class RolloutCollector:
    """Collect one GRPO group of ``group_size`` rollouts on one episode seed."""

    def __init__(
        self,
        env: Any,
        policy: Any,
        group_size: int = 4,
        max_steps: int = 100,
        instruction: str | None = None,
        normalize_advantages: bool = True,
    ) -> None:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.env = env
        self.policy = policy
        self.group_size = group_size
        self.max_steps = max_steps
        self.instruction = instruction
        self.normalize_advantages = normalize_advantages

    def _reset(self, seed: int | None) -> tuple[Any, dict[str, Any]]:
        try:
            result = self.env.reset(seed=seed)
        except TypeError:
            # Legacy environments: seed separately, then reset without a seed.
            seed_fn = getattr(self.env, "seed", None)
            if callable(seed_fn):
                seed_fn(seed)
            result = self.env.reset()
        return _unpack_reset(result)

    def _resolve_instruction(self, info: dict[str, Any], current: str) -> str:
        if self.instruction is not None:
            return self.instruction
        if info.get("instruction"):
            return str(info["instruction"])
        if current:
            return current
        return str(getattr(self.env, "instruction", "") or "")

    def _act(self, observation: Any, instruction: str) -> tuple[TokenActionChunk, float | None]:
        output = self.policy.act(observation, instruction)
        if not isinstance(output, dict):
            raise TypeError("policy.act() must return a dict")
        try:
            chunk = TokenActionChunk(
                output["actions"], output["tokens"], output["logprobs"]
            ).detach()
        except KeyError as exc:
            raise KeyError(f"policy.act() output is missing {exc.args[0]!r}") from exc
        value = output.get("value")
        return chunk, None if value is None else float(value)

    def collect_group(self, episode_seed: int | None, group_id: int = 0) -> GroupRollout:
        """Run ``group_size`` episodes, resetting with ``episode_seed`` each time."""
        rollouts: list[EpisodeRollout] = []
        instruction = ""
        for _ in range(self.group_size):
            observation, info = self._reset(episode_seed)
            instruction = self._resolve_instruction(info, instruction)
            transitions: list[Transition] = []
            terminated = False
            truncated = False

            for _ in range(self.max_steps):
                chunk, value = self._act(observation, instruction)
                action = (
                    chunk.actions.cpu().numpy() if torch.is_tensor(chunk.actions) else chunk.actions
                )
                result = self.env.step(action)
                observation, reward, terminated, truncated, info = _unpack_step(result)
                done = terminated or truncated
                transitions.append(
                    Transition(
                        observation=observation,
                        instruction=instruction,
                        action_chunk=chunk,
                        reward=reward,
                        done=done,
                        info=info,
                        value=value,
                    )
                )
                if done:
                    break

            if len(transitions) >= self.max_steps and not (terminated or truncated):
                # Treat the step budget as truncation, including for GAE bootstrapping.
                truncated = True
                transitions[-1].done = True

            rollouts.append(
                EpisodeRollout(
                    transitions=transitions,
                    episode_seed=episode_seed,
                    group_id=group_id,
                    terminated=terminated,
                    truncated=truncated,
                )
            )

        group = GroupRollout(
            group_id=group_id,
            rollouts=rollouts,
            episode_seed=episode_seed,
        )
        if self.normalize_advantages and rollouts:
            group.advantages = group_advantages(
                group.rewards, torch.zeros(len(rollouts), dtype=torch.long)
            )
        return group
