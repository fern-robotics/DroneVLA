"""π₀-FAST GRPO/PPO batch adapter for the NNX optimizer.

Rollouts store an OpenPI ``Observation`` containing the prompt and the sampled
FAST action tokens. ``batch_grpo_loss`` asks the model for teacher-forced token
log-probs and applies the clipped group objective. The actual update is owned by
``nnx_rl.make_nnx_grpo_step``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from .pi0_rl import grpo_loss, ppo_loss


@dataclass
class TokenRollout:
    observation: Any
    old_logprobs: jax.Array
    advantage: float
    ref_logprobs: jax.Array | None = None
    values: float = 0.0
    returns: float = 0.0


def _pad_rows(rows: Sequence[jax.Array]) -> tuple[jax.Array, jax.Array]:
    width = max((int(r.shape[-1]) for r in rows), default=1)
    out = jnp.zeros((len(rows), width), dtype=jnp.float32)
    mask = jnp.zeros((len(rows), width), dtype=jnp.float32)
    for i, row in enumerate(rows):
        n = int(row.shape[-1])
        out = out.at[i, :n].set(jnp.asarray(row, dtype=jnp.float32))
        mask = mask.at[i, :n].set(1.0)
    return out, mask


def batch_grpo_loss(model: Any, rollouts: Sequence[TokenRollout]) -> jax.Array:
    current, masks = [], []
    for r in rollouts:
        logp, mask = model.score_token_logprobs(None, r.observation)
        current.append(logp.reshape(-1))
        masks.append(mask.reshape(-1))
    cur, _ = _pad_rows(current)
    _, mask = _pad_rows(masks)
    old, _ = _pad_rows([r.old_logprobs for r in rollouts])
    ref_rows = [r.ref_logprobs for r in rollouts if r.ref_logprobs is not None]
    ref = None
    if len(ref_rows) == len(rollouts):
        ref, _ = _pad_rows(ref_rows)
    adv = jnp.asarray([r.advantage for r in rollouts], dtype=jnp.float32)
    return grpo_loss(cur, old, adv, mask, ref_logprobs=ref)


def batch_ppo_loss(model: Any, rollouts: Sequence[TokenRollout]) -> jax.Array:
    current, masks = [], []
    for r in rollouts:
        logp, mask = model.score_token_logprobs(None, r.observation)
        current.append(logp.reshape(-1))
        masks.append(mask.reshape(-1))
    cur, _ = _pad_rows(current)
    _, mask = _pad_rows(masks)
    old, _ = _pad_rows([r.old_logprobs for r in rollouts])
    adv = jnp.asarray([r.advantage for r in rollouts], dtype=jnp.float32)
    values = jnp.asarray([r.values for r in rollouts], dtype=jnp.float32)
    returns = jnp.asarray([r.returns for r in rollouts], dtype=jnp.float32)
    return ppo_loss(cur, old, adv, mask, values, returns)
