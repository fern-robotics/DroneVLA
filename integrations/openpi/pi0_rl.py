"""JAX GRPO/PPO optimizer core for pi0-FAST.

OpenPI is JAX, so the PyTorch trainer cannot update it directly. A model
adapter supplies ``score_tokens(params, tokens, contexts) -> logprobs``; this
module owns the clipped objective and the optimizer step.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence

import jax
import jax.numpy as jnp
import optax


class TokenScorer(Protocol):
    def __call__(self, params: Any, tokens: jax.Array, contexts: Sequence[Any]) -> jax.Array: ...


def _masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    return (values * mask).sum() / jnp.clip(mask.sum(), 1.0)


def grpo_loss(
    current_logprobs: jax.Array,
    old_logprobs: jax.Array,
    advantages: jax.Array,
    response_mask: jax.Array,
    *,
    clip_range: float = 0.2,
    beta: float = 0.0,
    ref_logprobs: jax.Array | None = None,
) -> jax.Array:
    """Clipped GRPO loss over ``(N, L)`` token log-probs."""
    mask = (response_mask > 0).astype(jnp.float32)
    ratio = jnp.exp(jnp.clip(current_logprobs - old_logprobs, -20.0, 20.0))
    clipped = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
    per_token = jnp.minimum(ratio * advantages[:, None], clipped * advantages[:, None])
    policy = -_masked_mean(per_token, mask)
    if beta <= 0.0:
        return policy
    ref = old_logprobs if ref_logprobs is None else ref_logprobs
    log_ratio = jnp.clip(ref - current_logprobs, -20.0, 20.0)
    kl = _masked_mean(jnp.expm1(log_ratio) - log_ratio, mask)
    return policy + beta * kl


def ppo_loss(
    current_logprobs: jax.Array,
    old_logprobs: jax.Array,
    advantages: jax.Array,
    response_mask: jax.Array,
    values: jax.Array,
    returns: jax.Array,
    *,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
) -> jax.Array:
    mask = (response_mask > 0).astype(jnp.float32)
    ratio = jnp.exp(jnp.clip(current_logprobs - old_logprobs, -20.0, 20.0))
    clipped = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
    policy = -_masked_mean(jnp.minimum(ratio * advantages[:, None], clipped * advantages[:, None]), mask)
    value = 0.5 * jnp.mean((values - returns) ** 2)
    return policy + value_coef * value


def make_grpo_update(score_tokens: TokenScorer, optimizer: optax.GradientTransformation):
    """Return ``step(params, tokens, old, advantages, mask, contexts)``."""
    def loss_fn(params, tokens, old, advantages, mask, contexts, ref):
        current = score_tokens(params, tokens, contexts)
        return grpo_loss(current, old, advantages, mask, beta=0.0, ref_logprobs=ref)

    @jax.jit
    def step(params, opt_state, tokens, old, advantages, mask, contexts, ref):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens, old, advantages, mask, contexts, ref)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return step


def make_ppo_update(score_tokens: TokenScorer, optimizer: optax.GradientTransformation):
    def loss_fn(params, tokens, old, advantages, mask, values, returns, contexts):
        current = score_tokens(params, tokens, contexts)
        return ppo_loss(current, old, advantages, mask, values, returns)

    @jax.jit
    def step(params, opt_state, tokens, old, advantages, mask, values, returns, contexts):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens, old, advantages, mask, values, returns, contexts)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return step
