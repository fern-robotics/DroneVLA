"""NNX optimizer steps for the π₀-FAST JAX GRPO/PPO trainer.

OpenPI models are Flax NNX modules, so the pure-array ``pi0_rl`` update cannot
own the parameters. This wrapper hands the model to ``nnx.value_and_grad`` and
lets ``nnx.Optimizer`` update the trainable LoRA parameters.
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import optax
from flax import nnx


def make_nnx_grpo_step(model: nnx.Module, optimizer: nnx.Optimizer, loss_fn: Callable[[Any, Any], jax.Array]):
    @nnx.jit
    def step(model, optimizer, batch):
        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model, batch)
        optimizer.update(model, grads)
        return loss

    return step


def make_nnx_ppo_step(model: nnx.Module, optimizer: nnx.Optimizer, loss_fn: Callable[[Any, Any], jax.Array]):
    @nnx.jit
    def step(model, optimizer, batch):
        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model, batch)
        optimizer.update(model, grads)
        return loss

    return step


def build_optimizer(model: nnx.Module, learning_rate: float, clip_norm: float = 1.0) -> nnx.Optimizer:
    tx = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)
