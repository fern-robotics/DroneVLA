import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.openpi.pi0_rl import grpo_loss, make_grpo_update, make_ppo_update


def toy_score(params, tokens, contexts):
    return jax.nn.log_softmax(params, axis=-1)[tokens]


def test_grpo_update_and_loss():
    params = jnp.zeros(8)
    tokens = jnp.array([[1, 2], [3, 4]])
    old = toy_score(params, tokens, None)
    adv = jnp.array([1.0, -1.0])
    mask = jnp.ones_like(old)
    opt = optax.sgd(0.1)
    state = opt.init(params)
    step = make_grpo_update(toy_score, opt)
    new, _, loss = step(params, state, tokens, old, adv, mask, None, old)
    assert jnp.isfinite(loss)
    assert not jnp.allclose(new, params)


def test_ppo_update():
    params = jnp.zeros(8)
    tokens = jnp.array([[1, 2]])
    old = toy_score(params, tokens, None)
    mask = jnp.ones_like(old)
    opt = optax.sgd(0.1)
    step = make_ppo_update(toy_score, opt)
    new, _, loss = step(params, opt.init(params), tokens, old, jnp.array([1.0]), mask, jnp.array([0.0]), jnp.array([1.0]), None)
    assert jnp.isfinite(loss)
    assert not jnp.allclose(new, params)
