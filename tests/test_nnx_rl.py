import jax
import jax.numpy as jnp
import optax
from flax import nnx

from integrations.openpi.nnx_rl import make_nnx_grpo_step


class Toy(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jnp.zeros(()))

    def __call__(self, x):
        return self.w[...] * x


def test_nnx_step_updates_param():
    model = Toy()
    optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)

    def loss_fn(model, batch):
        x, y = batch
        return jnp.mean((model(x) - y) ** 2)

    step = make_nnx_grpo_step(model, optimizer, loss_fn)
    before = model.w[...]
    loss = step(model, optimizer, (jnp.array([1.0]), jnp.array([1.0])))
    assert jnp.isfinite(loss)
    assert not jnp.allclose(model.w[...], before)
