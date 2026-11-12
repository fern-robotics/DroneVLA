import jax.numpy as jnp

from integrations.openpi.pi0_nnx_trainer import TokenRollout, batch_grpo_loss, batch_ppo_loss


class FakeModel:
    def __init__(self, width=3):
        self.width = width

    def score_token_logprobs(self, rng, observation):
        return jnp.zeros((1, self.width)), jnp.ones((1, self.width))


def test_batch_losses_finite():
    m = FakeModel()
    rollouts = [
        TokenRollout(observation=None, old_logprobs=jnp.zeros(3), advantage=1.0),
        TokenRollout(observation=None, old_logprobs=jnp.zeros(2), advantage=-1.0, values=0.1, returns=0.2),
    ]
    assert jnp.isfinite(batch_grpo_loss(m, rollouts))
    assert jnp.isfinite(batch_ppo_loss(m, rollouts))
