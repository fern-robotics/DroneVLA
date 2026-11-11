"""CPU-only tests for the self-contained DroneVL RL core."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.rl import (  # noqa: E402
    RolloutCollector,
    TokenActionChunk,
    compute_gae,
    group_advantages,
    grpo_loss,
    ppo_loss,
)


def test_group_advantages_normalizes_each_group():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    advantages = group_advantages(rewards, group_ids)
    torch.testing.assert_close(advantages, torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-5, rtol=1e-5)

    # Unequal group sizes and a single-member group.
    advantages = group_advantages(
        torch.tensor([0.0, 10.0, 100.0]), torch.tensor([0, 0, 1])
    )
    torch.testing.assert_close(advantages, torch.tensor([-1.0, 1.0, 0.0]), atol=1e-5, rtol=1e-5)

    # A zero-variance group must not produce NaNs.
    advantages = group_advantages(torch.tensor([2.0, 2.0, 2.0]), torch.tensor([4, 4, 4]))
    torch.testing.assert_close(advantages, torch.zeros(3))


def test_grpo_ratio_clipping():
    old = torch.zeros(1, 3)
    current = torch.log(torch.tensor([[2.0, 2.0, 0.9]]))
    advantages = torch.tensor([[1.0, -1.0, 3.0]])
    mask = torch.ones(1, 3)

    out = grpo_loss(current, old, advantages, mask, clip_range=0.2)
    # min(2*1, 1.2*1)=1.2; min(2*-1, 1.2*-1)=-2; min(0.9*3, 0.9*3)=2.7
    expected = -torch.tensor([1.2, -2.0, 2.7]).mean()
    torch.testing.assert_close(out.loss, expected)
    torch.testing.assert_close(out.policy_loss, expected)
    torch.testing.assert_close(out.ratio_mean, torch.tensor((2.0 + 2.0 + 0.9) / 3.0))
    torch.testing.assert_close(out.clip_fraction, torch.tensor(2.0 / 3.0))

    # Lower-bound clipping is active when the advantage is negative.
    lower = grpo_loss(
        torch.log(torch.tensor([[0.5]])), torch.zeros(1, 1), torch.tensor([[-1.0]]), torch.ones(1, 1)
    )
    torch.testing.assert_close(lower.loss, torch.tensor(0.8))  # min(-0.5, -0.8)


def test_grpo_ignores_masked_tokens_and_is_nan_safe():
    old = torch.zeros(1, 3)
    current = torch.log(torch.tensor([[2.0, 2.0, 0.9]])).requires_grad_(True)
    advantages = torch.tensor([[1.0, -1.0, 3.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    out = grpo_loss(current, old, advantages, mask)
    torch.testing.assert_close(out.loss, torch.tensor(0.4))  # -(1.2 - 2.0) / 2

    # Changing a masked token must not change the loss or receive gradient.
    changed = current.detach().clone().requires_grad_(True)
    with torch.no_grad():
        changed[0, 2] = torch.log(torch.tensor(100.0))
    changed_out = grpo_loss(changed, old, advantages, mask)
    torch.testing.assert_close(changed_out.loss, out.loss)
    changed_out.loss.backward()
    torch.testing.assert_close(changed.grad[0, 2], torch.tensor(0.0))

    # An all-zero mask yields a finite zero loss, not NaN.
    empty = grpo_loss(current, old, advantages, torch.zeros(1, 3))
    assert torch.isfinite(empty.loss)
    torch.testing.assert_close(empty.loss, torch.tensor(0.0))


def test_grpo_reference_kl_and_per_rollout_advantages():
    current = torch.zeros(2, 3, requires_grad=True)
    old = torch.zeros(2, 3)
    advantages = torch.tensor([1.0, -1.0])  # broadcast over each rollout's tokens
    mask = torch.ones(2, 3)
    ref = torch.full((2, 3), float(torch.log(torch.tensor(0.5))))

    out = grpo_loss(current, old, advantages, mask, beta=0.5, ref_logprobs=ref)
    expected_kl = 0.5 + float(torch.log(torch.tensor(2.0))) - 1.0
    torch.testing.assert_close(out.kl, torch.tensor(expected_kl), atol=1e-6, rtol=1e-6)
    expected_loss = -((1.0 - 1.0) / 2.0) + 0.5 * expected_kl
    torch.testing.assert_close(out.loss, torch.tensor(expected_loss), atol=1e-6, rtol=1e-6)
    # Without ref_logprobs, beta > 0 falls back to KL(old || current).
    fallback = grpo_loss(current.detach(), old, advantages, mask, beta=0.5)
    torch.testing.assert_close(fallback.kl, torch.tensor(0.0))
    out.loss.backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()


def test_ppo_loss_shapes_and_value_loss():
    # Zero advantage and no entropy: loss is only the weighted value loss.
    current = torch.zeros(2, 2)
    old = torch.zeros(2, 2)
    mask = torch.ones(2, 2)
    values = torch.tensor([1.0, 2.0], requires_grad=True)
    out = ppo_loss(
        current, old, torch.zeros(2), mask, values, torch.zeros(2), value_coef=0.5
    )
    torch.testing.assert_close(out.policy_loss, torch.tensor(0.0))
    torch.testing.assert_close(out.value_loss, torch.tensor(1.25))
    torch.testing.assert_close(out.loss, torch.tensor(0.625))
    assert out.loss.ndim == 0

    # Full loss with entropy, token masks, and value clipping.
    current = torch.zeros(2, 4, requires_grad=True)
    old = torch.zeros(2, 4)
    advantages = torch.tensor([[1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]])
    token_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    entropy = torch.rand(2, 4)
    values = torch.tensor([0.5, 0.7], requires_grad=True)
    out = ppo_loss(
        current,
        old,
        advantages,
        token_mask,
        values,
        torch.tensor([1.0, 0.0]),
        entropy=entropy,
        old_values=values.detach(),
        value_clip_range=0.2,
        entropy_coef=0.01,
    )
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert out.policy_loss.ndim == 0 and out.value_loss.ndim == 0 and out.entropy.ndim == 0
    out.loss.backward()
    assert current.grad is not None and values.grad is not None

    # All-masked response tokens: the actor contribution is zero.
    masked = ppo_loss(
        current.detach(),
        old,
        advantages,
        torch.zeros(2, 4),
        torch.tensor([0.5, 0.7]),
        torch.tensor([1.0, 0.0]),
        entropy_coef=0.0,
        value_coef=0.0,
    )
    torch.testing.assert_close(masked.policy_loss, torch.tensor(0.0))


def test_gae_matches_hand_computed_example():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 1.0])
    advantages, returns = compute_gae(rewards, values, dones, gamma=0.9, lam=0.8)

    # t=2: delta=0.5, A=0.5, R=1.0
    # t=1: delta=0.95, A=0.95+0.72*0.5=1.31, R=1.81
    # t=0: delta=0.95, A=0.95+0.72*1.31=1.8932, R=2.3932
    torch.testing.assert_close(advantages, torch.tensor([1.8932, 1.31, 0.5]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(returns, torch.tensor([2.3932, 1.81, 1.0]), atol=1e-6, rtol=1e-6)

    # A done in the middle blocks bootstrapping from the next episode.
    advantages, returns = compute_gae(
        torch.tensor([1.0, 1.0]), torch.tensor([0.5, 0.5]), torch.tensor([1.0, 0.0]), 0.9, 0.8
    )
    torch.testing.assert_close(advantages, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(returns, torch.tensor([1.0, 1.0]))


class _MockEnv:
    """Deterministic two-step task; reward equals the first action component."""

    def __init__(self):
        self.reset_seeds: list[int | None] = []
        self.t = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.t = 0
        return {"t": 0, "seed": seed}, {"instruction": "mock-instruction"}

    def step(self, action):
        self.t += 1
        reward = float(action[0])
        done = self.t >= 2
        return {"t": self.t, "seed": None}, reward, done, {"success": done}


class _MockPolicy:
    """Varies the action per call so rewards differ within a group."""

    def __init__(self):
        self.calls = 0
        self.instructions: list[str] = []

    def act(self, observation, instruction):
        self.calls += 1
        self.instructions.append(instruction)
        action = float(self.calls % 3 + 1)
        return {
            "actions": torch.tensor([action, 0.0, 0.0, 0.0]),
            "tokens": torch.tensor([10, 11, 12]),
            "logprobs": torch.tensor([-0.1, -0.2, -0.3]) * action,
            "value": 0.25,
        }


def test_rollout_group_consistency_and_logging():
    env = _MockEnv()
    policy = _MockPolicy()
    collector = RolloutCollector(env, policy, group_size=2, max_steps=2)
    group = collector.collect_group(episode_seed=7, group_id=3)

    assert group.group_id == 3
    assert group.episode_seed == 7
    assert group.num_rollouts == 2
    assert env.reset_seeds == [7, 7]  # same episode seed for every group member
    assert policy.calls == 4
    assert policy.instructions == ["mock-instruction"] * 4

    observation_sequences = []
    for rollout in group.rollouts:
        assert rollout.group_id == 3
        assert rollout.episode_seed == 7
        assert len(rollout.transitions) == 2  # every decision is logged
        assert rollout.length == 2
        observation_sequences.append([t.observation for t in rollout.transitions])
        for transition in rollout.transitions:
            assert isinstance(transition.action_chunk, TokenActionChunk)
            assert transition.action_chunk.tokens.tolist() == [10, 11, 12]
            assert transition.action_chunk.logprobs.shape == (3,)
            assert transition.instruction == "mock-instruction"
            assert transition.value == 0.25
        torch.testing.assert_close(
            rollout.rewards, torch.tensor([2.0, 3.0] if rollout is group.rollouts[0] else [1.0, 2.0])
        )

    # Same seed => same environment state trajectory across the group.
    assert observation_sequences[0] == observation_sequences[1]
    torch.testing.assert_close(group.rewards, torch.tensor([5.0, 3.0]))
    torch.testing.assert_close(group.advantages, torch.tensor([1.0, -1.0]), atol=1e-5, rtol=1e-5)

    token_data = group.rollouts[0].token_data()
    assert token_data["tokens"].shape == (6,)
    assert token_data["logprobs"].shape == (6,)
    torch.testing.assert_close(token_data["response_mask"], torch.ones(6))
