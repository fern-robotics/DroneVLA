import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.rl.trainer import collate_groups, grpo_update, ppo_update
from training.rl.types import EpisodeRollout, GroupRollout, TokenActionChunk, Transition


class ToyPolicy(torch.nn.Module):
    def __init__(self, vocab=8):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(vocab))

    def score_tokens(self, tokens, contexts):
        return torch.log_softmax(self.logits, dim=-1)[tokens]

    def values(self, contexts):
        return torch.zeros(len(contexts), requires_grad=True)


def make_group(rewards=(1.0, 0.0)):
    rollouts = []
    for r in rewards:
        transitions = [
            Transition(observation={"x": 0}, instruction="go", action_chunk=TokenActionChunk(
                actions=torch.zeros(4), tokens=torch.tensor([1, 2, 3]), logprobs=torch.tensor([-1.0, -1.0, -1.0])),
                reward=r, done=True)
        ]
        rollouts.append(EpisodeRollout(transitions=transitions, terminated=True))
    group = GroupRollout(group_id=0, rollouts=rollouts)
    group.advantages = torch.tensor([1.0, -1.0])
    return group


def test_collate_and_grpo_update():
    batch = collate_groups([make_group()])
    assert batch.tokens.shape == (2, 3)
    policy = ToyPolicy()
    opt = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = policy.logits.detach().clone()
    out = grpo_update(policy, opt, [make_group()])
    assert torch.isfinite(out.loss)
    assert not torch.equal(before, policy.logits.detach())


def test_ppo_update():
    policy = ToyPolicy()
    opt = torch.optim.SGD(policy.parameters(), lr=0.1)
    out = ppo_update(policy, opt, [make_group()])
    assert torch.isfinite(out.loss)
    assert out.value_loss.ndim == 0
