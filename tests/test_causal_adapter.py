import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.rl.adapters.causal import CausalVLMTokenPolicy, VLMContext


class DummyVLM(torch.nn.Module):
    def __init__(self, vocab=16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 8)
        self.head = torch.nn.Linear(8, vocab)

    def forward(self, input_ids, pixel_values=None):
        return type("Out", (), {"logits": self.head(self.embed(input_ids))})()


def test_score_tokens_and_values():
    policy = CausalVLMTokenPolicy(DummyVLM())
    ctx = [VLMContext(input_ids=torch.tensor([1, 2, 3]), pixel_values=None, action_start=4)]
    tokens = torch.tensor([[4, 5, 6]])
    logprobs = policy.score_tokens(tokens, ctx)
    assert logprobs.shape == (1, 3)
    assert torch.isfinite(logprobs).all()
    assert (logprobs <= 0).all()
    values = policy.values(ctx)
    assert values.shape == (1,)
