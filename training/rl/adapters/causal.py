"""Token-scoring adapter for OpenVLA-style causal vision-language models.

The RL trainer owns the algorithm; this adapter owns the model-specific forward
pass. A context carries the prompt token ids, optional pixel values, and the
position where action tokens begin. The adapter concatenates the sampled action
tokens, runs one forward pass, and returns log-probabilities at the action
positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass
class VLMContext:
    input_ids: torch.Tensor          # (Lp,)
    pixel_values: torch.Tensor | None
    action_start: int                # action token i is predicted by position action_start+i-1


class CausalVLMTokenPolicy(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.value_bias = torch.nn.Parameter(torch.zeros(()))

    def score_tokens(self, tokens: torch.Tensor, contexts: Sequence[Any]) -> torch.Tensor:
        if len(contexts) != tokens.shape[0]:
            raise ValueError("one VLMContext per rollout row is required")
        rows = []
        for row, ctx in zip(tokens, contexts):
            if not isinstance(ctx, VLMContext):
                raise TypeError("contexts must be VLMContext instances")
            ids = torch.cat([ctx.input_ids.to(tokens.device), row])
            kwargs = {"input_ids": ids.unsqueeze(0)}
            if ctx.pixel_values is not None:
                kwargs["pixel_values"] = ctx.pixel_values.unsqueeze(0).to(tokens.device)
            logits = self.model(**kwargs).logits[0]
            start = int(ctx.action_start) - 1
            end = start + int(row.numel())
            if start < 0 or end > logits.shape[0]:
                raise ValueError("action_start does not fit the prompt/action sequence")
            selected = logits[start:end]
            rows.append(torch.log_softmax(selected, dim=-1).gather(-1, row[:, None]).squeeze(-1))
        return torch.stack(rows)

    def values(self, contexts: Sequence[Any]) -> torch.Tensor:
        # A trained critic head can replace this bias; it keeps the PPO path
        # differentiable and shape-correct for the generic trainer.
        return self.value_bias.expand(len(contexts))
