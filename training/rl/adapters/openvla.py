"""OpenVLA prompt/image -> causal token-scoring context adapter."""
from __future__ import annotations

from typing import Any

import torch

from .causal import VLMContext


def build_openvla_context(processor: Any, image: Any, instruction: str) -> VLMContext:
    """Build a VLMContext for an OpenVLA-style processor.

    The prompt format matches the official OpenVLA inference path. The action
    tokens are appended after the prompt by the trainer.
    """
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, image)
    input_ids = inputs["input_ids"]
    if input_ids.ndim == 2:
        input_ids = input_ids[0]
    input_ids = input_ids.to(dtype=torch.long)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None and pixel_values.ndim == 4:
        pixel_values = pixel_values[0]
    return VLMContext(input_ids=input_ids, pixel_values=pixel_values, action_start=int(input_ids.numel()) + 1)
