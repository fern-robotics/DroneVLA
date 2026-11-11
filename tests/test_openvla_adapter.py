import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.rl.adapters.openvla import build_openvla_context


class DummyProcessor:
    def __call__(self, prompt, image):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]]),
                "pixel_values": torch.zeros(1, 3, 224, 224)}


def test_context_shapes():
    ctx = build_openvla_context(DummyProcessor(), None, "Move forward")
    assert ctx.input_ids.shape == (4,)
    assert ctx.pixel_values.shape == (3, 224, 224)
    assert ctx.action_start == 5
