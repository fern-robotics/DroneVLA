"""Reference-trajectory reward for native UAV-Flow rollouts.

Coordinates in task JSON and native logs are centimetres. The returned terms are
in metres so they are comparable across task lengths.
"""
from __future__ import annotations
import math
from typing import Any, Sequence

import numpy as np

from .tasks import reference_path_cm


def rollout_reward(world_xyz: Sequence[Sequence[float]], task: dict[str, Any]) -> dict[str, float]:
    pred = np.asarray(world_xyz, dtype=float)
    ref = np.asarray(reference_path_cm(task), dtype=float)
    if pred.size == 0 or ref.size == 0:
        return {"reward": -10.0, "progress": 0.0, "cross_track_m": 10.0, "terminal_m": 10.0}
    d = np.linalg.norm(pred[:, None, :] - ref[None, :, :], axis=-1)
    nn = d.argmin(axis=1)
    progress = float(nn.max()) / max(1, len(ref) - 1)
    cross_track_m = float(d.min(axis=1).mean()) / 100.0
    terminal_m = float(np.linalg.norm(pred[-1] - ref[-1])) / 100.0
    reward = 2.0 * progress - 0.5 * cross_track_m - 1.0 * terminal_m
    return {
        "reward": float(reward),
        "progress": progress,
        "cross_track_m": cross_track_m,
        "terminal_m": terminal_m,
    }
