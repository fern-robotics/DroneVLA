"""Self-contained PyTorch RL core for DroneVL (CPU-tested, model-agnostic)."""

from .grpo import GrpoLoss, group_advantages, grpo_loss
from .ppo import PpoLoss, compute_gae, ppo_loss
from .rollout import RolloutCollector
from .types import EpisodeRollout, GroupRollout, TokenActionChunk, Transition

__all__ = [
    "EpisodeRollout",
    "GroupRollout",
    "GrpoLoss",
    "PpoLoss",
    "RolloutCollector",
    "TokenActionChunk",
    "Transition",
    "compute_gae",
    "group_advantages",
    "grpo_loss",
    "ppo_loss",
]
