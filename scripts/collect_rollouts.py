#!/usr/bin/env python3
"""Collect GRPO/PPO rollout groups through the thin UAV adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from scripts.eval.evaluate import MockEnv, ScriptedPolicy
from scripts.eval.intents import evaluate_episode
from scripts.eval.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from scripts.eval.vla_policy import make_policy
from training.rl.rollout import RolloutCollector


class EpisodeEnv:
    """Adapt one manifest episode to the RolloutCollector reset/step API."""

    def __init__(self, episode: dict, env: object, max_steps: int = 200) -> None:
        self.episode = episode
        self.env = env
        self.max_steps = max_steps
        self.steps = 0
        self.instruction = episode["instruction"]
        self.trajectory: list[dict] = []

    def reset(self, seed: int | None = None):
        self.steps = 0
        self.trajectory = []
        obs = self.env.reset(self.episode["start_pose"])
        return {**obs, "instruction": self.instruction}, {"instruction": self.instruction}

    def step(self, action):
        obs, info = self.env.step(action)
        self.steps += 1
        self.trajectory.append({**obs, **info})
        done = bool(info.get("collision")) or self.steps >= self.max_steps
        reward = 0.0
        if done:
            evaluation = evaluate_episode(self.episode, self.trajectory, dt_s=0.2)
            reward = float(evaluation.success) + 0.5 * float(evaluation.intent_score) - 1.0 * float(evaluation.collision)
        return {**obs, "instruction": self.instruction}, reward, done, False, info


class TorchPolicyAdapter:
    """Require a policy server that returns action tokens and log-probabilities."""

    def __init__(self, policy: object) -> None:
        self.policy = policy

    def act(self, observation, instruction):
        out = self.policy.act(observation, instruction)
        if out.get("tokens") is None or out.get("logprobs") is None:
            raise RuntimeError(
                "RL collection requires a policy server that returns token ids and log-probabilities; "
                "evaluation-only servers returning actions alone are insufficient"
            )
        return {
            "actions": np.asarray(out["actions"], dtype=np.float32),
            "tokens": torch.as_tensor(out["tokens"], dtype=torch.long),
            "logprobs": torch.as_tensor(out["logprobs"], dtype=torch.float32),
            "value": out.get("value"),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    ap.add_argument("--out", default="runs/rollouts")
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--policy", choices=["mock", "websocket", "http"], default="mock")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--policy-url", default="")
    ap.add_argument("--env", choices=["mock", "airsim"], default="mock")
    args = ap.parse_args()
    manifest = load_manifest(Path(args.manifest))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for group_index in range(args.groups):
        episode = manifest["episodes"][group_index % len(manifest["episodes"])]
        base = MockEnv() if args.env == "mock" else None
        if base is None:
            from scripts.eval.airsim_env import AirSimEnv
            base = AirSimEnv()
        env = EpisodeEnv(episode, base)
        scripted = ScriptedPolicy(episode)
        policy = make_policy(args.policy, scripted=scripted, host=args.host, port=args.port, url=args.policy_url)
        collector = RolloutCollector(env, TorchPolicyAdapter(policy), group_size=args.group_size,
                                     max_steps=int(episode.get("max_steps", 200)))
        group = collector.collect_group(episode_seed=group_index, group_id=group_index)
        rows.append({
            "group_id": group.group_id,
            "episode_id": episode["episode_id"],
            "rewards": [r.total_reward for r in group.rollouts],
            "success": [r.success for r in group.rollouts],
            "advantages": group.advantages.tolist() if group.advantages is not None else None,
            "decisions": [r.length for r in group.rollouts],
        })
        print(json.dumps(rows[-1]), flush=True)
    (out / "groups.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
