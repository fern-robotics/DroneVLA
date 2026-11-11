#!/usr/bin/env python3
"""Thin UAV rollout adapter.

The policy (OpenVLA / pi0-FAST server) only sees:
  RGB image, instruction, UAV state, verified progress.

This adapter is deliberately small and homebrew: it is the only place that
knows about AirSim observations, action chunks, rewards and JSONL logging.
Training algorithms and SFT stay in the official repositories.

The `mock` environment is for pipeline smoke tests without a simulator.
"""
import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MockEnv:
    """Minimal approach task used to validate the adapter and logging."""
    marker_x: float = 10.0
    marker_y: float = 0.0
    stop_distance: float = 3.5
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, -10.0])
    yaw: float = 0.0
    collision: bool = False
    steps: int = 0

    def observe(self) -> dict[str, Any]:
        dist = math.hypot(self.pos[0] - self.marker_x, self.pos[1] - self.marker_y)
        return {
            "position": list(self.pos),
            "yaw": self.yaw,
            "distance": dist,
            "image": None,  # mock has no pixels
            "collision": self.collision,
        }

    def step(self, action: list[float], dt: float = 0.5) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        # action = [dx, dy, dz, dyaw] in metres/radians per dt
        self.pos[0] += action[0]
        self.pos[1] += action[1]
        self.pos[2] += action[2]
        self.yaw += action[3]
        self.steps += 1
        obs = self.observe()
        success = obs["distance"] <= self.stop_distance
        reward = (1.0 if success else 0.0) - 0.01 * self.steps
        return obs, reward, success, {"success": success, "terminal": "SUCCESS" if success else "NONE"}


class MockPolicy:
    """Fixed forward policy; stands in for a VLA server in smoke tests."""
    name = "mock-forward"

    def infer(self, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
        t0 = time.time()
        return {
            "actions": [[0.18, 0.0, 0.0, 0.0] for _ in range(8)],
            "tokens": [[1, 2, 3, 4] for _ in range(8)],
            "logprobs": [[-0.1, -0.2, -0.3, -0.4] for _ in range(8)],
            "policy_latency_ms": (time.time() - t0) * 1000.0,
        }


def run_episode(env: Any, policy: Any, instruction: str, out_dir: Path, episode_id: str,
                max_steps: int = 100, replan_steps: int = 4, dt: float = 0.5) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{episode_id}.jsonl"
    total_reward = 0.0
    success = False
    terminal = "MAX_STEPS"
    with log_path.open("w") as log:
        for step in range(max_steps):
            obs = env.observe()
            result = policy.infer(obs, instruction)
            actions = result["actions"][:replan_steps]
            for a in actions:
                if obs["distance"] <= env.stop_distance:
                    success = True
                    terminal = "SUCCESS"
                    break
                obs, reward, done, info = env.step(list(a), dt)
                total_reward += reward
                record = {
                    "episode_id": episode_id,
                    "step": step,
                    "instruction": instruction,
                    "state": {"position": obs["position"], "yaw": obs["yaw"], "distance": obs["distance"]},
                    "action": list(a),
                    "tokens": result.get("tokens", [None])[0] if result.get("tokens") else None,
                    "logprobs": result.get("logprobs", [None])[0] if result.get("logprobs") else None,
                    "reward": reward,
                    "cumulative_reward": total_reward,
                    "success": success,
                    "terminal": info.get("terminal", "NONE"),
                    "policy_latency_ms": result.get("policy_latency_ms"),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                if done:
                    success = True
                    terminal = "SUCCESS"
                    break
            if success:
                break
    summary = {
        "episode_id": episode_id,
        "instruction": instruction,
        "success": success,
        "terminal": terminal,
        "steps": env.steps,
        "cumulative_reward": total_reward,
        "log": str(log_path),
    }
    (out_dir / f"{episode_id}.summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/adapter-smoke")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out)
    summaries = []
    for i in range(args.episodes):
        env = MockEnv()
        summaries.append(run_episode(env, MockPolicy(), "Move to the red circle on the front side",
                                     out, f"mock-{i:02d}", max_steps=args.max_steps))
    (out / "summaries.json").write_text(json.dumps(summaries, indent=2))
    for s in summaries:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
