#!/usr/bin/env python3
"""Evaluate a policy on the frozen 30-episode AirSimNH UAV intent suite.

The evaluator is deliberately split into three pieces:

* ``MockEnv`` -- a deterministic kinematic point-mass environment for tests.
* ``ScriptedPolicy`` -- a deterministic waypoint policy derived from the
  manifest's success predicates.  It exists to exercise the evaluator and
  produce a reproducible smoke-test baseline; it is not a VLA.
* ``AirSimEnv`` (``scripts/eval/airsim_env.py``) -- the optional thin AirSim
  adapter used by ``--airsim``.  Importing/evaluating with ``--mock`` never
  imports the ``airsim`` package.

Outputs:
  <out>/episodes/<episode_id>.jsonl   one observed step per line
  <out>/per_episode.jsonl             one evaluated episode per line
  <out>/summary.csv                   flat per-episode table
  <out>/summary.json                  aggregate + per-episode summary

Usage::

    python -m scripts.eval.evaluate --mock
    python -m scripts.eval.evaluate --mock --episode-id id-approach-01
    python -m scripts.eval.evaluate --airsim --out runs/eval/airsim
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Allow both ``python -m scripts.eval.evaluate`` and
# ``python scripts/eval/evaluate.py``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.intents import evaluate_episode  # noqa: E402
from scripts.eval.manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    load_manifest,
)

ZERO_ACTION = (0.0, 0.0, 0.0, 0.0)


def _wrap_angle(angle: float) -> float:
    value = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if value <= -math.pi:
        value += 2.0 * math.pi
    return value


def _as_action(action: Any) -> np.ndarray:
    if isinstance(action, dict):
        values = [
            action.get("dx", 0.0),
            action.get("dy", 0.0),
            action.get("dz", 0.0),
            action.get("dyaw", 0.0),
        ]
    else:
        values = list(action)
    if len(values) != 4:
        raise ValueError("action must have four components [dx, dy, dz, dyaw]")
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("action contains non-finite values")
    return arr


def _action_list(action: Any) -> list[float]:
    try:
        return [float(v) for v in _as_action(action)]
    except (TypeError, ValueError):
        return [float("nan")] * 4


def _marker_position(episode: dict[str, Any], marker_id: str) -> np.ndarray:
    for marker in episode.get("markers", []):
        if marker.get("marker_id") == marker_id:
            return np.asarray(marker["position_ned_m"], dtype=float)
    raise KeyError(f"{episode['episode_id']}: unknown marker {marker_id!r}")


class MockEnv:
    """Deterministic point-mass mock environment with a simple safety gate.

    Actions are ``[dx, dy, dz, dyaw]`` in metres/radians per step.  Translation
    is clipped to ``max_step_m`` and yaw to ``max_yaw_step_rad``; a clip is
    counted as a safety intervention.  A malformed or non-finite action is
    logged as an invalid action and produces no motion.
    """

    def __init__(
        self,
        markers: Sequence[dict[str, Any]] | None = None,
        dt_s: float = 0.2,
        max_step_m: float = 0.5,
        max_yaw_step_rad: float = 0.35,
        collision_radius_m: float = 0.45,
        ground_z_m: float = -0.2,
    ) -> None:
        self.markers = list(markers or [])
        self.dt_s = float(dt_s)
        self.max_step_m = float(max_step_m)
        self.max_yaw_step_rad = float(max_yaw_step_rad)
        self.collision_radius_m = float(collision_radius_m)
        self.ground_z_m = float(ground_z_m)
        self.position = np.zeros(3, dtype=float)
        self.yaw = 0.0
        self.steps = 0
        self.last_collision = False

    def reset(self, start_pose: dict[str, Any]) -> dict[str, Any]:
        self.position = np.asarray(
            start_pose["position_ned_m"], dtype=float
        ).copy()
        self.yaw = float(start_pose.get("yaw_rad", 0.0))
        self.steps = 0
        self.last_collision = self._in_collision(self.position)
        return self.observe()

    def _in_collision(self, position: np.ndarray) -> bool:
        if position[2] >= self.ground_z_m:
            return True
        for marker in self.markers:
            pos = marker["position_ned_m"]
            if (
                math.hypot(position[0] - pos[0], position[1] - pos[1])
                <= self.collision_radius_m
                and abs(position[2] - pos[2]) <= 1.0
            ):
                return True
        return False

    def observe(self) -> dict[str, Any]:
        return {
            "position_ned_m": [float(v) for v in self.position],
            "yaw_rad": float(self.yaw),
            "t_s": float(self.steps) * self.dt_s,
            "rgb": None,
            "collision": bool(self.last_collision),
        }

    def step(self, action: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        info = {
            "collision": False,
            "intervention": False,
            "invalid_action": False,
        }
        try:
            arr = _as_action(action)
        except (TypeError, ValueError):
            info["invalid_action"] = True
            self.steps += 1
            self.last_collision = self._in_collision(self.position)
            info["collision"] = self.last_collision
            return self.observe(), info

        delta = arr[:3].copy()
        dyaw = float(arr[3])
        norm = float(np.linalg.norm(delta))
        if norm > self.max_step_m + 1e-9:
            delta *= self.max_step_m / norm
            info["intervention"] = True
        if abs(dyaw) > self.max_yaw_step_rad + 1e-9:
            dyaw = math.copysign(self.max_yaw_step_rad, dyaw)
            info["intervention"] = True

        new_position = self.position + delta
        if not np.all(np.isfinite(new_position)):
            info["invalid_action"] = True
            self.steps += 1
            self.last_collision = self._in_collision(self.position)
            info["collision"] = self.last_collision
            return self.observe(), info

        self.position = new_position
        self.yaw = _wrap_angle(self.yaw + dyaw)
        self.steps += 1
        self.last_collision = self._in_collision(self.position)
        info["collision"] = self.last_collision
        return self.observe(), info

    def close(self) -> None:
        return None


class ScriptedPolicy:
    """Deterministic waypoint policy derived from manifest predicates.

    The policy is intentionally simple: it plans a geometric route that
    satisfies each subgoal, then follows it with fixed-size actions.  It is the
    mock's deterministic baseline, not a learned policy.
    """

    def __init__(
        self,
        episode: dict[str, Any],
        max_step_m: float = 0.5,
        arc_step_deg: float = 20.0,
        orbit_margin_deg: float = 40.0,
        pass_margin_m: float = 0.4,
        hold_margin_s: float = 0.6,
    ) -> None:
        self.episode = episode
        self.max_step_m = float(max_step_m)
        self.arc_step_deg = float(arc_step_deg)
        self.orbit_margin_deg = float(orbit_margin_deg)
        self.pass_margin_m = float(pass_margin_m)
        self.hold_margin_s = float(hold_margin_s)
        self.dt_s = float(episode.get("dt_s", 0.2))
        self.waypoints: list[np.ndarray] = []
        self.index = 0
        self.done = False
        self._plan()

    def _plan(self) -> None:
        episode = self.episode
        pos = np.asarray(
            episode["start_pose"]["position_ned_m"], dtype=float
        ).copy()
        start_xy = pos[:2].copy()

        predicates = list(episode.get("success_predicates", []))
        for pred_index, pred in enumerate(predicates):
            ptype = pred["type"]
            next_type = (
                predicates[pred_index + 1].get("type")
                if pred_index + 1 < len(predicates)
                else None
            )
            if ptype == "altitude":
                target = pos.copy()
                target[2] = float(pred["target_z_m"])
                self.waypoints.append(target)
                pos = target
                continue

            marker = _marker_position(episode, pred["marker_id"])
            mx, my, mz = marker

            if ptype == "distance_to_marker":
                vec = pos[:2] - marker[:2]
                dist = float(np.linalg.norm(vec))
                target_distance = float(pred["distance_m"])
                if dist <= 1e-9:
                    target = np.array([mx, my, pos[2]], dtype=float)
                else:
                    target = np.array(
                        [
                            mx + vec[0] / dist * target_distance,
                            my + vec[1] / dist * target_distance,
                            pos[2],
                        ],
                        dtype=float,
                    )
                self.waypoints.append(target)
                pos = target
            elif ptype == "pass_side":
                direction = marker[:2] - start_xy
                norm = float(np.linalg.norm(direction))
                if norm <= 1e-9:
                    direction = np.array([1.0, 0.0])
                    norm = 1.0
                direction /= norm
                right = np.array([-direction[1], direction[0]])
                sign = -1.0 if pred["side"] == "left" else 1.0
                offset = sign * (float(pred["clearance_m"]) + self.pass_margin_m)
                pre = np.array(
                    [
                        mx - 2.0 * direction[0] + right[0] * offset,
                        my - 2.0 * direction[1] + right[1] * offset,
                        pos[2],
                    ]
                )
                cross = np.array([mx + right[0] * offset, my + right[1] * offset, pos[2]])
                post = np.array(
                    [
                        mx + 2.0 * direction[0] + right[0] * offset,
                        my + 2.0 * direction[1] + right[1] * offset,
                        pos[2],
                    ]
                )
                if next_type == "orbit":
                    # The crossing sample is enough to satisfy the pass
                    # predicate.  Starting the orbit from the crossing avoids
                    # an unnecessary backward bearing sweep around the marker.
                    self.waypoints.extend([pre, cross])
                    pos = cross
                else:
                    self.waypoints.extend([pre, cross, post])
                    pos = post
            elif ptype == "orbit":
                radius = float(pred["radius_m"])
                sign = 1.0 if pred["direction"] == "clockwise" else -1.0
                bearing = math.atan2(pos[1] - my, pos[0] - mx)
                first = np.array(
                    [
                        mx + radius * math.cos(bearing),
                        my + radius * math.sin(bearing),
                        pos[2],
                    ]
                )
                self.waypoints.append(first)
                total_deg = float(pred["min_angle_deg"]) + self.orbit_margin_deg
                n_arc = max(1, int(math.ceil(total_deg / self.arc_step_deg)))
                for k in range(1, n_arc + 1):
                    angle = bearing + sign * math.radians(self.arc_step_deg * k)
                    self.waypoints.append(
                        np.array(
                            [
                                mx + radius * math.cos(angle),
                                my + radius * math.sin(angle),
                                pos[2],
                            ]
                        )
                    )
                pos = self.waypoints[-1]
            elif ptype == "hold_at_distance":
                vec = pos[:2] - marker[:2]
                dist = float(np.linalg.norm(vec))
                target_distance = float(pred["distance_m"])
                if dist <= 1e-9:
                    target = np.array([mx, my, pos[2]], dtype=float)
                else:
                    target = np.array(
                        [
                            mx + vec[0] / dist * target_distance,
                            my + vec[1] / dist * target_distance,
                            pos[2],
                        ],
                        dtype=float,
                    )
                self.waypoints.append(target)
                repeats = int(
                    math.ceil(
                        (float(pred["duration_s"]) + self.hold_margin_s)
                        / max(self.dt_s, 1e-6)
                    )
                )
                for _ in range(repeats):
                    self.waypoints.append(target.copy())
                pos = target
            else:  # pragma: no cover - validate_manifest rejects unknown types
                raise ValueError(f"unknown predicate type: {ptype!r}")

    def act(self, observation: dict[str, Any]) -> tuple[list[float], float]:
        started = time.perf_counter()
        if self.index >= len(self.waypoints):
            self.done = True
            action = list(ZERO_ACTION)
        else:
            position = np.asarray(observation["position_ned_m"], dtype=float)
            target = self.waypoints[self.index]
            delta = target - position
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-12:
                self.index += 1
                action = list(ZERO_ACTION)
            else:
                if norm > self.max_step_m:
                    delta *= self.max_step_m / norm
                action = [float(delta[0]), float(delta[1]), float(delta[2]), 0.0]
        latency_ms = (time.perf_counter() - started) * 1000.0
        return action, latency_ms


@dataclass
class EpisodeRun:
    summary: dict[str, Any]
    trajectory: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    latencies_ms: list[float]


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _step_record(
    episode_id: str,
    step: int,
    observation: dict[str, Any],
    action: Any,
    info: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "step": int(step),
        "t_s": float(observation["t_s"]),
        "position_ned_m": [float(v) for v in observation["position_ned_m"]],
        "yaw_rad": float(observation["yaw_rad"]),
        "action": _action_list(action),
        "collision": bool(info.get("collision", False)),
        "intervention": bool(info.get("intervention", False)),
        "invalid_action": bool(info.get("invalid_action", False)),
        "policy_latency_ms": float(latency_ms),
    }


def run_episode(
    env: Any,
    policy: Any,
    episode: dict[str, Any],
    *,
    max_steps: int | None = None,
) -> EpisodeRun:
    """Roll out one episode and evaluate the observed trajectory."""
    episode_id = episode["episode_id"]
    dt_s = float(episode.get("dt_s", 0.2))
    step_limit = int(max_steps if max_steps is not None else episode["max_steps"])

    observation = env.reset(episode["start_pose"])
    trajectory: list[dict[str, Any]] = [
        {
            "position_ned_m": [float(v) for v in observation["position_ned_m"]],
            "yaw_rad": float(observation["yaw_rad"]),
            "t_s": 0.0,
            "collision": bool(observation.get("collision", False)),
            "invalid_action": False,
        }
    ]
    steps: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    interventions = 0
    collisions = 0

    for step in range(1, step_limit + 1):
        action, latency_ms = policy.act(observation)
        latencies_ms.append(float(latency_ms))
        next_observation, info = env.step(action)
        interventions += int(bool(info.get("intervention", False)))
        collisions += int(bool(info.get("collision", False)))
        record = _step_record(
            episode_id, step, next_observation, action, info, latency_ms
        )
        steps.append(record)
        trajectory.append(
            {
                "position_ned_m": list(record["position_ned_m"]),
                "yaw_rad": record["yaw_rad"],
                "t_s": record["t_s"],
                "collision": record["collision"],
                "invalid_action": record["invalid_action"],
            }
        )
        observation = next_observation
        if info.get("collision", False):
            break
        if getattr(policy, "done", False):
            break

    evaluation = evaluate_episode(episode, trajectory, dt_s=dt_s)
    summary = {
        "episode_id": episode_id,
        "split": episode["split"],
        "ood": bool(episode["ood"]),
        "family": episode["family"],
        "template": episode["template"],
        "instruction": episode["instruction"],
        "success": bool(evaluation.success),
        "intent_score": float(evaluation.intent_score),
        "intent_adherence_score": float(evaluation.intent_score),
        "subgoal_completion": float(evaluation.subgoal_completion),
        "completed_subgoals": int(evaluation.completed_subgoals),
        "total_subgoals": int(evaluation.total_subgoals),
        "collision": bool(evaluation.collision),
        "collisions": int(collisions),
        "invalid_actions": int(evaluation.invalid_action_count),
        "invalid_action_count": int(evaluation.invalid_action_count),
        "interventions": int(interventions),
        "steps": len(steps),
        "latency_ms_mean": (
            float(statistics.fmean(latencies_ms)) if latencies_ms else 0.0
        ),
        "latency_ms_median": (
            float(statistics.median(latencies_ms)) if latencies_ms else 0.0
        ),
        "latency_ms_p95": _p95(latencies_ms),
        "path_length_m": float(evaluation.path_length_m),
        "reference_path_m": float(evaluation.reference_path_m),
        "path_efficiency": (
            None
            if evaluation.path_efficiency is None
            else float(evaluation.path_efficiency)
        ),
        "max_steps": step_limit,
    }
    return EpisodeRun(
        summary=summary,
        trajectory=trajectory,
        steps=steps,
        latencies_ms=latencies_ms,
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=False) + "\n")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_id",
        "split",
        "ood",
        "family",
        "template",
        "success",
        "intent_score",
        "intent_adherence_score",
        "subgoal_completion",
        "completed_subgoals",
        "total_subgoals",
        "collision",
        "collisions",
        "invalid_actions",
        "invalid_action_count",
        "interventions",
        "steps",
        "latency_ms_mean",
        "latency_ms_median",
        "latency_ms_p95",
        "path_length_m",
        "reference_path_m",
        "path_efficiency",
        "max_steps",
        "instruction",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fieldnames}
            if flat["path_efficiency"] is None:
                flat["path_efficiency"] = ""
            writer.writerow(flat)


def _aggregate(
    rows: list[dict[str, Any]],
    all_latencies_ms: list[float],
    mode: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    n = len(rows)
    successes = sum(1 for row in rows if row["success"])
    id_rows = [row for row in rows if row["split"] == "id"]
    ood_rows = [row for row in rows if row["split"] == "ood"]
    efficiencies = [
        float(row["path_efficiency"])
        for row in rows
        if row["path_efficiency"] is not None
    ]

    def slice_stats(subset: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(subset)
        wins = sum(1 for row in subset if row["success"])
        return {
            "episodes": count,
            "successes": wins,
            "success_rate": (wins / count) if count else 0.0,
            "mean_intent_score": (
                sum(row["intent_score"] for row in subset) / count if count else 0.0
            ),
        }

    return {
        "schema_version": "1.0",
        "suite_id": manifest.get("suite_id", "uav_intent_30"),
        "mode": mode,
        "episodes": n,
        "successes": successes,
        "success_rate": (successes / n) if n else 0.0,
        "id": slice_stats(id_rows),
        "ood": slice_stats(ood_rows),
        "mean_intent_score": (
            sum(row["intent_score"] for row in rows) / n if n else 0.0
        ),
        "mean_subgoal_completion": (
            sum(row["subgoal_completion"] for row in rows) / n if n else 0.0
        ),
        "collisions": sum(row["collisions"] for row in rows),
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "interventions": sum(row["interventions"] for row in rows),
        "total_steps": sum(row["steps"] for row in rows),
        "mean_path_efficiency": (
            sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
        ),
        "latency_ms_mean": (
            float(statistics.fmean(all_latencies_ms)) if all_latencies_ms else 0.0
        ),
        "latency_ms_median": (
            float(statistics.median(all_latencies_ms)) if all_latencies_ms else 0.0
        ),
        "latency_ms_p95": _p95(all_latencies_ms),
        "per_episode": rows,
    }


def _make_mock_env(episode: dict[str, Any]) -> MockEnv:
    return MockEnv(markers=episode.get("markers", []), dt_s=float(episode.get("dt_s", 0.2)))


def _make_airsim_env(args: argparse.Namespace, episode: dict[str, Any]) -> Any:
    # Imported lazily so ``--mock`` and test collection never import airsim.
    from scripts.eval.airsim_env import AirSimEnv

    return AirSimEnv(
        ip=args.airsim_ip,
        port=args.airsim_port,
        vehicle_name=args.airsim_vehicle,
        camera_name=args.airsim_camera,
        dt_s=float(episode.get("dt_s", 0.2)),
    )


def run_evaluation(
    manifest: dict[str, Any],
    out_dir: str | Path,
    *,
    mode: str = "mock",
    episode_ids: Sequence[str] | None = None,
    max_episodes: int | None = None,
    airsim_args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    """Run the suite and write episode JSONL, summary CSV and summary JSON."""
    if mode not in ("mock", "airsim"):
        raise ValueError("mode must be 'mock' or 'airsim'")
    out = Path(out_dir)
    episode_dir = out / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    episodes = list(manifest["episodes"])
    if episode_ids:
        wanted = set(episode_ids)
        known = {ep["episode_id"] for ep in episodes}
        unknown = wanted - known
        if unknown:
            raise ValueError(f"unknown episode id(s): {sorted(unknown)}")
        episodes = [ep for ep in episodes if ep["episode_id"] in wanted]
    if max_episodes is not None:
        episodes = episodes[: max(0, int(max_episodes))]
    if not episodes:
        raise ValueError("no episodes selected")

    rows: list[dict[str, Any]] = []
    all_latencies: list[float] = []
    shared_airsim: Any = None
    try:
        for episode in episodes:
            if mode == "airsim":
                if shared_airsim is None:
                    if airsim_args is None:
                        raise ValueError("airsim mode requires airsim arguments")
                    shared_airsim = _make_airsim_env(airsim_args, episode)
                env = shared_airsim
            else:
                env = _make_mock_env(episode)

            policy = ScriptedPolicy(episode)
            result = run_episode(env, policy, episode)
            rows.append(result.summary)
            all_latencies.extend(result.latencies_ms)
            _write_jsonl(
                episode_dir / f"{episode['episode_id']}.jsonl", result.steps
            )
            if mode == "mock":
                env.close()
    finally:
        if shared_airsim is not None:
            closer = getattr(shared_airsim, "close", None)
            if callable(closer):
                closer()

    _write_jsonl(out / "per_episode.jsonl", rows)
    _write_summary_csv(out / "summary.csv", rows)
    summary = _aggregate(rows, all_latencies, mode, manifest)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="frozen manifest path",
    )
    parser.add_argument(
        "--out",
        default="runs/eval/uav_intent_30",
        help="output directory (default: runs/eval/uav_intent_30)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock",
        action="store_true",
        help="use the deterministic mock environment (default)",
    )
    mode.add_argument(
        "--airsim",
        action="store_true",
        help="use the AirSimNH adapter hook (requires the airsim package/server)",
    )
    parser.add_argument(
        "--episode-id",
        action="append",
        default=[],
        help="run only this episode id (repeatable)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="cap the number of selected episodes",
    )
    parser.add_argument("--airsim-ip", default="127.0.0.1")
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--airsim-vehicle", default="Drone0")
    parser.add_argument("--airsim-camera", default="0")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    mode = "airsim" if args.airsim else "mock"
    manifest = load_manifest(args.manifest)
    try:
        summary = run_evaluation(
            manifest,
            args.out,
            mode=mode,
            episode_ids=args.episode_id or None,
            max_episodes=args.max_episodes,
            airsim_args=args,
        )
    except Exception as exc:
        # Keep the optional AirSim dependency optional: report a concise,
        # actionable error instead of a traceback when AirSim is unavailable.
        from scripts.eval.airsim_env import AirSimUnavailableError

        if isinstance(exc, AirSimUnavailableError):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise
    print(
        f"{mode}: {summary['successes']}/{summary['episodes']} success "
        f"({summary['success_rate']:.1%}), "
        f"intent_score={summary['mean_intent_score']:.3f}, "
        f"collisions={summary['collisions']}, "
        f"interventions={summary['interventions']}, "
        f"path_efficiency={summary['mean_path_efficiency']:.3f}"
    )
    print(f"wrote {Path(args.out) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
