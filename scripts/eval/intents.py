#!/usr/bin/env python3
"""Geometric intent predicates and episode scoring for the UAV eval suite.

All positions are world-frame NED metres: x north, y east, z down.  Yaw is
clockwise-positive around +z, matching AirSim.  Trajectory entries may be:

* a mapping with ``position_ned_m`` (or ``position`` / ``pos``) and optional
  ``t_s``, ``collision`` and ``invalid_action`` keys, as written by
  ``scripts/eval/evaluate.py``; or
* a 3-element sequence ``[x, y, z]``.

Predicates are evaluator-only.  They never inspect a policy's claimed intent;
they scan the observed trajectory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

TAU = 2.0 * math.pi
EPS = 1e-9


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    value = (angle + math.pi) % TAU - math.pi
    if value <= -math.pi:
        value += TAU
    return value


def horizontal_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Horizontal (x/y) distance between two NED points."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _position_of(state: Any) -> Sequence[float]:
    if isinstance(state, dict):
        for key in ("position_ned_m", "position", "pos"):
            if key in state:
                return state[key]
        raise ValueError("trajectory mapping has no position_ned_m/position/pos")
    return state


def _time_of(state: Any) -> float | None:
    if not isinstance(state, dict):
        return None
    for key in ("t_s", "time_s", "t"):
        if key in state:
            return float(state[key])
    return None


def _normalize_trajectory(
    trajectory: Iterable[Any],
    dt_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    states = list(trajectory)
    if not states:
        raise ValueError("trajectory must contain at least one state")
    points: list[list[float]] = []
    times: list[float] = []
    for i, state in enumerate(states):
        raw = _position_of(state)
        if len(raw) != 3:
            raise ValueError(f"state {i}: expected a 3-D NED position")
        point = [float(v) for v in raw]
        if not all(math.isfinite(v) for v in point):
            raise ValueError(f"state {i}: non-finite position {point}")
        points.append(point)
        t = _time_of(state)
        times.append(float(i) * float(dt_s) if t is None else t)
    return np.asarray(points, dtype=float), np.asarray(times, dtype=float)


def _marker_xyz(marker_ned: Sequence[float]) -> np.ndarray:
    arr = np.asarray(marker_ned, dtype=float).reshape(-1)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError("marker position must be a finite 3-D NED point")
    return arr


@dataclass
class PredicateResult:
    """Outcome of one geometric subgoal predicate on a trajectory."""

    predicate_type: str
    satisfied: bool
    progress: float
    end_index: int = -1
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.predicate_type,
            "satisfied": bool(self.satisfied),
            "progress": float(self.progress),
            "end_index": int(self.end_index),
            "detail": dict(self.detail),
        }


@dataclass
class EpisodeEvaluation:
    """Aggregate evaluator output for one episode."""

    success: bool
    intent_score: float
    subgoal_completion: float
    completed_subgoals: int
    total_subgoals: int
    subgoals: list[dict[str, Any]]
    collision: bool
    invalid_action_count: int
    path_length_m: float
    reference_path_m: float
    path_efficiency: float | None
    final_position_ned_m: list[float]

    @property
    def intent_adherence_score(self) -> float:
        return self.intent_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "intent_score": float(self.intent_score),
            "intent_adherence_score": float(self.intent_score),
            "subgoal_completion": float(self.subgoal_completion),
            "completed_subgoals": int(self.completed_subgoals),
            "total_subgoals": int(self.total_subgoals),
            "subgoals": list(self.subgoals),
            "collision": bool(self.collision),
            "invalid_action_count": int(self.invalid_action_count),
            "path_length_m": float(self.path_length_m),
            "reference_path_m": float(self.reference_path_m),
            "path_efficiency": (
                None if self.path_efficiency is None else float(self.path_efficiency)
            ),
            "final_position_ned_m": [float(v) for v in self.final_position_ned_m],
        }


# ---------------------------------------------------------------------------
# Individual geometric predicates.  The public functions accept raw
# trajectories; the private ``_*_arrays`` variants are reused by
# evaluate_episode() so the trajectory is normalized once.
# ---------------------------------------------------------------------------


def distance_to_marker(
    trajectory: Iterable[Any],
    marker_ned: Sequence[float],
    distance_m: float,
    tolerance_m: float = 0.0,
    start_index: int = 0,
    axis: str = "horizontal",
    dt_s: float = 0.1,
) -> PredicateResult:
    """Reward being within ``tolerance_m`` of the requested stand-off.

    ``axis='horizontal'`` ignores altitude, matching AirSimNH landmark
    stand-off evaluation.  ``axis='3d'`` uses full Euclidean distance.
    """
    points, _ = _normalize_trajectory(trajectory, dt_s=dt_s)
    return _distance_to_marker_arrays(
        points, marker_ned, distance_m, tolerance_m, start_index, axis
    )


def _distance_to_marker_arrays(
    points: np.ndarray,
    marker_ned: Sequence[float],
    distance_m: float,
    tolerance_m: float,
    start_index: int,
    axis: str,
) -> PredicateResult:
    marker = _marker_xyz(marker_ned)
    n = len(points)
    start = max(0, min(int(start_index), n - 1))
    if axis not in ("horizontal", "3d"):
        raise ValueError("axis must be 'horizontal' or '3d'")
    if axis == "horizontal":
        distances = np.hypot(points[:, 0] - marker[0], points[:, 1] - marker[1])
    else:
        distances = np.linalg.norm(points - marker, axis=1)
    errors = np.abs(distances - float(distance_m))
    start_error = float(errors[start])
    best_index = int(np.argmin(errors[start:])) + start
    best_error = float(errors[best_index])

    hits = np.nonzero(errors[start:] <= float(tolerance_m) + 1e-9)[0]
    if hits.size:
        end = start + int(hits[0])
        satisfied = True
        progress = 1.0
    else:
        end = -1
        satisfied = False
        if start_error <= EPS:
            progress = 0.0
        else:
            progress = max(0.0, min(1.0, 1.0 - best_error / start_error))

    return PredicateResult(
        predicate_type="distance_to_marker",
        satisfied=satisfied,
        progress=progress,
        end_index=end,
        detail={
            "marker_ned_m": marker.tolist(),
            "target_distance_m": float(distance_m),
            "tolerance_m": float(tolerance_m),
            "axis": axis,
            "final_distance_m": float(distances[-1]),
            "best_distance_m": float(distances[best_index]),
            "best_error_m": best_error,
        },
    )


def _side_sign(side: str) -> float:
    normalized = side.strip().lower()
    if normalized in ("left", "l"):
        return -1.0
    if normalized in ("right", "r"):
        return 1.0
    raise ValueError(f"unknown pass side: {side!r}")


def pass_side(
    trajectory: Iterable[Any],
    marker_ned: Sequence[float],
    side: str,
    clearance_m: float = 1.0,
    origin_ned: Sequence[float] | None = None,
    start_index: int = 0,
    dt_s: float = 0.1,
) -> PredicateResult:
    """Check crossing a marker's transverse plane on the requested side.

    The plane is perpendicular to the horizontal direction from ``origin_ned``
    (default: the first trajectory point) to the marker.  Crossing is a
    transition from the near side to the far side.  At the crossing sample the
    lateral offset in the plane must be at least ``clearance_m`` on the
    requested side; positive lateral is the right side when facing the marker.
    """
    points, _ = _normalize_trajectory(trajectory, dt_s=dt_s)
    return _pass_side_arrays(
        points, marker_ned, side, clearance_m, origin_ned, start_index
    )


def _pass_side_arrays(
    points: np.ndarray,
    marker_ned: Sequence[float],
    side: str,
    clearance_m: float,
    origin_ned: Sequence[float] | None,
    start_index: int,
) -> PredicateResult:
    marker = _marker_xyz(marker_ned)
    n = len(points)
    start = max(0, min(int(start_index), n - 1))
    origin = np.asarray(
        points[start] if origin_ned is None else origin_ned, dtype=float
    ).reshape(-1)
    if origin.shape != (3,):
        raise ValueError("origin_ned must be a 3-D NED point")

    direction_xy = marker[:2] - origin[:2]
    norm = float(np.linalg.norm(direction_xy))
    if norm <= EPS:
        raise ValueError("origin and marker coincide; pass plane is undefined")
    direction_xy /= norm
    right_xy = np.array([-direction_xy[1], direction_xy[0]])

    rel_xy = points[:, :2] - marker[:2]
    plane = rel_xy @ direction_xy
    lateral = rel_xy @ right_xy

    sign = _side_sign(side)
    cross_index = -1
    cross_lateral = 0.0
    for i in range(start + 1, n):
        if plane[i - 1] < 0.0 <= plane[i]:
            cross_index = i
            cross_lateral = float(lateral[i])
            break

    if cross_index >= 0:
        correct = (
            cross_lateral <= -float(clearance_m) - 1e-9
            if sign < 0
            else cross_lateral >= float(clearance_m) + 1e-9
        )
        satisfied = bool(correct)
        progress = 1.0 if satisfied else 0.6
        end = cross_index
    else:
        satisfied = False
        end = -1
        start_plane = float(plane[start])
        if start_plane < 0.0:
            min_plane = float(np.min(plane[start:]))
            fraction = max(0.0, min(1.0, (start_plane - min_plane) / abs(start_plane)))
            progress = 0.5 * fraction
        else:
            progress = 0.0

    return PredicateResult(
        predicate_type="pass_side",
        satisfied=satisfied,
        progress=progress,
        end_index=end,
        detail={
            "marker_ned_m": marker.tolist(),
            "side": side.strip().lower(),
            "clearance_m": float(clearance_m),
            "origin_ned_m": origin.tolist(),
            "direction_xy": direction_xy.tolist(),
            "crossed": cross_index >= 0,
            "crossing_index": cross_index,
            "crossing_lateral_m": cross_lateral,
            "final_plane_m": float(plane[-1]),
        },
    )


def _direction_sign(direction: str) -> float:
    normalized = direction.strip().lower().replace("_", "-")
    if normalized in ("clockwise", "cw", "clock-wise"):
        return 1.0
    if normalized in ("counterclockwise", "counter-clockwise", "ccw"):
        return -1.0
    raise ValueError(f"unknown orbit direction: {direction!r}")


def orbit(
    trajectory: Iterable[Any],
    marker_ned: Sequence[float],
    direction: str,
    min_angle_deg: float = 180.0,
    radius_m: float = 3.5,
    radius_tolerance_m: float = 2.5,
    min_radius_m: float = 1.0,
    start_index: int = 0,
    dt_s: float = 0.1,
) -> PredicateResult:
    """Check accumulated bearing sweep around a marker.

    Bearing is ``atan2(east, north)`` in NED, so an increasing bearing is a
    clockwise sweep when viewed from above.  Only consecutive samples whose
    radii lie in ``[min_radius_m, radius_m + radius_tolerance_m]`` contribute.
    """
    points, _ = _normalize_trajectory(trajectory, dt_s=dt_s)
    return _orbit_arrays(
        points,
        marker_ned,
        direction,
        min_angle_deg,
        radius_m,
        radius_tolerance_m,
        min_radius_m,
        start_index,
    )


def _orbit_arrays(
    points: np.ndarray,
    marker_ned: Sequence[float],
    direction: str,
    min_angle_deg: float,
    radius_m: float,
    radius_tolerance_m: float,
    min_radius_m: float,
    start_index: int,
) -> PredicateResult:
    marker = _marker_xyz(marker_ned)
    n = len(points)
    start = max(0, min(int(start_index), n - 1))
    sign = _direction_sign(direction)
    target = math.radians(float(min_angle_deg))
    max_radius = float(radius_m) + float(radius_tolerance_m)

    rel = points[:, :2] - marker[:2]
    radii = np.hypot(rel[:, 0], rel[:, 1])
    bearings = np.arctan2(rel[:, 1], rel[:, 0])

    accumulated = 0.0
    end = -1
    for i in range(start + 1, n):
        if (
            min_radius_m - 1e-9 <= radii[i - 1] <= max_radius + 1e-9
            and min_radius_m - 1e-9 <= radii[i] <= max_radius + 1e-9
        ):
            accumulated += wrap_angle(float(bearings[i] - bearings[i - 1])) * sign
        if accumulated >= target - 1e-9:
            end = i
            break

    satisfied = end >= 0
    progress = max(0.0, min(1.0, accumulated / max(target, EPS)))
    return PredicateResult(
        predicate_type="orbit",
        satisfied=satisfied,
        progress=progress,
        end_index=end,
        detail={
            "marker_ned_m": marker.tolist(),
            "direction": direction,
            "min_angle_deg": float(min_angle_deg),
            "accumulated_deg": math.degrees(accumulated),
            "radius_m": float(radius_m),
            "min_radius_m": float(min_radius_m),
            "max_radius_m": max_radius,
            "final_radius_m": float(radii[-1]),
        },
    )


def hold_at_distance(
    trajectory: Iterable[Any],
    marker_ned: Sequence[float],
    distance_m: float,
    tolerance_m: float = 0.5,
    duration_s: float = 2.0,
    start_index: int = 0,
    dt_s: float = 0.1,
) -> PredicateResult:
    """Check a continuous dwell inside a distance-to-marker annulus."""
    points, times = _normalize_trajectory(trajectory, dt_s=dt_s)
    return _hold_at_distance_arrays(
        points, times, marker_ned, distance_m, tolerance_m, duration_s, start_index
    )


def _hold_at_distance_arrays(
    points: np.ndarray,
    times: np.ndarray,
    marker_ned: Sequence[float],
    distance_m: float,
    tolerance_m: float,
    duration_s: float,
    start_index: int,
) -> PredicateResult:
    marker = _marker_xyz(marker_ned)
    n = len(points)
    start = max(0, min(int(start_index), n - 1))
    distances = np.hypot(points[:, 0] - marker[0], points[:, 1] - marker[1])
    inside = np.abs(distances - float(distance_m)) <= float(tolerance_m) + 1e-9

    run_start: float | None = None
    best_hold = 0.0
    end = -1
    for i in range(start, n):
        if inside[i]:
            if run_start is None:
                run_start = float(times[i])
            held = float(times[i]) - run_start
            best_hold = max(best_hold, held)
            if held >= float(duration_s) - 1e-9:
                end = i
                break
        else:
            run_start = None

    satisfied = end >= 0
    progress = max(0.0, min(1.0, best_hold / max(float(duration_s), EPS)))
    return PredicateResult(
        predicate_type="hold_at_distance",
        satisfied=satisfied,
        progress=progress,
        end_index=end,
        detail={
            "marker_ned_m": marker.tolist(),
            "target_distance_m": float(distance_m),
            "tolerance_m": float(tolerance_m),
            "required_duration_s": float(duration_s),
            "best_hold_s": best_hold,
            "final_distance_m": float(distances[-1]),
        },
    )


def altitude(
    trajectory: Iterable[Any],
    target_z_m: float,
    tolerance_m: float = 0.5,
    start_index: int = 0,
    dt_s: float = 0.1,
) -> PredicateResult:
    """Check that the trajectory reaches a target NED altitude (z down)."""
    points, _ = _normalize_trajectory(trajectory, dt_s=dt_s)
    return _altitude_arrays(points, target_z_m, tolerance_m, start_index)


def _altitude_arrays(
    points: np.ndarray,
    target_z_m: float,
    tolerance_m: float,
    start_index: int,
) -> PredicateResult:
    n = len(points)
    start = max(0, min(int(start_index), n - 1))
    errors = np.abs(points[:, 2] - float(target_z_m))
    start_error = float(errors[start])
    best_index = int(np.argmin(errors[start:])) + start
    best_error = float(errors[best_index])
    hits = np.nonzero(errors[start:] <= float(tolerance_m) + 1e-9)[0]

    if hits.size:
        end = start + int(hits[0])
        satisfied = True
        progress = 1.0
    else:
        end = -1
        satisfied = False
        progress = (
            0.0
            if start_error <= EPS
            else max(0.0, min(1.0, 1.0 - best_error / start_error))
        )

    return PredicateResult(
        predicate_type="altitude",
        satisfied=satisfied,
        progress=progress,
        end_index=end,
        detail={
            "target_z_m": float(target_z_m),
            "tolerance_m": float(tolerance_m),
            "start_z_m": float(points[start, 2]),
            "final_z_m": float(points[-1, 2]),
            "achieved_delta_m": float(points[start, 2] - points[-1, 2]),
        },
    )


# ---------------------------------------------------------------------------
# Aggregate episode evaluation
# ---------------------------------------------------------------------------


def _marker_for(episode: dict[str, Any], marker_id: str) -> list[float]:
    for marker in episode.get("markers", []):
        if marker.get("marker_id") == marker_id:
            return [float(v) for v in marker["position_ned_m"]]
    raise KeyError(f"{episode.get('episode_id')}: unknown marker {marker_id!r}")


def evaluate_episode(
    episode: dict[str, Any],
    trajectory: Iterable[Any],
    dt_s: float | None = None,
) -> EpisodeEvaluation:
    """Evaluate one episode's observed trajectory.

    Subgoals from ``episode['success_predicates']`` are checked in order on
    successive trajectory slices.  Mission success requires every subgoal and
    no collision.  The intent score is the mean per-subgoal progress, so a
    partially completed first subgoal receives partial credit while skipped
    later subgoals receive zero.
    """
    default_dt = float(dt_s if dt_s is not None else episode.get("dt_s", 0.1))
    states = list(trajectory)
    points, times = _normalize_trajectory(states, dt_s=default_dt)
    predicates = list(episode.get("success_predicates", []))
    if not predicates:
        raise ValueError(f"{episode.get('episode_id')}: no success_predicates")

    start_pose = episode.get("start_pose", {})
    origin = start_pose.get("position_ned_m")

    results: list[PredicateResult] = []
    cursor = 0
    completed = 0
    failed = False

    for pred in predicates:
        ptype = pred.get("type")
        if failed:
            results.append(
                PredicateResult(
                    predicate_type=str(ptype),
                    satisfied=False,
                    progress=0.0,
                    end_index=-1,
                    detail={"skipped_after_failure": True},
                )
            )
            continue

        if ptype == "distance_to_marker":
            result = _distance_to_marker_arrays(
                points,
                _marker_for(episode, pred["marker_id"]),
                float(pred["distance_m"]),
                float(pred.get("tolerance_m", 0.0)),
                cursor,
                str(pred.get("axis", "horizontal")),
            )
        elif ptype == "pass_side":
            result = _pass_side_arrays(
                points,
                _marker_for(episode, pred["marker_id"]),
                str(pred["side"]),
                float(pred.get("clearance_m", 1.0)),
                origin,
                cursor,
            )
        elif ptype == "orbit":
            result = _orbit_arrays(
                points,
                _marker_for(episode, pred["marker_id"]),
                str(pred["direction"]),
                float(pred.get("min_angle_deg", 180.0)),
                float(pred.get("radius_m", 3.5)),
                float(pred.get("radius_tolerance_m", 2.5)),
                float(pred.get("min_radius_m", 1.0)),
                cursor,
            )
        elif ptype == "hold_at_distance":
            result = _hold_at_distance_arrays(
                points,
                times,
                _marker_for(episode, pred["marker_id"]),
                float(pred["distance_m"]),
                float(pred.get("tolerance_m", 0.5)),
                float(pred["duration_s"]),
                cursor,
            )
        elif ptype == "altitude":
            result = _altitude_arrays(
                points,
                float(pred["target_z_m"]),
                float(pred.get("tolerance_m", 0.5)),
                cursor,
            )
        else:
            raise ValueError(f"unknown predicate type: {ptype!r}")

        results.append(result)
        if result.satisfied:
            completed += 1
            if result.end_index >= 0:
                cursor = result.end_index
        else:
            failed = True

    total = len(predicates)
    intent_score = (
        sum(max(0.0, min(1.0, r.progress)) for r in results) / total
        if total
        else 0.0
    )

    collision = any(
        bool(state.get("collision")) for state in states if isinstance(state, dict)
    )
    invalid_action_count = sum(
        1
        for state in states
        if isinstance(state, dict) and bool(state.get("invalid_action"))
    )

    if len(points) > 1:
        path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    else:
        path_length = 0.0
    reference = float(episode.get("reference_path_m", 0.0) or 0.0)
    if path_length <= EPS:
        efficiency: float | None = 0.0
    elif reference > 0.0:
        efficiency = max(0.0, min(1.0, reference / path_length))
    else:
        efficiency = None

    success = completed == total and not collision
    return EpisodeEvaluation(
        success=success,
        intent_score=float(max(0.0, min(1.0, intent_score))),
        subgoal_completion=float(completed / total) if total else 0.0,
        completed_subgoals=completed,
        total_subgoals=total,
        subgoals=[r.to_dict() for r in results],
        collision=collision,
        invalid_action_count=invalid_action_count,
        path_length_m=path_length,
        reference_path_m=reference,
        path_efficiency=efficiency,
        final_position_ned_m=[float(v) for v in points[-1]],
    )
