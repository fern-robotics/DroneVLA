#!/usr/bin/env python3
"""Frozen 30-episode AirSimNH UAV intent evaluation manifest.

The suite is deliberately small and deterministic: 22 in-distribution (ID)
episodes and 8 out-of-distribution (OOD) episodes.  ID/OOD episodes are
separated by *intent template*, not by a random split.  An OOD episode may
therefore use a known geometric intent (for example, "orbit clockwise") but it
must phrase and compose that intent with a template that never appears in ID.

World coordinates are NED: x is north/forward, y is east/right, z is down.
Yaw is clockwise-positive around +z, matching AirSim.

Usage::

    python -m scripts.eval.manifest --write
    python -m scripts.eval.manifest --check
    python -m scripts.eval.manifest --out /tmp/uav_intent_30.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
SUITE_ID = "uav_intent_30"
REFERENCE_FRAME = "world_NED"
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "manifests" / "uav_intent_30.json"
)

FAMILIES = frozenset(
    {
        "approach",
        "pass_left",
        "pass_right",
        "orbit_cw",
        "orbit_ccw",
        "hold_at_distance",
        "climb",
        "descend",
        "composed",
    }
)

# Explicit template whitelists.  The two sets are disjoint by construction and
# validate_manifest() enforces that every episode uses a template from the
# matching split.
ID_TEMPLATES = frozenset(
    {
        "approach:stop_at_distance",
        "pass:left_side",
        "pass:right_side",
        "orbit:clockwise_min_arc",
        "orbit:counterclockwise_min_arc",
        "hold_at:distance_duration",
        "altitude:climb",
        "altitude:descend",
        "composed:approach_then_pass_left",
        "composed:pass_right_then_orbit_cw",
        "composed:climb_then_hold_at",
        "composed:descend_then_orbit_ccw",
    }
)

OOD_TEMPLATES = frozenset(
    {
        "ood:approach_halt_short",
        "ood:pass_left_keep_right",
        "ood:pass_right_keep_left",
        "ood:orbit_cw_long_arc",
        "ood:orbit_ccw_long_arc",
        "ood:hold_at_offset",
        "ood:composed_approach_climb_hold",
        "ood:composed_descend_pass_left_orbit_ccw",
    }
)

PREDICATE_TYPES = frozenset(
    {"distance_to_marker", "pass_side", "orbit", "hold_at_distance", "altitude"}
)

REQUIRED_EPISODE_KEYS = frozenset(
    {
        "episode_id",
        "split",
        "ood",
        "family",
        "template",
        "instruction",
        "markers",
        "start_pose",
        "success_predicates",
        "max_steps",
    }
)


def _marker(
    marker_id: str,
    color: str,
    shape: str,
    position_ned_m: tuple[float, float, float],
    size_m: float = 1.5,
) -> dict[str, Any]:
    return {
        "marker_id": marker_id,
        "color": color,
        "shape": shape,
        "position_ned_m": [float(v) for v in position_ned_m],
        "size_m": float(size_m),
    }


def _distance_pred(
    marker_id: str, distance_m: float, tolerance_m: float
) -> dict[str, Any]:
    return {
        "type": "distance_to_marker",
        "marker_id": marker_id,
        "distance_m": float(distance_m),
        "tolerance_m": float(tolerance_m),
        "axis": "horizontal",
    }


def _pass_pred(
    marker_id: str, side: str, clearance_m: float
) -> dict[str, Any]:
    if side not in ("left", "right"):
        raise ValueError(f"unknown pass side: {side}")
    return {
        "type": "pass_side",
        "marker_id": marker_id,
        "side": side,
        "clearance_m": float(clearance_m),
    }


def _orbit_pred(
    marker_id: str,
    direction: str,
    min_angle_deg: float,
    radius_m: float,
    radius_tolerance_m: float = 2.5,
) -> dict[str, Any]:
    if direction not in ("clockwise", "counterclockwise"):
        raise ValueError(f"unknown orbit direction: {direction}")
    return {
        "type": "orbit",
        "marker_id": marker_id,
        "direction": direction,
        "min_angle_deg": float(min_angle_deg),
        "radius_m": float(radius_m),
        "radius_tolerance_m": float(radius_tolerance_m),
    }


def _hold_pred(
    marker_id: str,
    distance_m: float,
    tolerance_m: float,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "type": "hold_at_distance",
        "marker_id": marker_id,
        "distance_m": float(distance_m),
        "tolerance_m": float(tolerance_m),
        "duration_s": float(duration_s),
    }


def _altitude_pred(
    target_z_m: float, tolerance_m: float, direction: str
) -> dict[str, Any]:
    if direction not in ("climb", "descend"):
        raise ValueError(f"unknown altitude direction: {direction}")
    return {
        "type": "altitude",
        "target_z_m": float(target_z_m),
        "tolerance_m": float(tolerance_m),
        "direction": direction,
    }


def _find_marker(episode: dict[str, Any], marker_id: str) -> list[float]:
    for marker in episode["markers"]:
        if marker["marker_id"] == marker_id:
            return [float(v) for v in marker["position_ned_m"]]
    raise KeyError(f"{episode['episode_id']}: unknown marker {marker_id}")


def _reference_path_m(episode: dict[str, Any]) -> float:
    """Nominal route length implied by the predicate sequence.

    This is a deterministic planning-length reference, not a rerun of the
    scripted policy.  ``evaluate_episode`` divides it by the observed path
    length and caps efficiency at 1.0.
    """
    pos = [float(v) for v in episode["start_pose"]["position_ned_m"]]
    total = 0.0
    start_xy = (pos[0], pos[1])
    for pred in episode["success_predicates"]:
        ptype = pred["type"]
        if ptype == "altitude":
            total += abs(pos[2] - float(pred["target_z_m"]))
            pos[2] = float(pred["target_z_m"])
            continue

        marker = _find_marker(episode, pred["marker_id"])
        mx, my, mz = marker

        if ptype == "distance_to_marker":
            vx, vy = pos[0] - mx, pos[1] - my
            dist = math.hypot(vx, vy)
            target = float(pred["distance_m"])
            if dist < 1e-9:
                goal = [mx, my, pos[2]]
            else:
                goal = [mx + vx / dist * target, my + vy / dist * target, pos[2]]
            total += math.dist(pos, goal)
            pos = goal
        elif ptype == "pass_side":
            dx, dy = mx - start_xy[0], my - start_xy[1]
            n = math.hypot(dx, dy)
            if n < 1e-9:
                dx, dy, n = 1.0, 0.0, 1.0
            dx, dy = dx / n, dy / n
            rx, ry = -dy, dx
            sign = -1.0 if pred["side"] == "left" else 1.0
            offset = sign * float(pred["clearance_m"])
            pre = [mx - 2.0 * dx + rx * offset, my - 2.0 * dy + ry * offset, pos[2]]
            cross = [mx + rx * offset, my + ry * offset, pos[2]]
            post = [mx + 2.0 * dx + rx * offset, my + 2.0 * dy + ry * offset, pos[2]]
            total += math.dist(pos, pre) + math.dist(pre, cross) + math.dist(cross, post)
            pos = post
        elif ptype == "orbit":
            radius = float(pred["radius_m"])
            bearing = math.atan2(pos[1] - my, pos[0] - mx)
            first = [mx + radius * math.cos(bearing), my + radius * math.sin(bearing), pos[2]]
            total += math.dist(pos, first)
            angle = math.radians(float(pred["min_angle_deg"]))
            sign = 1.0 if pred["direction"] == "clockwise" else -1.0
            total += radius * angle
            end_bearing = bearing + sign * angle
            pos = [
                mx + radius * math.cos(end_bearing),
                my + radius * math.sin(end_bearing),
                pos[2],
            ]
        elif ptype == "hold_at_distance":
            vx, vy = pos[0] - mx, pos[1] - my
            dist = math.hypot(vx, vy)
            target = float(pred["distance_m"])
            if dist < 1e-9:
                goal = [mx, my, pos[2]]
            else:
                goal = [mx + vx / dist * target, my + vy / dist * target, pos[2]]
            total += math.dist(pos, goal)
            pos = goal
        else:  # pragma: no cover - validate_manifest rejects unknown types
            raise ValueError(f"unknown predicate type: {ptype}")
    return total


def _episode(
    *,
    episode_id: str,
    split: str,
    family: str,
    template: str,
    instruction: str,
    markers: list[dict[str, Any]],
    start_position: tuple[float, float, float],
    start_yaw: float,
    predicates: list[dict[str, Any]],
    max_steps: int,
    dt_s: float = 0.2,
) -> dict[str, Any]:
    if split not in ("id", "ood"):
        raise ValueError(f"bad split {split!r}")
    if family not in FAMILIES:
        raise ValueError(f"bad family {family!r}")
    episode = {
        "episode_id": episode_id,
        "split": split,
        "ood": split == "ood",
        "family": family,
        "template": template,
        "instruction": instruction,
        "reference_frame": REFERENCE_FRAME,
        "markers": markers,
        "start_pose": {
            "position_ned_m": [float(v) for v in start_position],
            "yaw_rad": float(start_yaw),
        },
        "success_predicates": predicates,
        "max_steps": int(max_steps),
        "dt_s": float(dt_s),
    }
    episode["reference_path_m"] = round(_reference_path_m(episode), 3)
    return episode


def build_manifest() -> dict[str, Any]:
    """Build the deterministic 30-episode suite (22 ID, 8 OOD)."""
    episodes: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ ID
    # Approach: three stand-off distances and bearings.
    episodes.append(
        _episode(
            episode_id="id-approach-01",
            split="id",
            family="approach",
            template="approach:stop_at_distance",
            instruction="Approach the red circle and stop 4.0 m away.",
            markers=[_marker("M1", "red", "circle", (10.0, 0.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_distance_pred("M1", 4.0, 0.6)],
            max_steps=100,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-approach-02",
            split="id",
            family="approach",
            template="approach:stop_at_distance",
            instruction="Approach the blue square and stop 5.0 m away.",
            markers=[_marker("M1", "blue", "square", (12.0, 4.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_distance_pred("M1", 5.0, 0.6)],
            max_steps=120,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-approach-03",
            split="id",
            family="approach",
            template="approach:stop_at_distance",
            instruction="Approach the green cross and stop 3.0 m away.",
            markers=[_marker("M1", "green", "cross", (9.0, -3.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_distance_pred("M1", 3.0, 0.5)],
            max_steps=100,
        )
    )

    # Pass left/right.  The start is intentionally offset from the marker's
    # lateral axis so the direction-to-marker plane is not ambiguous.
    episodes.append(
        _episode(
            episode_id="id-pass-left-01",
            split="id",
            family="pass_left",
            template="pass:left_side",
            instruction="Pass the red circle on its left side with 1.2 m clearance.",
            markers=[_marker("M1", "red", "circle", (10.0, 0.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_pass_pred("M1", "left", 1.2)],
            max_steps=110,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-pass-left-02",
            split="id",
            family="pass_left",
            template="pass:left_side",
            instruction="Pass the yellow triangle on its left side with 1.5 m clearance.",
            markers=[_marker("M1", "yellow", "triangle", (11.0, -2.0, -10.0))],
            start_position=(0.0, 1.0, -10.0),
            start_yaw=0.0,
            predicates=[_pass_pred("M1", "left", 1.5)],
            max_steps=120,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-pass-right-01",
            split="id",
            family="pass_right",
            template="pass:right_side",
            instruction="Pass the blue square on its right side with 1.2 m clearance.",
            markers=[_marker("M1", "blue", "square", (10.0, 2.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_pass_pred("M1", "right", 1.2)],
            max_steps=110,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-pass-right-02",
            split="id",
            family="pass_right",
            template="pass:right_side",
            instruction="Pass the green cross on its right side with 1.5 m clearance.",
            markers=[_marker("M1", "green", "cross", (11.0, 1.0, -10.0))],
            start_position=(0.0, -1.0, -10.0),
            start_yaw=0.0,
            predicates=[_pass_pred("M1", "right", 1.5)],
            max_steps=120,
        )
    )

    # Orbit clockwise / counterclockwise.
    episodes.append(
        _episode(
            episode_id="id-orbit-cw-01",
            split="id",
            family="orbit_cw",
            template="orbit:clockwise_min_arc",
            instruction="Orbit clockwise around the red circle for at least 180 degrees at 3.5 m radius.",
            markers=[_marker("M1", "red", "circle", (10.0, 0.0, -10.0))],
            start_position=(6.5, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_orbit_pred("M1", "clockwise", 180.0, 3.5)],
            max_steps=220,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-orbit-cw-02",
            split="id",
            family="orbit_cw",
            template="orbit:clockwise_min_arc",
            instruction="Orbit clockwise around the green cross for at least 200 degrees at 3.0 m radius.",
            markers=[_marker("M1", "green", "cross", (11.0, 3.0, -10.0))],
            start_position=(8.0, 3.0, -10.0),
            start_yaw=0.0,
            predicates=[_orbit_pred("M1", "clockwise", 200.0, 3.0)],
            max_steps=240,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-orbit-ccw-01",
            split="id",
            family="orbit_ccw",
            template="orbit:counterclockwise_min_arc",
            instruction="Orbit counterclockwise around the blue square for at least 180 degrees at 3.5 m radius.",
            markers=[_marker("M1", "blue", "square", (10.0, 0.0, -10.0))],
            start_position=(6.5, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_orbit_pred("M1", "counterclockwise", 180.0, 3.5)],
            max_steps=220,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-orbit-ccw-02",
            split="id",
            family="orbit_ccw",
            template="orbit:counterclockwise_min_arc",
            instruction="Orbit counterclockwise around the yellow triangle for at least 200 degrees at 3.0 m radius.",
            markers=[_marker("M1", "yellow", "triangle", (12.0, -2.0, -10.0))],
            start_position=(9.0, -2.0, -10.0),
            start_yaw=0.0,
            predicates=[_orbit_pred("M1", "counterclockwise", 200.0, 3.0)],
            max_steps=240,
        )
    )

    # Hold at a fixed stand-off distance.
    episodes.append(
        _episode(
            episode_id="id-hold-01",
            split="id",
            family="hold_at_distance",
            template="hold_at:distance_duration",
            instruction="Hold 4.0 m from the red circle for 2.0 s.",
            markers=[_marker("M1", "red", "circle", (10.0, 0.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_hold_pred("M1", 4.0, 0.5, 2.0)],
            max_steps=150,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-hold-02",
            split="id",
            family="hold_at_distance",
            template="hold_at:distance_duration",
            instruction="Hold 5.0 m from the blue square for 1.5 s.",
            markers=[_marker("M1", "blue", "square", (11.0, 3.0, -10.0))],
            start_position=(1.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_hold_pred("M1", 5.0, 0.5, 1.5)],
            max_steps=150,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-hold-03",
            split="id",
            family="hold_at_distance",
            template="hold_at:distance_duration",
            instruction="Hold 3.0 m from the yellow triangle for 2.5 s.",
            markers=[_marker("M1", "yellow", "triangle", (9.0, -4.0, -10.0))],
            start_position=(0.0, -1.0, -10.0),
            start_yaw=0.0,
            predicates=[_hold_pred("M1", 3.0, 0.4, 2.5)],
            max_steps=180,
        )
    )

    # Altitude-only primitives.  The marker is a scene landmark, not a target.
    episodes.append(
        _episode(
            episode_id="id-climb-01",
            split="id",
            family="climb",
            template="altitude:climb",
            instruction="Climb 3.0 m.",
            markers=[_marker("M1", "red", "circle", (10.0, 0.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_altitude_pred(-13.0, 0.4, "climb")],
            max_steps=80,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-climb-02",
            split="id",
            family="climb",
            template="altitude:climb",
            instruction="Climb 2.0 m.",
            markers=[_marker("M1", "blue", "square", (12.0, 2.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_altitude_pred(-12.0, 0.4, "climb")],
            max_steps=80,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-descend-01",
            split="id",
            family="descend",
            template="altitude:descend",
            instruction="Descend 2.0 m.",
            markers=[_marker("M1", "green", "cross", (10.0, -2.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_altitude_pred(-8.0, 0.4, "descend")],
            max_steps=80,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-descend-02",
            split="id",
            family="descend",
            template="altitude:descend",
            instruction="Descend 2.5 m.",
            markers=[_marker("M1", "yellow", "triangle", (11.0, 2.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_altitude_pred(-7.5, 0.4, "descend")],
            max_steps=80,
        )
    )

    # Composed ID sequences.
    episodes.append(
        _episode(
            episode_id="id-comp-approach-pass-left",
            split="id",
            family="composed",
            template="composed:approach_then_pass_left",
            instruction=(
                "Approach the red circle to 6.0 m, then pass it on the left "
                "with 1.2 m clearance."
            ),
            markers=[_marker("M1", "red", "circle", (10.0, 0.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[
                _distance_pred("M1", 6.0, 0.6),
                _pass_pred("M1", "left", 1.2),
            ],
            max_steps=160,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-comp-pass-right-orbit-cw",
            split="id",
            family="composed",
            template="composed:pass_right_then_orbit_cw",
            instruction=(
                "Pass the blue square on the right with 1.2 m clearance, then "
                "orbit it clockwise for at least 180 degrees at 3.5 m radius."
            ),
            markers=[_marker("M1", "blue", "square", (10.0, 0.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[
                _pass_pred("M1", "right", 1.2),
                _orbit_pred("M1", "clockwise", 180.0, 3.5),
            ],
            max_steps=260,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-comp-climb-hold",
            split="id",
            family="composed",
            template="composed:climb_then_hold_at",
            instruction=(
                "Climb 3.0 m, then hold 4.0 m from the green cross for 1.5 s."
            ),
            markers=[_marker("M1", "green", "cross", (10.0, 2.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[
                _altitude_pred(-13.0, 0.4, "climb"),
                _hold_pred("M1", 4.0, 0.5, 1.5),
            ],
            max_steps=200,
        )
    )
    episodes.append(
        _episode(
            episode_id="id-comp-descend-orbit-ccw",
            split="id",
            family="composed",
            template="composed:descend_then_orbit_ccw",
            instruction=(
                "Descend 2.0 m, then orbit the yellow triangle counterclockwise "
                "for at least 180 degrees at 3.5 m radius."
            ),
            markers=[_marker("M1", "yellow", "triangle", (10.0, 0.0, -10.0))],
            start_position=(6.5, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[
                _altitude_pred(-8.0, 0.4, "descend"),
                _orbit_pred("M1", "counterclockwise", 180.0, 3.5),
            ],
            max_steps=260,
        )
    )

    # ----------------------------------------------------------------- OOD
    # Eight templates that never appear in the ID set.  They re-use the same
    # geometric intents with new phrasing, longer arcs, and new compositions.
    episodes.append(
        _episode(
            episode_id="ood-approach-01",
            split="ood",
            family="approach",
            template="ood:approach_halt_short",
            instruction="Fly toward the red circle, coming to rest 3.0 m short of it.",
            markers=[_marker("M1", "red", "circle", (12.0, 2.0, -10.0))],
            start_position=(0.0, -1.0, -10.0),
            start_yaw=0.0,
            predicates=[_distance_pred("M1", 3.0, 0.6)],
            max_steps=120,
        )
    )
    episodes.append(
        _episode(
            episode_id="ood-pass-left-01",
            split="ood",
            family="pass_left",
            template="ood:pass_left_keep_right",
            instruction=(
                "Slide past the blue square, keeping it to your right with "
                "1.4 m clearance."
            ),
            markers=[_marker("M1", "blue", "square", (10.0, 1.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_pass_pred("M1", "left", 1.4)],
            max_steps=120,
        )
    )
    episodes.append(
        _episode(
            episode_id="ood-pass-right-01",
            split="ood",
            family="pass_right",
            template="ood:pass_right_keep_left",
            instruction=(
                "Slide past the green cross, keeping it to your left with "
                "1.4 m clearance."
            ),
            markers=[_marker("M1", "green", "cross", (10.0, -1.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_pass_pred("M1", "right", 1.4)],
            max_steps=120,
        )
    )
    episodes.append(
        _episode(
            episode_id="ood-orbit-cw-01",
            split="ood",
            family="orbit_cw",
            template="ood:orbit_cw_long_arc",
            instruction=(
                "Make a 240-degree clockwise turn around the yellow triangle at "
                "roughly 3.5 m."
            ),
            markers=[_marker("M1", "yellow", "triangle", (11.0, 2.0, -10.0))],
            start_position=(7.5, 2.0, -10.0),
            start_yaw=0.0,
            predicates=[_orbit_pred("M1", "clockwise", 240.0, 3.5)],
            max_steps=260,
        )
    )
    episodes.append(
        _episode(
            episode_id="ood-orbit-ccw-01",
            split="ood",
            family="orbit_ccw",
            template="ood:orbit_ccw_long_arc",
            instruction=(
                "Make a 240-degree counterclockwise turn around the red circle "
                "at roughly 3.5 m."
            ),
            markers=[_marker("M1", "red", "circle", (10.0, -2.0, -10.0))],
            start_position=(6.5, -2.0, -10.0),
            start_yaw=0.0,
            predicates=[_orbit_pred("M1", "counterclockwise", 240.0, 3.5)],
            max_steps=260,
        )
    )
    episodes.append(
        _episode(
            episode_id="ood-hold-01",
            split="ood",
            family="hold_at_distance",
            template="ood:hold_at_offset",
            instruction=(
                "Remain 5.0 m from the green cross for 3.0 s without drifting."
            ),
            markers=[_marker("M1", "green", "cross", (12.0, 0.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[_hold_pred("M1", 5.0, 0.5, 3.0)],
            max_steps=200,
        )
    )
    episodes.append(
        _episode(
            episode_id="ood-comp-approach-climb-hold",
            split="ood",
            family="composed",
            template="ood:composed_approach_climb_hold",
            instruction=(
                "Approach the blue square to 5 m, climb 2 m, then hold 4 m "
                "from it for 1.5 s."
            ),
            markers=[_marker("M1", "blue", "square", (10.0, 0.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[
                _distance_pred("M1", 5.0, 0.6),
                _altitude_pred(-12.0, 0.4, "climb"),
                _hold_pred("M1", 4.0, 0.5, 1.5),
            ],
            max_steps=240,
        )
    )
    episodes.append(
        _episode(
            episode_id="ood-comp-descend-pass-left-orbit-ccw",
            split="ood",
            family="composed",
            template="ood:composed_descend_pass_left_orbit_ccw",
            instruction=(
                "Descend 2 m, pass the yellow triangle on the left with 1.3 m "
                "clearance, then orbit it counterclockwise for at least 180 "
                "degrees at 3.5 m radius."
            ),
            markers=[_marker("M1", "yellow", "triangle", (10.0, 1.0, -10.0))],
            start_position=(0.0, 0.0, -10.0),
            start_yaw=0.0,
            predicates=[
                _altitude_pred(-8.0, 0.4, "descend"),
                _pass_pred("M1", "left", 1.3),
                _orbit_pred("M1", "counterclockwise", 180.0, 3.5),
            ],
            max_steps=300,
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "reference_frame": REFERENCE_FRAME,
        "description": (
            "Frozen 30-episode AirSimNH UAV intent suite: 22 ID + 8 OOD, "
            "deterministic NED geometry, no randomness."
        ),
        "episodes": episodes,
    }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Raise ValueError when the frozen manifest violates its contract."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    if manifest.get("suite_id") != SUITE_ID:
        raise ValueError(f"suite_id must be {SUITE_ID!r}")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("episodes must be a list")

    ids = [e.get("episode_id") for e in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("episode_id values must be unique")
    if len(episodes) != 30:
        raise ValueError(f"expected 30 episodes, got {len(episodes)}")

    id_episodes = [e for e in episodes if e.get("split") == "id"]
    ood_episodes = [e for e in episodes if e.get("split") == "ood"]
    if len(id_episodes) != 22 or len(ood_episodes) != 8:
        raise ValueError(
            f"expected 22 ID and 8 OOD episodes, got {len(id_episodes)}/{len(ood_episodes)}"
        )

    id_templates = {e.get("template") for e in id_episodes}
    ood_templates = {e.get("template") for e in ood_episodes}
    overlap = id_templates & ood_templates
    if overlap:
        raise ValueError(f"ID and OOD templates overlap: {sorted(overlap)}")
    if not id_templates <= ID_TEMPLATES:
        raise ValueError(
            f"unknown ID template(s): {sorted(id_templates - ID_TEMPLATES)}"
        )
    if not ood_templates <= OOD_TEMPLATES:
        raise ValueError(
            f"unknown OOD template(s): {sorted(ood_templates - OOD_TEMPLATES)}"
        )

    for ep in episodes:
        eid = ep.get("episode_id", "<missing>")
        missing = REQUIRED_EPISODE_KEYS - set(ep)
        if missing:
            raise ValueError(f"{eid}: missing keys {sorted(missing)}")
        split = ep["split"]
        if split not in ("id", "ood"):
            raise ValueError(f"{eid}: split must be 'id' or 'ood'")
        if ep["ood"] is not (split == "ood"):
            raise ValueError(f"{eid}: ood flag disagrees with split")
        if ep["family"] not in FAMILIES:
            raise ValueError(f"{eid}: unknown family {ep['family']!r}")
        template = ep["template"]
        if split == "id" and template not in ID_TEMPLATES:
            raise ValueError(f"{eid}: template {template!r} is not an ID template")
        if split == "ood" and template not in OOD_TEMPLATES:
            raise ValueError(f"{eid}: template {template!r} is not an OOD template")
        if not isinstance(ep["instruction"], str) or not ep["instruction"].strip():
            raise ValueError(f"{eid}: empty instruction")
        if not isinstance(ep["max_steps"], int) or ep["max_steps"] <= 0:
            raise ValueError(f"{eid}: max_steps must be a positive integer")
        if not _finite_number(ep.get("reference_path_m", 0.0)) or ep["reference_path_m"] <= 0:
            raise ValueError(f"{eid}: reference_path_m must be positive")

        markers = ep["markers"]
        if not isinstance(markers, list) or not markers:
            raise ValueError(f"{eid}: at least one marker is required")
        marker_ids: set[str] = set()
        for marker in markers:
            mid = marker.get("marker_id")
            if not mid or mid in marker_ids:
                raise ValueError(f"{eid}: marker ids must be non-empty and unique")
            marker_ids.add(mid)
            pos = marker.get("position_ned_m")
            if not isinstance(pos, list) or len(pos) != 3 or not all(_finite_number(v) for v in pos):
                raise ValueError(f"{eid}: marker {mid} needs a finite 3-D NED position")

        pose = ep["start_pose"]
        pos = pose.get("position_ned_m")
        if not isinstance(pos, list) or len(pos) != 3 or not all(_finite_number(v) for v in pos):
            raise ValueError(f"{eid}: start position must be a finite 3-D NED position")
        if not _finite_number(pose.get("yaw_rad")):
            raise ValueError(f"{eid}: start yaw must be finite")

        predicates = ep["success_predicates"]
        if not isinstance(predicates, list) or not predicates:
            raise ValueError(f"{eid}: at least one success predicate is required")
        for pred in predicates:
            ptype = pred.get("type")
            if ptype not in PREDICATE_TYPES:
                raise ValueError(f"{eid}: unknown predicate type {ptype!r}")
            if ptype != "altitude":
                if pred.get("marker_id") not in marker_ids:
                    raise ValueError(f"{eid}: predicate references unknown marker")
            if ptype == "distance_to_marker":
                if pred["distance_m"] < 0 or pred["tolerance_m"] < 0:
                    raise ValueError(f"{eid}: distance/tolerance must be non-negative")
            elif ptype == "pass_side":
                if pred.get("side") not in ("left", "right"):
                    raise ValueError(f"{eid}: bad pass side")
                if pred["clearance_m"] < 0:
                    raise ValueError(f"{eid}: clearance must be non-negative")
            elif ptype == "orbit":
                if pred.get("direction") not in ("clockwise", "counterclockwise"):
                    raise ValueError(f"{eid}: bad orbit direction")
                if pred["min_angle_deg"] <= 0 or pred["radius_m"] <= 0:
                    raise ValueError(f"{eid}: orbit angle/radius must be positive")
            elif ptype == "hold_at_distance":
                if pred["duration_s"] <= 0 or pred["tolerance_m"] < 0:
                    raise ValueError(f"{eid}: hold duration must be positive")
            elif ptype == "altitude":
                if pred.get("direction") not in ("climb", "descend"):
                    raise ValueError(f"{eid}: bad altitude direction")
                if pred["tolerance_m"] < 0:
                    raise ValueError(f"{eid}: altitude tolerance must be non-negative")


def write_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    manifest: dict[str, Any] | None = None,
) -> Path:
    """Write a validated manifest to ``path`` (parents created as needed)."""
    target = Path(path)
    data = build_manifest() if manifest is None else manifest
    validate_manifest(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return target


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Load and validate a frozen manifest from disk."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"manifest not found: {target}\n"
            "Generate the frozen suite with: python -m scripts.eval.manifest --write"
        )
    data = json.loads(target.read_text(encoding="utf-8"))
    validate_manifest(data)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_MANIFEST_PATH),
        help="manifest path (default: scripts/eval/manifests/uav_intent_30.json)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the deterministic suite to --out",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="load and validate the manifest at --out",
    )
    args = parser.parse_args(argv)

    if args.write:
        written = write_manifest(args.out)
        print(f"wrote {written}")
    if args.check:
        data = load_manifest(args.out)
        id_n = sum(1 for e in data["episodes"] if e["split"] == "id")
        ood_n = sum(1 for e in data["episodes"] if e["split"] == "ood")
        print(
            f"ok: {len(data['episodes'])} episodes "
            f"({id_n} ID, {ood_n} OOD), templates disjoint"
        )
    if not args.write and not args.check:
        data = build_manifest()
        validate_manifest(data)
        print(json.dumps(
            {
                "suite_id": data["suite_id"],
                "episodes": len(data["episodes"]),
                "id": sum(1 for e in data["episodes"] if e["split"] == "id"),
                "ood": sum(1 for e in data["episodes"] if e["split"] == "ood"),
                "out": str(DEFAULT_MANIFEST_PATH),
            },
            indent=2,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
