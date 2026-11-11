"""Provision UAV intent markers in an AirSimNH scene.

The evaluation manifest is the source of truth for marker positions. Assets are
spawned once and then moved for each episode; they are not part of the policy
observation code.
"""
from __future__ import annotations

from typing import Any, Iterable


def _pose(airsim: Any, position: Iterable[float], yaw: float = 0.0) -> Any:
    x, y, z = [float(v) for v in position]
    return airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0.0, 0.0, float(yaw)))


def ensure_markers(client: Any, airsim: Any, markers: list[dict[str, Any]], prefix: str = "dronevl_marker") -> list[str]:
    """Spawn or move marker cubes; returns the object names."""
    existing = set()
    try:
        existing = set(client.simListSceneObjects(".*"))
    except Exception:
        pass
    names = []
    for marker in markers:
        name = f"{prefix}_{marker['marker_id']}"
        names.append(name)
        scale = marker.get("scale", [0.2, 1.0, 1.0])
        pose = _pose(airsim, marker["position_ned_m"], float(marker.get("yaw_rad", 0.0)))
        if name in existing:
            client.simSetObjectPose(name, pose, teleport=True)
        else:
            client.simSpawnObject(name, marker.get("asset", "Cube"), pose, airsim.Vector3r(*[float(v) for v in scale]), False)
    return names


def clear_markers(client: Any, names: Iterable[str]) -> None:
    for name in names:
        try:
            client.simDestroyObject(name)
        except Exception:
            pass
