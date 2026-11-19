"""Native UAV-Flow-Sim pose <-> DroneVL delta4 conversion.

UAV-Flow-Sim uses an UnrealCV UE campus with 6-DoF pose/position control.
DroneVL's current contract is a 4-D body-FRD delta [dx,dy,dz,dyaw]. This module
keeps the conversion explicit so rollout/evaluation can use the native pose
without silently mixing frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose6D:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class Delta4:
    dx: float
    dy: float
    dz: float
    dyaw: float


def pose_delta_to_delta4(previous: Pose6D, current: Pose6D) -> Delta4:
    """World NED pose delta -> body-FRD 4-D delta using the previous yaw."""
    wx = current.x - previous.x
    wy = current.y - previous.y
    wz = current.z - previous.z
    c, s = math.cos(previous.yaw), math.sin(previous.yaw)
    return Delta4(
        dx=c * wx + s * wy,
        dy=-s * wx + c * wy,
        dz=wz,
        dyaw=wrap_angle(current.yaw - previous.yaw),
    )


def delta4_to_pose(previous: Pose6D, delta: Delta4) -> Pose6D:
    """Body-FRD 4-D delta -> world NED pose (roll/pitch held constant)."""
    c, s = math.cos(previous.yaw), math.sin(previous.yaw)
    wx = c * delta.dx - s * delta.dy
    wy = s * delta.dx + c * delta.dy
    return Pose6D(
        x=previous.x + wx,
        y=previous.y + wy,
        z=previous.z + delta.dz,
        roll=previous.roll,
        pitch=previous.pitch,
        yaw=wrap_angle(previous.yaw + delta.dyaw),
    )
