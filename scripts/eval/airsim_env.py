#!/usr/bin/env python3
"""Thin AirSimNH environment adapter for the UAV intent eval suite.

This module intentionally does **not** import the ``airsim`` package at import
time.  The package is loaded only when an ``AirSimEnv`` is constructed, so the
mock evaluator and the unit tests can run with just numpy/Pillow installed.

Markers are part of the AirSimNH scene manifest; this adapter never spawns or
moves marker assets.  It only:

* resets the vehicle to a manifest start pose,
* captures an RGB frame when asked,
* steps a four-dimensional delta action ``[dx, dy, dz, dyaw]``, and
* exposes world-NED position and yaw.

The delta is applied kinematically with ``simSetVehiclePose`` for deterministic
evaluation.  This is an evaluation bridge, not a low-level flight controller;
use AirSim's velocity/position controllers when real dynamics are required.
"""
from __future__ import annotations

import io
import math
import time
from typing import Any

import numpy as np


class AirSimUnavailableError(RuntimeError):
    """Raised when the ``airsim`` package or an AirSimNH RPC endpoint is absent."""


def _load_airsim() -> Any:
    try:
        import airsim  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only without AirSim
        raise AirSimUnavailableError(
            "The AirSim Python package is not importable. Install it with "
            "`pip install airsim` (and make sure the `airsim` package matches "
            "your AirSimNH version), then start the AirSimNH environment with "
            "its RPC server enabled on 127.0.0.1:41451. "
            f"Original import error: {exc}"
        ) from exc
    return airsim


def _wrap_angle(angle: float) -> float:
    value = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if value <= -math.pi:
        value += 2.0 * math.pi
    return value


def _normalize_action(action: Any) -> np.ndarray:
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
        raise ValueError("action must be [dx, dy, dz, dyaw]")
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("action contains non-finite values")
    return arr


class AirSimEnv:
    """Thin AirSimNH wrapper used by ``evaluate.py --airsim``."""

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 41451,
        vehicle_name: str = "Drone0",
        camera_name: str = "0",
        dt_s: float = 0.2,
        max_step_m: float = 0.5,
        max_yaw_step_rad: float = 0.35,
        timeout_s: float = 10.0,
    ) -> None:
        self.airsim = _load_airsim()
        self.ip = ip
        self.port = int(port)
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self.dt_s = float(dt_s)
        self.max_step_m = float(max_step_m)
        self.max_yaw_step_rad = float(max_yaw_step_rad)
        self.timeout_s = float(timeout_s)
        self.steps = 0
        self.last_collision = False
        self._teleport_step = True

        try:
            self.client = self.airsim.MultirotorClient(ip=ip, port=port)
            self.client.confirmConnection()
            try:
                vehicles = self.client.listVehicles()
                if vehicles and self.vehicle_name not in vehicles:
                    self.vehicle_name = vehicles[0]
            except Exception:
                pass
        except Exception as exc:  # pragma: no cover - requires a live server
            raise AirSimUnavailableError(
                "Could not connect to an AirSimNH RPC endpoint at "
                f"{ip}:{port}. Start AirSimNH, enable its RPC server, and "
                "check that the vehicle name/camera name match the scene. "
                f"Original error: {exc}"
            ) from exc

        self._enable_control()

    def _enable_control(self) -> None:
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _pose(self) -> Any:
        return self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)

    def position_ned_m(self) -> list[float]:
        pose = self._pose()
        return [
            float(pose.position.x_val),
            float(pose.position.y_val),
            float(pose.position.z_val),
        ]

    def yaw_rad(self) -> float:
        pose = self._pose()
        return float(self.airsim.to_eularian_angles(pose.orientation)[2])

    @property
    def position(self) -> list[float]:
        """Current world-NED position ``[x, y, z]``."""
        return self.position_ned_m()

    @property
    def yaw(self) -> float:
        """Current yaw in radians (clockwise-positive around +z)."""
        return self.yaw_rad()

    def observe(self, capture_image: bool = False) -> dict[str, Any]:
        """Return the current position/yaw, optionally with an RGB frame."""
        pose = self._pose()
        observation: dict[str, Any] = {
            "position_ned_m": [
                float(pose.position.x_val),
                float(pose.position.y_val),
                float(pose.position.z_val),
            ],
            "yaw_rad": float(self.airsim.to_eularian_angles(pose.orientation)[2]),
            "t_s": float(self.steps) * self.dt_s,
            "rgb": None,
            "collision": bool(self.last_collision),
        }
        if capture_image:
            observation["rgb"] = self.capture_rgb()
        return observation

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------
    def reset(self, start_pose: dict[str, Any]) -> dict[str, Any]:
        """Reset the vehicle to a manifest start pose."""
        pose = start_pose["position_ned_m"]
        yaw = float(start_pose.get("yaw_rad", 0.0))
        try:
            self.client.reset()
        except Exception:
            # Some AirSim builds do not expose reset for multirotors; setting
            # the pose below is sufficient for this evaluation adapter.
            pass
        self._enable_control()
        self._set_pose(pose, yaw)
        self.steps = 0
        self.last_collision = False
        self._teleport_step = True
        # Give AirSimNH one rendered frame after teleporting.
        time.sleep(min(0.05, self.timeout_s))
        return self.observe(capture_image=False)

    def _set_pose(self, position: Any, yaw: float) -> None:
        pose = self.airsim.Pose(
            self.airsim.Vector3r(
                float(position[0]), float(position[1]), float(position[2])
            ),
            self.airsim.to_quaternion(0.0, 0.0, float(yaw)),
        )
        self.client.simSetVehiclePose(
            pose, ignore_collision=True, vehicle_name=self.vehicle_name
        )

    def step(self, action: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply a four-dimensional delta action.

        Returns ``(observation, info)`` where ``info`` contains ``collision``,
        ``intervention`` (delta clipped by the safety limits) and
        ``invalid_action``.
        """
        info = {
            "collision": False,
            "intervention": False,
            "invalid_action": False,
        }
        try:
            arr = _normalize_action(action)
        except (TypeError, ValueError):
            info["invalid_action"] = True
            self.steps += 1
            info["collision"] = self.last_collision
            return self.observe(capture_image=False), info

        pose = self._pose()
        position = np.array(
            [pose.position.x_val, pose.position.y_val, pose.position.z_val],
            dtype=float,
        )
        yaw = float(self.airsim.to_eularian_angles(pose.orientation)[2])

        delta = arr[:3].copy()
        norm = float(np.linalg.norm(delta))
        if norm > self.max_step_m + 1e-9:
            delta *= self.max_step_m / norm
            info["intervention"] = True
        dyaw = float(arr[3])
        if abs(dyaw) > self.max_yaw_step_rad + 1e-9:
            dyaw = math.copysign(self.max_yaw_step_rad, dyaw)
            info["intervention"] = True

        new_position = position + delta
        if not np.all(np.isfinite(new_position)):
            info["invalid_action"] = True
            self.steps += 1
            info["collision"] = self.last_collision
            return self.observe(capture_image=False), info

        self._set_pose(new_position, _wrap_angle(yaw + dyaw))
        self.steps += 1
        time.sleep(0.05)
        try:
            collision_info = self.client.simGetCollisionInfo(
                vehicle_name=self.vehicle_name
            )
            self.last_collision = bool(collision_info.has_collided)
        except Exception:
            self.last_collision = False
        if self._teleport_step:
            # simGetCollisionInfo can stay latched for one frame after a teleport.
            self.last_collision = False
            self._teleport_step = False
        info["collision"] = self.last_collision
        return self.observe(capture_image=False), info

    # ------------------------------------------------------------------
    # RGB capture
    # ------------------------------------------------------------------
    def capture_rgb(self) -> np.ndarray:
        """Return the current RGB camera frame as ``uint8[H, W, 3]``."""
        response = self.client.simGetImage(
            self.camera_name,
            self.airsim.ImageType.Scene,
            vehicle_name=self.vehicle_name,
        )
        if not response:
            raise AirSimUnavailableError(
                "AirSim returned an empty image. Check that the camera name "
                f"{self.camera_name!r} exists for vehicle {self.vehicle_name!r}."
            )
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover - Pillow is a test dependency
            raise AirSimUnavailableError(
                "Pillow is required to decode AirSim RGB frames: "
                "`pip install pillow`."
            ) from exc
        image = Image.open(io.BytesIO(response)).convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    def close(self) -> None:
        """Release API control without stopping the simulator."""
        try:
            self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
        except Exception:
            pass
        try:
            self.client.armDisarm(False, vehicle_name=self.vehicle_name)
        except Exception:
            pass
