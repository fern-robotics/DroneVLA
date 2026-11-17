"""Native UAV-Flow-Sim environment wrapper.

The upstream UAV-Flow-Eval uses UnrealZoo Gym and UnrealCV. This wrapper keeps
the episode contract small enough for DroneVL policies: reset to a task JSON,
return RGB + native 6-DoF pose, and execute a local 4-D delta as an absolute
UnrealCV pose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .action import Delta4, Pose6D, delta4_to_pose, wrap_angle


class UnrealCVClient(Protocol):
    def set_obj_location(self, name: str, location: list[float]) -> Any: ...
    def set_rotation(self, name: str, yaw: float) -> Any: ...
    def set_cam(self, camera: int, location: list[float], rotation: list[float]) -> Any: ...
    def get_image(self, camera: int, mode: str) -> Any: ...
    def get_obj_location(self, name: str) -> list[float]: ...
    def get_obj_rotation(self, name: str) -> list[float]: ...


@dataclass
class StepResult:
    image: Any
    pose: Pose6D
    done: bool
    info: dict[str, Any]


class UAVFlowSimEnv:
    def __init__(self, client: UnrealCVClient, player: str, camera: int = 0, resolution: int = 256) -> None:
        self.client = client
        self.player = player
        self.camera = camera
        self.resolution = resolution
        self.initial = Pose6D(0, 0, 0, 0, 0, 0)
        self.pose = self.initial
        self.steps = 0

    def reset(self, task: dict[str, Any]) -> StepResult:
        p = task["initial_pos"]
        self.initial = Pose6D(float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]))
        self.pose = self.initial
        self.steps = 0
        self._place(self.initial)
        self._create_object(task)
        self._set_camera(self.initial)
        return self._observe(done=False)

    def step(self, action: Delta4) -> StepResult:
        local = delta4_to_pose(self.pose, action)
        # The 4-D delta is local to the current pose; convert back to world.
        from .action import pose_delta_to_delta4
        # delta4_to_pose already treats the input in world axes; reapply via initial yaw frame.
        c, s = __import__("math").cos(self.initial.yaw), __import__("math").sin(self.initial.yaw)
        world = Pose6D(
            self.initial.x + c * local.x - s * local.y,
            self.initial.y + s * local.x + c * local.y,
            local.z,
            local.roll,
            local.pitch,
            wrap_angle(self.initial.yaw + local.yaw),
        )
        self.pose = world
        self.steps += 1
        self._place(world)
        self._set_camera(world)
        return self._observe(done=False)

    def _observe(self, done: bool) -> StepResult:
        image = self.client.get_image(self.camera, "lit")
        loc = self.client.get_obj_location(self.player)
        rot = self.client.get_obj_rotation(self.player)
        pose = Pose6D(float(loc[0]), float(loc[1]), float(loc[2]),
                      float(rot[0]), float(rot[2]), float(rot[1]))
        return StepResult(image=image, pose=pose, done=done, info={"steps": self.steps})

    def _place(self, pose: Pose6D) -> None:
        self.client.set_obj_location(self.player, [pose.x, pose.y, pose.z])
        self.client.set_rotation(self.player, pose.yaw - 180.0)

    def _set_camera(self, pose: Pose6D) -> None:
        self.client.set_cam(self.camera, [pose.x, pose.y, pose.z],
                            [pose.roll, pose.pitch, pose.yaw])

    def _create_object(self, task: dict[str, Any]) -> None:
        # Object placement is handled by the upstream task setup; kept as a hook.
        return None
