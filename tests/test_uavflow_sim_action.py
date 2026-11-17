import math

from integrations.uavflow_sim.action import Delta4, Pose6D, delta4_to_pose, pose_delta_to_delta4


def test_round_trip_straight():
    p0 = Pose6D(0, 0, -10, 0, 0, 0)
    d = Delta4(1.0, 0.0, 0.2, 0.1)
    p1 = delta4_to_pose(p0, d)
    back = pose_delta_to_delta4(p0, p1)
    assert math.isclose(back.dx, d.dx, abs_tol=1e-9)
    assert math.isclose(back.dy, d.dy, abs_tol=1e-9)
    assert math.isclose(back.dz, d.dz, abs_tol=1e-9)
    assert math.isclose(back.dyaw, d.dyaw, abs_tol=1e-9)


def test_body_frame_rotates_with_yaw():
    p0 = Pose6D(0, 0, -10, 0, 0, math.pi / 2)
    p1 = Pose6D(1, 0, -10, 0, 0, math.pi / 2)
    d = pose_delta_to_delta4(p0, p1)
    assert math.isclose(d.dx, 0.0, abs_tol=1e-9)
    assert math.isclose(d.dy, -1.0, abs_tol=1e-9)
