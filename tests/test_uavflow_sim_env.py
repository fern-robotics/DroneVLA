from integrations.uavflow_sim.action import Delta4
from integrations.uavflow_sim.env import UAVFlowSimEnv


class FakeClient:
    def __init__(self):
        self.location = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.calls = []

    def set_obj_location(self, name, location):
        self.location = list(location); self.calls.append(("loc", name, tuple(location)))

    def set_rotation(self, name, yaw):
        self.rotation = [0, yaw, 0]; self.calls.append(("rot", name, yaw))

    def set_cam(self, camera, location, rotation):
        self.calls.append(("cam", camera, tuple(location), tuple(rotation)))

    def get_image(self, camera, mode): return "rgb"

    def get_obj_location(self, name): return list(self.location)

    def get_obj_rotation(self, name): return list(self.rotation)


def test_reset_and_step():
    c = FakeClient()
    env = UAVFlowSimEnv(c, "player")
    task = {"initial_pos": [10, 20, 30, 0, 90, 0]}
    r = env.reset(task)
    assert r.image == "rgb"
    assert env.pose.x == 10 and env.pose.y == 20
    r = env.step(Delta4(1, 0, 0, 0))
    assert env.steps == 1
    assert any(x[0] == "loc" for x in c.calls)
