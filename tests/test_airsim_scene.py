import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval.airsim_scene import clear_markers, ensure_markers


class FakeAirsim:
    class Vector3r:
        def __init__(self, x, y, z): self.x_val, self.y_val, self.z_val = x, y, z
    class Pose:
        def __init__(self, position, orientation): self.position, self.orientation = position, orientation
    @staticmethod
    def to_quaternion(*args): return tuple(args)


class FakeClient:
    def __init__(self): self.objects = {}; self.spawned = []; self.destroyed = []; self.materials = []
    def simListSceneObjects(self, regex): return list(self.objects)
    def simSpawnObject(self, name, asset, pose, scale, physics):
        self.objects[name] = (asset, pose, scale); self.spawned.append(name)
    def simSetObjectPose(self, name, pose, teleport=True): self.objects[name] = pose
    def simDestroyObject(self, name): self.destroyed.append(name); self.objects.pop(name, None)
    def simSetObjectMaterial(self, name, material): self.materials.append((name, material))


class AirSimSceneTest(unittest.TestCase):
    def test_spawn_move_clear(self):
        c = FakeClient(); a = FakeAirsim()
        markers = [{"marker_id": "C", "color": "red", "position_ned_m": [10, 0, -10], "scale": [0.2, 1.5, 1.5]}]
        names = ensure_markers(c, a, markers)
        self.assertEqual(names, ["dronevl_marker_C"])
        self.assertEqual(c.materials, [("dronevl_marker_C", "/Game/StarterContent/Materials/M_Red")])
        self.assertEqual(c.spawned, names)
        ensure_markers(c, a, [{"marker_id": "C", "position_ned_m": [12, 0, -10]}])
        self.assertEqual(c.spawned, ["dronevl_marker_C"])
        clear_markers(c, names)
        self.assertEqual(c.destroyed, names)
