import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval.evaluate import (  # noqa: E402
    MockEnv,
    ScriptedPolicy,
    run_episode,
    run_evaluation,
)
from scripts.eval.intents import (  # noqa: E402
    altitude,
    distance_to_marker,
    evaluate_episode,
    hold_at_distance,
    orbit,
    pass_side,
)
from scripts.eval.manifest import (  # noqa: E402
    ID_TEMPLATES,
    OOD_TEMPLATES,
    build_manifest,
    load_manifest,
)


def _orbit_points(marker, radius, start_deg, end_deg, n=181):
    points = []
    for i in range(n):
        bearing = math.radians(start_deg + (end_deg - start_deg) * i / (n - 1))
        points.append(
            [
                marker[0] + radius * math.cos(bearing),
                marker[1] + radius * math.sin(bearing),
                marker[2],
            ]
        )
    return points


class ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()

    def test_counts_and_explicit_splits(self):
        episodes = self.manifest["episodes"]
        self.assertEqual(len(episodes), 30)
        id_eps = [e for e in episodes if e["split"] == "id"]
        ood_eps = [e for e in episodes if e["split"] == "ood"]
        self.assertEqual(len(id_eps), 22)
        self.assertEqual(len(ood_eps), 8)
        self.assertTrue(all(e["ood"] is False for e in id_eps))
        self.assertTrue(all(e["ood"] is True for e in ood_eps))

    def test_id_ood_templates_are_disjoint(self):
        id_templates = {
            e["template"] for e in self.manifest["episodes"] if e["split"] == "id"
        }
        ood_templates = {
            e["template"] for e in self.manifest["episodes"] if e["split"] == "ood"
        }
        self.assertFalse(id_templates & ood_templates)
        self.assertTrue(id_templates <= ID_TEMPLATES)
        self.assertTrue(ood_templates <= OOD_TEMPLATES)

    def test_frozen_manifest_matches_generator(self):
        self.assertEqual(self.manifest, build_manifest())

    def test_episode_schema_and_coverage(self):
        required = {
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
        families = set()
        ids = set()
        for ep in self.manifest["episodes"]:
            self.assertTrue(required <= set(ep))
            self.assertNotIn(ep["episode_id"], ids)
            ids.add(ep["episode_id"])
            families.add(ep["family"])
            self.assertGreater(ep["max_steps"], 0)
            self.assertTrue(ep["instruction"])
            self.assertGreaterEqual(len(ep["markers"]), 1)
            for marker in ep["markers"]:
                self.assertEqual(len(marker["position_ned_m"]), 3)
                self.assertTrue(all(math.isfinite(v) for v in marker["position_ned_m"]))
            self.assertEqual(len(ep["start_pose"]["position_ned_m"]), 3)
            self.assertGreaterEqual(len(ep["success_predicates"]), 1)
            for pred in ep["success_predicates"]:
                self.assertIn(
                    pred["type"],
                    {
                        "distance_to_marker",
                        "pass_side",
                        "orbit",
                        "hold_at_distance",
                        "altitude",
                    },
                )
        self.assertEqual(
            families,
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
            },
        )


class PredicateTest(unittest.TestCase):
    def test_distance_to_marker(self):
        marker = [10.0, 0.0, -10.0]
        good = distance_to_marker(
            [[0.0, 0.0, -10.0], [4.0, 0.0, -10.0], [6.0, 0.0, -10.0]],
            marker,
            6.0,
            0.5,
        )
        self.assertTrue(good.satisfied)
        self.assertEqual(good.end_index, 1)
        self.assertAlmostEqual(good.progress, 1.0)

        bad = distance_to_marker(
            [[0.0, 0.0, -10.0], [6.0, 0.0, -10.0]], marker, 2.0, 0.25
        )
        self.assertFalse(bad.satisfied)
        self.assertEqual(bad.end_index, -1)
        self.assertLess(bad.progress, 1.0)

    def test_pass_side_plane(self):
        marker = [10.0, 0.0, -10.0]
        trajectory = [
            [0.0, 0.0, -10.0],
            [8.0, -2.0, -10.0],
            [10.0, -2.0, -10.0],
            [12.0, -2.0, -10.0],
        ]
        left = pass_side(
            trajectory, marker, "left", 1.5, origin_ned=[0.0, 0.0, -10.0]
        )
        self.assertTrue(left.satisfied)
        self.assertTrue(left.detail["crossed"])
        self.assertLessEqual(left.detail["crossing_lateral_m"], -1.5)

        wrong_side = pass_side(
            trajectory, marker, "right", 1.5, origin_ned=[0.0, 0.0, -10.0]
        )
        self.assertFalse(wrong_side.satisfied)
        self.assertAlmostEqual(wrong_side.progress, 0.6)

    def test_orbit_angle_and_direction(self):
        marker = [10.0, 0.0, -10.0]
        # Bearings increase from 180 to 380 degrees => clockwise in NED.
        clockwise = _orbit_points(marker, 3.5, 180.0, 380.0)
        cw = orbit(clockwise, marker, "clockwise", 180.0, 3.5)
        self.assertTrue(cw.satisfied)
        self.assertAlmostEqual(cw.detail["accumulated_deg"], 180.0, delta=1.0)
        cw_200 = orbit(clockwise, marker, "clockwise", 200.0, 3.5)
        self.assertTrue(cw_200.satisfied)
        self.assertAlmostEqual(
            cw_200.detail["accumulated_deg"], 200.0, delta=1.0
        )

        wrong = orbit(clockwise, marker, "counterclockwise", 180.0, 3.5)
        self.assertFalse(wrong.satisfied)
        self.assertLess(wrong.progress, 1.0)

        # Bearings decrease from 180 to -20 degrees => counterclockwise.
        ccw_points = _orbit_points(marker, 3.5, 180.0, -20.0)
        ccw = orbit(ccw_points, marker, "counterclockwise", 180.0, 3.5)
        self.assertTrue(ccw.satisfied)

    def test_hold_at_distance(self):
        marker = [10.0, 0.0, -10.0]
        states = [
            {"position_ned_m": [6.0, 0.0, -10.0], "t_s": i * 0.2}
            for i in range(21)
        ]
        good = hold_at_distance(states, marker, 4.0, 0.5, 2.0)
        self.assertTrue(good.satisfied)
        self.assertAlmostEqual(good.detail["best_hold_s"], 2.0)

        short = hold_at_distance(states, marker, 4.0, 0.5, 6.0)
        self.assertFalse(short.satisfied)
        self.assertLess(short.progress, 1.0)

    def test_altitude(self):
        climb = [
            [0.0, 0.0, -10.0],
            [0.0, 0.0, -11.5],
            [0.0, 0.0, -13.0],
        ]
        good = altitude(climb, -13.0, 0.2)
        self.assertTrue(good.satisfied)
        self.assertEqual(good.end_index, 2)
        self.assertAlmostEqual(good.detail["achieved_delta_m"], 3.0)

        bad = altitude(climb, -8.0, 0.2)
        self.assertFalse(bad.satisfied)


class EvaluationTest(unittest.TestCase):
    def _composed_episode(self):
        return {
            "episode_id": "unit-composed",
            "split": "id",
            "ood": False,
            "family": "composed",
            "template": "unit:test",
            "instruction": "Approach, then pass left.",
            "reference_frame": "world_NED",
            "markers": [
                {
                    "marker_id": "M1",
                    "color": "red",
                    "shape": "circle",
                    "position_ned_m": [10.0, 0.0, -10.0],
                    "size_m": 1.5,
                }
            ],
            "start_pose": {"position_ned_m": [0.0, 0.0, -10.0], "yaw_rad": 0.0},
            "success_predicates": [
                {
                    "type": "distance_to_marker",
                    "marker_id": "M1",
                    "distance_m": 6.0,
                    "tolerance_m": 0.5,
                },
                {
                    "type": "pass_side",
                    "marker_id": "M1",
                    "side": "left",
                    "clearance_m": 1.5,
                },
            ],
            "reference_path_m": 10.0,
            "max_steps": 100,
            "dt_s": 0.2,
        }

    def test_evaluate_episode_returns_full_metrics(self):
        trajectory = [
            {"position_ned_m": [0.0, 0.0, -10.0], "t_s": 0.0},
            {"position_ned_m": [4.0, 0.0, -10.0], "t_s": 0.2},
            {"position_ned_m": [8.0, -2.0, -10.0], "t_s": 0.4},
            {"position_ned_m": [10.0, -2.0, -10.0], "t_s": 0.6},
            {"position_ned_m": [12.0, -2.0, -10.0], "t_s": 0.8},
        ]
        result = evaluate_episode(self._composed_episode(), trajectory)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.intent_score, 1.0)
        self.assertAlmostEqual(result.subgoal_completion, 1.0)
        self.assertEqual(result.completed_subgoals, 2)
        self.assertEqual(result.total_subgoals, 2)
        self.assertFalse(result.collision)
        self.assertEqual(result.invalid_action_count, 0)
        self.assertGreater(result.path_length_m, 0.0)
        self.assertIsNotNone(result.path_efficiency)
        self.assertGreater(result.path_efficiency, 0.0)
        self.assertLessEqual(result.path_efficiency, 1.0)
        self.assertEqual(len(result.subgoals), 2)

    def test_evaluate_episode_flags_collision_and_invalid_actions(self):
        trajectory = [
            {"position_ned_m": [0.0, 0.0, -10.0], "t_s": 0.0},
            {
                "position_ned_m": [4.0, 0.0, -10.0],
                "t_s": 0.2,
                "collision": True,
                "invalid_action": True,
            },
        ]
        result = evaluate_episode(self._composed_episode(), trajectory)
        self.assertFalse(result.success)
        self.assertTrue(result.collision)
        self.assertEqual(result.invalid_action_count, 1)


class MockEvaluatorTest(unittest.TestCase):
    def test_mock_evaluator_output_schema(self):
        manifest = load_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eval"
            summary = run_evaluation(
                manifest,
                out,
                mode="mock",
                episode_ids=["id-approach-01"],
            )
            self.assertEqual(summary["episodes"], 1)
            self.assertEqual(summary["successes"], 1)
            for key in (
                "success_rate",
                "mean_intent_score",
                "collisions",
                "interventions",
                "latency_ms_median",
                "latency_ms_p95",
                "mean_path_efficiency",
                "per_episode",
            ):
                self.assertIn(key, summary)

            self.assertTrue((out / "episodes" / "id-approach-01.jsonl").exists())
            self.assertTrue((out / "per_episode.jsonl").exists())
            self.assertTrue((out / "summary.csv").exists())
            self.assertTrue((out / "summary.json").exists())

            step_lines = (out / "episodes" / "id-approach-01.jsonl").read_text().splitlines()
            self.assertGreater(len(step_lines), 0)
            step = json.loads(step_lines[0])
            for key in (
                "episode_id",
                "step",
                "t_s",
                "position_ned_m",
                "yaw_rad",
                "action",
                "collision",
                "intervention",
                "invalid_action",
                "policy_latency_ms",
            ):
                self.assertIn(key, step)

            row = json.loads((out / "per_episode.jsonl").read_text().splitlines()[0])
            for key in (
                "episode_id",
                "success",
                "intent_score",
                "subgoal_completion",
                "collisions",
                "interventions",
                "latency_ms_median",
                "path_efficiency",
            ):
                self.assertIn(key, row)

            header = (out / "summary.csv").read_text().splitlines()[0].split(",")
            for key in (
                "success",
                "intent_score",
                "collisions",
                "interventions",
                "latency_ms_median",
                "path_efficiency",
            ):
                self.assertIn(key, header)

    def test_successful_mock_approach_episode(self):
        manifest = load_manifest()
        episode = next(
            e for e in manifest["episodes"] if e["episode_id"] == "id-approach-01"
        )
        env = MockEnv(markers=episode["markers"], dt_s=episode["dt_s"])
        run = run_episode(env, ScriptedPolicy(episode), episode)
        self.assertTrue(run.summary["success"])
        self.assertAlmostEqual(run.summary["intent_score"], 1.0)
        self.assertEqual(run.summary["completed_subgoals"], 1)
        self.assertEqual(run.summary["total_subgoals"], 1)
        self.assertEqual(run.summary["collisions"], 0)
        self.assertGreater(run.summary["steps"], 0)
        self.assertGreater(run.summary["path_length_m"], 0.0)
        self.assertIsNotNone(run.summary["path_efficiency"])

    def test_full_mock_suite_is_deterministic_and_succeeds(self):
        manifest = load_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eval"
            summary = run_evaluation(manifest, out, mode="mock")
            self.assertEqual(summary["episodes"], 30)
            self.assertEqual(summary["successes"], 30)
            self.assertEqual(summary["collisions"], 0)
            self.assertEqual(summary["interventions"], 0)
            self.assertEqual(summary["id"]["episodes"], 22)
            self.assertEqual(summary["ood"]["episodes"], 8)
            for row in summary["per_episode"]:
                self.assertTrue(row["success"], row["episode_id"])
                self.assertAlmostEqual(row["intent_score"], 1.0, places=6)
                self.assertFalse(math.isnan(row["latency_ms_median"]))

            # A second run must produce identical trajectories (latency is the
            # only intentionally non-deterministic field).
            first = (out / "episodes" / "id-orbit-cw-01.jsonl").read_text()
            out2 = Path(tmp) / "eval2"
            run_evaluation(
                manifest, out2, mode="mock", episode_ids=["id-orbit-cw-01"]
            )
            second = (out2 / "episodes" / "id-orbit-cw-01.jsonl").read_text()

            def strip_latency(text):
                return [
                    {
                        key: value
                        for key, value in json.loads(line).items()
                        if key != "policy_latency_ms"
                    }
                    for line in text.splitlines()
                ]

            self.assertEqual(strip_latency(first), strip_latency(second))

    def test_airsim_module_import_is_lazy(self):
        import scripts.eval.airsim_env as airsim_env

        self.assertTrue(hasattr(airsim_env, "AirSimEnv"))
        if importlib.util.find_spec("airsim") is None:
            with self.assertRaises(airsim_env.AirSimUnavailableError) as ctx:
                airsim_env.AirSimEnv()
            self.assertIn("pip install airsim", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
