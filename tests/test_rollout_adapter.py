import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.airsim_rollout import MockEnv, MockPolicy, run_episode


class RolloutAdapterTest(unittest.TestCase):
    def test_mock_episode_logs_every_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = run_episode(MockEnv(), MockPolicy(), "Move forward", out, "ep-0", max_steps=100)
            self.assertTrue(summary["success"])
            lines = (out / "ep-0.jsonl").read_text().splitlines()
            self.assertGreater(len(lines), 0)
            first = json.loads(lines[0])
            for key in ("episode_id", "step", "instruction", "state", "action", "tokens", "logprobs", "reward"):
                self.assertIn(key, first)
            self.assertIsNotNone(first["tokens"])


if __name__ == "__main__":
    unittest.main()
