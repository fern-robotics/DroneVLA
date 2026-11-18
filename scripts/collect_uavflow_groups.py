#!/usr/bin/env python3
"""Collect native UAV-Flow B0 rollout groups and native rewards."""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.uavflow_sim.rl import rollout_reward

PYTHON = "/workspace/venv-uavflow/bin/python"
RUNNER = "/workspace/DroneVLA-uavflow/scripts/eval/uavflow_native_policy.py"
ROOT = "/workspace/UAV-Flow/UAV-Flow-Eval"


def run_one(task: str, out: Path, steps: int, cmd_prefix: list[str]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chown", "-R", "ue4:ue4", str(out)], check=False)
    cmd = cmd_prefix + [PYTHON, RUNNER, "--task", task, "--out", str(out),
                        "--max-steps", str(steps), "--image-cam", "0",
                        "--action-mode", "absolute", "--state-format", "4",
                        "--action-scale", "1.0", "--save-frames"]
    log = out / "runner.log"
    with log.open("w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=1800, check=False)
    rows = [json.loads(l) for l in (out / "episode.jsonl").read_text().splitlines() if l.strip()]
    t = json.loads(Path(task).read_text())
    target = t.get("target_pos")
    final_dist = None
    if target and rows:
        w = rows[-1]["world_xyz"]
        final_dist = math.hypot(w[0] - target[0], w[1] - target[1]) / 100.0
    rec = {"dir": str(out), "steps": len(rows), "final_target_distance_m": final_dist}
    if rows and t.get("reference_path_raw"):
        rec.update(rollout_reward([r["world_xyz"] for r in rows], t))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chown", "-R", "ue4:ue4", str(out)], check=False)
    prefix = ["runuser", "-u", "ue4", "--", "env", "DISPLAY=:99", "UnrealEnv=/workspace/UnrealEnv"]
    groups = []
    for g in range(args.groups):
        rollouts = []
        for k in range(args.group_size):
            d = out / f"g{g}_r{k}"
            try:
                rec = run_one(args.task, d, args.steps, prefix)
            except Exception as exc:
                rec = {"dir": str(d), "steps": 0, "final_target_distance_m": None, "error": str(exc)}
            dist = rec.get("final_target_distance_m")
            if "reward" not in rec:
                rec["reward"] = 0.0 if dist is None else -float(dist)
            rollouts.append(rec)
            print(g, k, rec, flush=True)
        groups.append({"group": g, "task": args.task, "rollouts": rollouts})
    (out / "groups.json").write_text(json.dumps(groups, indent=2))
    print("wrote", out / "groups.json")


if __name__ == "__main__":
    main()
