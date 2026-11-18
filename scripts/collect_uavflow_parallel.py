#!/usr/bin/env python3
"""Collect 4-rollout groups for a 100-task manifest using 4 native engines."""
from __future__ import annotations
import argparse, json, math, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.uavflow_sim.rl import rollout_reward

PYTHON = "/workspace/venv-uavflow/bin/python"
PARALLEL = "/workspace/DroneVLA-uavflow/scripts/run_parallel_rollouts.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--adaptive-steps", action="store_true", default=True)
    ap.add_argument("--no-adaptive-steps", dest="adaptive_steps", action="store_false")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    groups = []
    for i, task_path in enumerate(manifest["tasks"]):
        tid = Path(task_path).stem
        gdir = out / tid
        if (gdir / "group.json").exists():
            groups.append(json.loads((gdir / "group.json").read_text()))
            continue
        task = json.loads(Path(task_path).read_text())
        ref = task.get("reference_path_raw") or []
        steps = min(400, max(50, int(math.ceil(1.5 * len(ref))))) if args.adaptive_steps and ref else args.steps
        cmd = [PYTHON, PARALLEL, "--task", task_path, "--out", str(gdir),
               "--workers", str(args.workers), "--steps", str(steps), "--stagger", "3"]
        try:
            subprocess.run(cmd, stdin=subprocess.DEVNULL, timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired:
            print("timeout", tid, flush=True)
        task = json.loads(Path(task_path).read_text())
        rollouts = []
        for w in range(args.workers):
            ep = gdir / f"w{w}" / "episode.jsonl"
            if not ep.exists():
                continue
            rows = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
            rollouts.append({"dir": str(gdir / f"w{w}"),
                             **rollout_reward([r["world_xyz"] for r in rows], task)})
        if rollouts:
            group = {"group": len(groups), "task": task_path, "rollouts": rollouts}
            groups.append(group)
            (gdir / "group.json").write_text(json.dumps(group, indent=2))
        (out / "groups.json").write_text(json.dumps(groups, indent=2))
        print(i, tid, len(rollouts), flush=True)
    print(json.dumps({"groups": len(groups), "out": str(out / "groups.json")}))


if __name__ == "__main__":
    main()
