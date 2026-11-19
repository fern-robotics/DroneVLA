#!/usr/bin/env python3
"""Launch multiple native UAV-Flow rollouts with per-engine UnrealCV ports.

The UnrealCV launcher picks the next free port (9000, 9001, ...). Workers are
started with a stagger so the port selection does not race.
"""
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path

PYTHON = "/workspace/venv-uavflow/bin/python"
RUNNER = "/workspace/DroneVLA-uavflow/scripts/eval/uavflow_native_policy.py"
ROOT = "/workspace/UAV-Flow/UAV-Flow-Eval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--stagger", type=float, default=8.0)
    ap.add_argument("--worker-timeout", type=float, default=90.0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    prefix = ["runuser", "-u", "ue4", "--", "env", "DISPLAY=:99", "UnrealEnv=/workspace/UnrealEnv"]
    procs = []
    for i in range(args.workers):
        d = out / f"w{i}"
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["chown", "-R", "ue4:ue4", str(d)], check=False)
        cmd = prefix + [PYTHON, RUNNER, "--task", args.task, "--out", str(d),
                        "--max-steps", str(args.steps), "--image-cam", "0",
                        "--action-mode", "absolute", "--state-format", "4",
                        "--action-scale", "1.0", "--save-frames"]
        log = (d / "runner.log").open("w")
        procs.append((i, subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(args.stagger)
    results = []
    for i, p, log in procs:
        try:
            code = p.wait(timeout=args.worker_timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            subprocess.run(["pkill", "-f", f"uavflow_native_policy.py.*{out / f'w{i}'}"], check=False)
            code = -9
        log.close()
        results.append({"worker": i, "exit": code,
                        "episode": str(out / f"w{i}" / "episode.jsonl")})
    (out / "parallel.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
