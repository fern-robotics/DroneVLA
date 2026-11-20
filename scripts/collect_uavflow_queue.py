#!/usr/bin/env python3
"""Queue-based EG-GRPO collection: independent rollout jobs, no wave barrier."""
from __future__ import annotations
import argparse, json, math, os, signal, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.uavflow_sim.rl import rollout_reward


def run_job(args, task_path, task, k, out):
    d = Path(out) / Path(task_path).stem / f"r{k}"
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chmod", "-R", "a+rwX", str(d)], check=False)
    ref = task.get("reference_path_raw") or []
    steps = min(400, max(50, math.ceil(1.5 * len(ref)))) if ref else 20
    cmd = ["runuser", "-u", "ue4", "--", "env", "DISPLAY=:99",
           f"UnrealEnv={args.unreal_env}", args.python, args.runner,
           "--task", task_path, "--out", str(d), "--max-steps", str(steps),
           "--image-cam", "0", "--action-mode", "absolute", "--state-format", "4",
           "--action-scale", "1.0", "--save-frames"]
    log = (d / "runner.log").open("w")
    p = subprocess.Popen(cmd, cwd=args.root, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ep = d / "episode.jsonl"
    deadline = time.time() + min(args.timeout, max(300, steps * 6))
    stable_since = None
    try:
        while time.time() < deadline:
            if ep.exists():
                now = time.time(); mt = ep.stat().st_mtime
                if now - mt > 2.0:
                    if stable_since is None: stable_since = now
                    if now - stable_since >= 1.0: break
                else:
                    stable_since = None
            if p.poll() is not None: break
            time.sleep(1)
        if p.poll() is None:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            print("reaped", Path(task_path).stem, k, flush=True)
    finally:
        log.close()
    if not ep.exists():
        return None
    rows = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
    return {"dir": str(d), **rollout_reward([r["world_xyz"] for r in rows], task)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--python", default="/opt/venv-uavflow/bin/python")
    ap.add_argument("--runner", default="/workspace/DroneVLA-uavflow/scripts/eval/uavflow_native_policy.py")
    ap.add_argument("--root", default="/workspace/UAV-Flow/UAV-Flow-Eval")
    ap.add_argument("--unreal-env", default="/workspace-2/UnrealEnv")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    groups = []
    for ti, task_path in enumerate(manifest["tasks"]):
        gdir = out / Path(task_path).stem
        if (gdir / "group.json").exists():
            groups.append(json.loads((gdir / "group.json").read_text())); continue
        task = json.loads(Path(task_path).read_text())
        good = {}
        for attempt in range(args.retries + 1):
            pending = [k for k in range(4) if k not in good]
            if not pending: break
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(run_job, args, task_path, task, k, out): k for k in pending}
                for fut in as_completed(futs):
                    k = futs[fut]
                    try: r = fut.result()
                    except Exception as e: print("job_error", k, e, flush=True); r = None
                    if r: good[k] = r
            print("attempt", ti, attempt, len(good), flush=True)
        if len(good) < 4:
            print("incomplete", task_path, len(good), flush=True); continue
        group = {"group": len(groups), "task": task_path,
                 "rollouts": [good[k] for k in sorted(good)]}
        groups.append(group); (gdir / "group.json").write_text(json.dumps(group, indent=2))
        (out / "groups.json").write_text(json.dumps(groups, indent=2))
        print("group", len(groups), Path(task_path).stem, flush=True)
    print(json.dumps({"groups": len(groups), "out": str(out / "groups.json")}))


if __name__ == "__main__":
    main()
