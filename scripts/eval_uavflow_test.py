#!/usr/bin/env python3
"""Evaluate a served B0/GRPO policy on the 273 held-out UAV-Flow test tasks."""
from __future__ import annotations
import argparse, json, math, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

PYTHON = "/workspace/venv-uavflow/bin/python"
RUNNER = "/workspace/DroneVLA-uavflow/scripts/eval/uavflow_native_policy.py"
ROOT = "/workspace/UAV-Flow/UAV-Flow-Eval"


def dtw_ndtW(pred: np.ndarray, ref: np.ndarray) -> float:
    d = np.linalg.norm(pred[:, None, :] - ref[None, :, :], axis=-1)
    n, m = d.shape
    cost = np.full((n + 1, m + 1), np.inf); cost[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = d[i - 1, j - 1] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    scale = max(1.0, float(np.linalg.norm(ref[-1] - ref[0])))
    return float(math.exp(-cost[n, m] / (scale * max(n, m))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="/workspace/UAV-Flow/UAV-Flow-Eval/test_jsons")
    ap.add_argument("--out", default="/workspace/runs/eval_b0_grpo_100")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--proxy", default="http://127.0.0.1:5007/predict")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-delay", type=float, default=3.0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tasks = sorted(Path(args.test_dir).glob("*.json"))
    done = {p.stem for p in (out / "results").glob("*.json")} if (out / "results").exists() else set()

    def run(task_path: Path) -> dict:
        tid = task_path.stem
        task = json.loads(task_path.read_text())
        target = np.array((task.get("target_pos") or task.get("end_pos"))[:3], dtype=float)
        ref = np.array(task["reference_path_raw"], dtype=float)[:, :3]
        runs = []
        for k in range(args.rollouts):
            d = out / "runs" / tid / f"r{k}"; d.mkdir(parents=True, exist_ok=True)
            subprocess.run(["chown", "-R", "ue4:ue4", str(d)], check=False)
            cmd = ["runuser", "-u", "ue4", "--", "env", "DISPLAY=:99", "UnrealEnv=/workspace/UnrealEnv",
                   PYTHON, RUNNER, "--task", str(task_path), "--out", str(d), "--max-steps", str(args.steps),
                   "--image-cam", "0", "--action-mode", "absolute", "--state-format", "4", "--action-scale", "1.0",
                   "--proxy", args.proxy]
            for attempt in range(args.retries + 1):
                with (d / "runner.log").open("w") as f:
                    subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, timeout=120)
                if (d / "episode.jsonl").exists():
                    break
                # Kill only this task's runner/engine; other workers are untouched.
                subprocess.run(["pkill", "-f", f"uavflow_native_policy.py.*{d}"], check=False)
                time.sleep(args.retry_delay)
            if not (d / "episode.jsonl").exists():
                raise RuntimeError(f"no episode after {args.retries + 1} attempts: {tid}")
            rows = [json.loads(l) for l in (d / "episode.jsonl").read_text().splitlines() if l.strip()]
            pred = np.array([r["world_xyz"] for r in rows], dtype=float)
            final_m = float(np.linalg.norm(pred[-1] - target)) / 100.0
            runs.append({"steps": len(rows), "final_target_m": final_m,
                         "ndtW": dtw_ndtW(pred, ref), "success": bool(final_m < 2.0), "dir": str(d)})
        return {"task": tid, "rollouts": runs,
                "success": float(np.mean([r["success"] for r in runs])),
                "pass_at_3": bool(any(r["success"] for r in runs)),
                "final_target_m": float(np.mean([r["final_target_m"] for r in runs])),
                "ndtW": float(np.mean([r["ndtW"] for r in runs]))}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run, t): t for t in tasks if t.stem not in done}
        for fut in as_completed(futures):
            try:
                rec = fut.result(); results.append(rec)
                (out / "results").mkdir(exist_ok=True)
                (out / "results" / f"{rec['task']}.json").write_text(json.dumps(rec))
                print(json.dumps(rec), flush=True)
            except Exception as exc:
                print("ERR", futures[fut].stem, exc, flush=True)
    all_results = [json.loads(p.read_text()) for p in (out / "results").glob("*.json")]
    summary = {"tasks": len(all_results),
               "success_rate": float(np.mean([r["success"] for r in all_results])) if all_results else 0.0,
               "pass_at_3": float(np.mean([r.get("pass_at_3", False) for r in all_results])) if all_results else 0.0,
               "mean_ndtW": float(np.mean([r["ndtW"] for r in all_results])) if all_results else 0.0,
               "mean_final_target_m": float(np.mean([r["final_target_m"] for r in all_results])) if all_results else 0.0}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
