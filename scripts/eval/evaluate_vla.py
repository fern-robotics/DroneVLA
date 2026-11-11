#!/usr/bin/env python3
"""Evaluate a VLA policy on the frozen 30-episode AirSimNH intent suite.

Unlike ``evaluate.py`` (which tests the evaluator with a scripted policy), this
CLI drives a real policy server and logs every decision, action, token and
log-probability it receives.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.evaluate import MockEnv, ScriptedPolicy
from scripts.eval.intents import evaluate_episode
from scripts.eval.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from scripts.eval.vla_policy import make_policy


def run(args: argparse.Namespace) -> dict:
    manifest = load_manifest(Path(args.manifest))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_airsim = args.env == "airsim"
    if use_airsim:
        from scripts.eval.airsim_env import AirSimEnv
        env = AirSimEnv(ip=args.airsim_ip, port=args.airsim_port, dt_s=args.dt)
    else:
        env = MockEnv()

    video = None
    if args.video:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("--video requires opencv-python") from exc
    rows = []
    for episode in manifest["episodes"]:
        if args.episode_id and episode["episode_id"] != args.episode_id:
            continue
        scripted = ScriptedPolicy(episode)
        marker_names = []
        if use_airsim:
            from scripts.eval.airsim_scene import ensure_markers
            marker_names = ensure_markers(env.client, env.airsim, episode.get("markers", []))
        policy = make_policy(args.policy, scripted=scripted, host=args.host, port=args.port, url=args.policy_url)
        trajectory = []
        obs = env.reset(episode["start_pose"])
        start = np.asarray(obs["position_ned_m"], dtype=float)
        start_yaw = float(obs["yaw_rad"])
        ep_log = (out / "episodes" / f"{episode['episode_id']}.jsonl")
        ep_log.parent.mkdir(parents=True, exist_ok=True)
        with ep_log.open("w") as log:
            for step in range(int(episode.get("max_steps", 200))):
                obs = env.observe(capture_image=use_airsim and args.capture_image) if use_airsim else env.observe()
                if args.video and obs.get("rgb") is not None:
                    frame = np.asarray(obs["rgb"])
                    if video is None:
                        h, w = frame.shape[:2]
                        video = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                pos = np.asarray(obs["position_ned_m"], dtype=float)
                state = [pos[0] - start[0], pos[1] - start[1], pos[2] - start[2],
                         (float(obs["yaw_rad"]) - start_yaw + np.pi) % (2 * np.pi) - np.pi]
                observation = {"rgb": obs.get("rgb"), "state": state, "position_ned_m": pos.tolist(),
                               "yaw_rad": float(obs["yaw_rad"])}
                result = policy.act(observation, episode["instruction"])
                actions = np.asarray(result.get("actions", []), dtype=float)
                if actions.ndim == 1:
                    actions = actions[None, :]
                for a in actions[: args.replan_steps]:
                    obs, info = env.step(a.tolist())
                    trajectory.append({**obs, **info, "action": a.tolist()})
                    log.write(json.dumps({
                        "episode_id": episode["episode_id"], "step": step, "action": a.tolist(),
                        "state": state, "tokens": result.get("tokens"), "logprobs": result.get("logprobs"),
                        "latency_ms": result.get("policy_latency_ms"), "info": info,
                    }) + "\n")
                    if info.get("collision"):
                        break
                if trajectory and trajectory[-1].get("collision"):
                    break
        if use_airsim:
            from scripts.eval.airsim_scene import clear_markers
            clear_markers(env.client, marker_names)
        evaluation = evaluate_episode(episode, trajectory, dt_s=args.dt)
        row = {
            "episode_id": episode["episode_id"], "split": episode.get("split"), "ood": episode.get("ood"),
            "success": evaluation.success, "intent_score": evaluation.intent_score,
            "collision": any(t.get("collision") for t in trajectory),
            "steps": len(trajectory), "subgoal_completion": evaluation.subgoal_completion,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    if video is not None:
        video.release()

    summary = {
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])) if rows else 0.0,
        "mean_intent_score": float(np.mean([r["intent_score"] for r in rows])) if rows else 0.0,
        "collisions": int(sum(r["collision"] for r in rows)),
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    ap.add_argument("--out", default="runs/eval_vla")
    ap.add_argument("--env", choices=["mock", "airsim"], default="mock")
    ap.add_argument("--policy", choices=["mock", "websocket", "http"], default="mock")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--policy-url", default="")
    ap.add_argument("--episode-id", default="")
    ap.add_argument("--replan-steps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--airsim-ip", default="127.0.0.1")
    ap.add_argument("--airsim-port", type=int, default=41451)
    ap.add_argument("--capture-image", action="store_true", help="capture RGB for a VLA; off for scripted mock smoke tests")
    ap.add_argument("--video", default="", help="write an MP4 of the policy's camera frames")
    args = ap.parse_args()
    summary = run(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
