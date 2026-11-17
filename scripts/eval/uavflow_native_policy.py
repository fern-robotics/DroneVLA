#!/usr/bin/env python3
"""Minimal native UAV-Flow-Sim closed-loop runner for a 4-D policy server."""
from __future__ import annotations
import argparse, base64, io, json, math, time, urllib.request
import cv2, gym, gym_unrealcv, numpy as np
from PIL import Image


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="UnrealTrack-DowntownWest-ContinuousColor-v0")
    ap.add_argument("--task", required=True)
    ap.add_argument("--proxy", default="http://127.0.0.1:5007/predict")
    ap.add_argument("--out", default="/workspace/runs/uavflow_native")
    ap.add_argument("--max-steps", type=int, default=100)
    args = ap.parse_args()
    task = json.load(open(args.task))
    p = task["initial_pos"]
    init = [float(v) for v in p]
    env = gym.make(args.env_id)
    u = env.unwrapped
    u.agents_category = ["drone"]
    if not u.launched:
        u.launched = u.launch_ue_env()
    player = u.player_list[0]
    u.unrealcv.set_obj_location(player, init[:3])
    u.unrealcv.set_rotation(player, init[4] - 180)
    u.unrealcv.set_cam(0, init[:3], [init[3], init[4], init[5]])
    pos = [0.0, 0.0, 0.0]  # local to initial
    yaw = 0.0
    writer = None
    log = open(args.out + "/episode.jsonl", "w")
    for step in range(args.max_steps):
        img = u.unrealcv.get_image(0, "lit")
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(args.out + "/native_2k.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
        state = [pos[0], pos[1], pos[2], 0.0, math.degrees(yaw), 0.0]
        payload = {"image": base64.b64encode(buf.getvalue()).decode(), "proprio": state,
                   "instr": task["instruction"], "progress": 0.0}
        t0 = time.time()
        req = urllib.request.Request(args.proxy, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        actions = result.get("action", [])
        a = actions[0] if actions else [0, 0, 0, 0]
        dx, dy, dz, dyaw = [float(v) for v in a[:4]]
        c, s = math.cos(yaw), math.sin(yaw)
        pos[0] += c * dx - s * dy
        pos[1] += s * dx + c * dy
        pos[2] += dz
        yaw = wrap(yaw + dyaw)
        world = [init[0] + math.cos(init[4]) * pos[0] - math.sin(init[4]) * pos[1],
                 init[1] + math.sin(init[4]) * pos[0] + math.cos(init[4]) * pos[1],
                 init[2] + pos[2]]
        u.unrealcv.set_obj_location(player, world)
        u.unrealcv.set_rotation(player, init[4] + math.degrees(yaw) - 180)
        u.unrealcv.set_cam(0, world, [0, init[4] + math.degrees(yaw), 0])
        log.write(json.dumps({"step": step, "local_xyz": pos, "yaw": yaw,
                              "world_xyz": world, "action": a,
                              "latency_ms": (time.time() - t0) * 1000}) + "\n")
        log.flush()
    log.close()
    if writer is not None:
        writer.release()
    print(json.dumps({"steps": args.max_steps, "final_local": pos, "yaw": yaw}))


if __name__ == "__main__":
    main()
