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
    ap.add_argument("--image-cam", type=int, default=0)
    ap.add_argument("--action-scale", type=float, default=1.0, help="1.0 for native cm actions, 100.0 for metre-delta checkpoints")
    ap.add_argument("--action-mode", choices=["delta", "absolute"], default="delta")
    ap.add_argument("--state-format", choices=["4", "6"], default="6")
    args = ap.parse_args()
    task = json.load(open(args.task))
    p = task["initial_pos"]
    init = [float(v) for v in p]
    env = gym.make(args.env_id)
    u = env.unwrapped
    u.agents_category = ["drone"]
    if not u.launched:
        u.launched = u.launch_ue_env()
    player = next((p for p in u.player_list if "Drone" in p), u.player_list[0])
    cam = args.image_cam
    u.unrealcv.set_viewport(player)
    u.unrealcv.set_phy(player, 0)
    u.unrealcv.set_obj_location(player, init[:3])
    u.unrealcv.set_rotation(player, init[4] - 180)
    u.unrealcv.set_cam(cam, [init[0], init[1], init[2] + 60], [init[3], init[4], init[5]])
    target = task.get("target_pos")
    target = [float(v) for v in target[:3]] if target else None
    pos = [0.0, 0.0, 0.0]  # local to initial
    yaw = 0.0
    writer = None
    log = open(args.out + "/episode.jsonl", "w")
    for step in range(args.max_steps):
        img = u.unrealcv.get_image(cam, "lit")
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(args.out + "/native_2k.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
        if args.state_format == "4":
            state = [pos[0], pos[1], pos[2], math.degrees(yaw)]
        else:
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
        if args.action_mode == "absolute":
            pos = [float(a[0]), float(a[1]), float(a[2])]
            yaw = float(a[3])
        else:
            dx, dy, dz, dyaw = [float(v) for v in a[:4]]
            c, s = math.cos(yaw), math.sin(yaw)
            pos[0] += c * dx - s * dy
            pos[1] += s * dx + c * dy
            pos[2] += dz
            yaw = wrap(yaw + dyaw)
        # UAV-Flow/UnrealZoo world coordinates are centimetres; policy actions are metres.
        world = [init[0] + args.action_scale * (math.cos(init[4]) * pos[0] - math.sin(init[4]) * pos[1]),
                 init[1] + args.action_scale * (math.sin(init[4]) * pos[0] + math.cos(init[4]) * pos[1]),
                 init[2] + args.action_scale * pos[2]]
        u.unrealcv.set_obj_location(player, world)
        u.unrealcv.set_rotation(player, init[4] + math.degrees(yaw) - 180)
        u.unrealcv.set_cam(cam, [world[0], world[1], world[2] + 60], [0, init[4] + math.degrees(yaw), 0])
        rec = {"step": step, "local_xyz": pos, "yaw": yaw,
               "world_xyz": world, "action": a,
               "vla_actions": actions,
               "tokens": result.get("tokens"),
               "logprobs": result.get("logprobs"),
               "latency_ms": (time.time() - t0) * 1000}
        if target is not None:
            dx_t = 0.01 * (world[0] - target[0])
            dy_t = 0.01 * (world[1] - target[1])
            rec["target_distance_m"] = math.hypot(dx_t, dy_t)
            rec["target_range_m"] = math.sqrt(dx_t * dx_t + dy_t * dy_t + (0.01 * (world[2] - target[2])) ** 2)
        log.write(json.dumps(rec) + "\n")
        log.flush()
    log.close()
    if writer is not None:
        writer.release()
    print(json.dumps({"steps": args.max_steps, "final_local": pos, "yaw": yaw}))


if __name__ == "__main__":
    main()
