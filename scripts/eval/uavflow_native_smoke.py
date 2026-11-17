#!/usr/bin/env python3
"""Minimal native UAV-Flow-Sim smoke: launch UE, set pose, capture one RGB frame."""
from __future__ import annotations
import argparse, json
import gym
import gym_unrealcv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="UnrealTrack-DowntownWest-ContinuousColor-v0")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default="/workspace/runs/uavflow_native")
    args = ap.parse_args()
    task = json.load(open(args.task))
    env = gym.make(args.env_id)
    u = env.unwrapped
    u.agents_category = ["drone"]
    if not u.launched:
        u.launched = u.launch_ue_env()
    player = u.player_list[0]
    cam = 0
    p = task["initial_pos"]
    u.unrealcv.set_obj_location(player, p[:3])
    u.unrealcv.set_rotation(player, p[4] - 180)
    u.unrealcv.set_cam(cam, p[:3], [p[3], p[4], p[5]])
    img = u.unrealcv.get_image(cam, "lit")
    import cv2
    cv2.imwrite(args.out + "/native_smoke.png", img)
    print(json.dumps({"player": player, "cam": cam, "shape": list(img.shape), "pose": p}))


if __name__ == "__main__":
    main()
