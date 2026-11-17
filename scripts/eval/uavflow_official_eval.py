#!/usr/bin/env python3
"""Official UAV-Flow control loop with direct UnrealCV setup.

The upstream batch script hangs in gym_unrealcv's reset on the released
UnrealCV server. This keeps its control_loop/action composition and replaces
only environment initialization.
"""
from __future__ import annotations
import argparse, json, sys, time
import gym
import gym_unrealcv

sys.path.insert(0, "/workspace/UAV-Flow/UAV-Flow-Eval")
from batch_run_act_all import control_loop, create_obj_if_needed, set_cam  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--server-port", type=int, default=5007)
    args = ap.parse_args()
    task = json.load(open(args.task))
    env = gym.make("UnrealTrack-DowntownWest-ContinuousColor-v0")
    u = env.unwrapped
    u.agents_category = ["drone"]
    if not u.launched:
        u.launched = u.launch_ue_env()
    player = next(p for p in u.player_list if "Drone" in p)
    cam = u.cam_list[u.player_list.index(player)]
    u.player_list = [player]
    u.cam_list = [cam]
    u.unrealcv.set_viewport(player)
    u.unrealcv.set_phy(player, 0)
    initial = [float(v) for v in task["initial_pos"]]
    u.unrealcv.set_obj_location(player, initial[:3])
    u.unrealcv.set_rotation(player, initial[4] - 180)
    set_cam(env)
    obj_info = None
    if "obj_id" in task and "use_obj" in task and task.get("target_pos"):
        obj_info = {"use_obj": task["use_obj"], "obj_id": task["obj_id"],
                    "obj_pos": task["target_pos"][:3], "obj_rot": [0, 0, 0]}
        create_obj_if_needed(env, obj_info)
        set_cam(env)
    time.sleep(2.0)
    trajectory = []
    control_loop(initial, env=env, instruction=task["instruction"], max_steps=args.max_steps,
                 trajectory_log=trajectory, server_url=f"http://127.0.0.1:{args.server_port}/predict")
    json.dump(trajectory, open(args.out + "/official_episode.json", "w"), indent=2)
    print(json.dumps({"steps": len(trajectory), "final": trajectory[-1] if trajectory else None}))


if __name__ == "__main__":
    main()
