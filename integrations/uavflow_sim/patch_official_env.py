#!/usr/bin/env python3
"""Patch UAV-Flow's batch runner to bypass the broken gym_unrealcv reset on Linux.

Only environment setup is replaced; control_loop, action composition, prompts,
and task JSONs stay upstream.
"""
from __future__ import annotations
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/UAV-Flow/UAV-Flow-Eval/batch_run_act_all.py")
    s = path.read_text()
    if "DRONEVL_MINIMAL_ENV" in s:
        print("already patched")
        return
    start = s.index("    env = gym.make(args.env_id)")
    end = s.index("    json_folder = args.json_folder", start)
    block = '''    # DRONEVL_MINIMAL_ENV: released UnrealCV server hangs in gym_unrealcv reset.
    env = gym.make(args.env_id)
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
    set_cam(env)

'''
    s = s[:start] + block + s[end:]
    path.write_text(s)
    print("patched", path)


if __name__ == "__main__":
    main()
