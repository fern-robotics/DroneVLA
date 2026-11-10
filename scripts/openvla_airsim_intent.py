#!/usr/bin/env python3
"""OpenVLA-UAV intent mission in AirSimNH: approach a provisioned red marker."""
import argparse, dataclasses, json, math, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path("/workspace/thesis")
sys.path.insert(0, str(ROOT / "src"))
from dronevl.episodes import Episode
from dronevl.simulators.airsim import AirSimSimulator

MARKER = (10.0, 0.0, -10.0)
ARENA = {
    "schema_version": "1.0",
    "family": "visual_grounding",
    "variant": "openvla-intent",
    "reference_frame": "world_NED",
    "initial_target_hidden": False,
    "markers": [{
        "marker_id": "C", "asset": "Cube",
        "position_ned_m": list(MARKER), "scale": [0.2, 1.5, 1.5],
        "texture": "environments/airsimnh/markers/red-circle.png",
        "texture_sha256": "4245b9c4b22ddb75c0b652660ba4bcdd7278b3db147172a2492d9178b1cf93be",
        "inspection_position_ned_m": [7.0, 0.0, -10.0],
        "inspection_yaw_rad": 0.0,
    }],
}

def rot(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c*v[0]-s*v[1], s*v[0]+c*v[1], v[2]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/openvla-uav")
    ap.add_argument("--instruction", default="Move to the red circle on the front side")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--exec-duration", type=float, default=0.5)
    ap.add_argument("--stop-distance", type=float, default=3.2)
    ap.add_argument("--out", default="/workspace/thesis/runs/openvla-intent.json")
    args = ap.parse_args()
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).cuda().eval()
    base = Episode.load(ROOT / "environments/airsimnh/episodes/square.json")
    ep = dataclasses.replace(base, scene="AirSimNH", marker_arena=ARENA, instruction=args.instruction)
    sim = AirSimSimulator(scene="AirSimNH", server_family="microsoft-airsim")
    sim.prepare(ep)
    snap0 = sim.observe()
    start = np.array(snap0.state.position_ned_m, dtype=float)
    yaw0 = snap0.state.yaw_rad
    rows = []
    min_dist = 1e9
    try:
        for step in range(args.steps):
            sim.pause()
            tmp = Path(tempfile.mkdtemp()) / "frame"
            snap = sim.observe(sensor_dir=tmp)
            pos = np.array(snap.state.position_ned_m, dtype=float)
            yaw = snap.state.yaw_rad
            dist = float(np.linalg.norm((pos - np.array(MARKER))[:2]))
            min_dist = min(min_dist, dist)
            rel = np.array([(pos[0]-start[0])*100.0, (pos[1]-start[1])*100.0, (pos[2]-start[2])*100.0,
                            math.degrees((yaw-yaw0+math.pi)%(2*math.pi)-math.pi)])
            proprio = ",".join(str(round(float(x), 1)) for x in rel)
            prompt = f"In: Current State: {proprio}, What action should the uav take to {args.instruction}?\nOut:"
            image = Image.open(tmp / "rgb.png").convert("RGB")
            inputs = proc(prompt, image).to("cuda", dtype=torch.bfloat16)
            t0 = time.time()
            with torch.inference_mode():
                pred = model.predict_action(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                            unnorm_key="sim", do_sample=False)
            dt_inf = time.time() - t0
            sim.resume()
            delta = np.array(pred, dtype=float)
            v_body = delta[:3] / 100.0 / args.exec_duration
            v_ned = rot(v_body, yaw)
            yaw_rate = delta[3] / args.exec_duration
            row = {"step": step, "dist": round(dist, 2), "pred": [round(float(x), 2) for x in delta],
                   "pos": [round(float(x), 2) for x in pos], "infer_s": round(dt_inf, 3),
                   "collision": bool(snap.collision_id)}
            rows.append(row)
            if step % 5 == 0 or dist <= args.stop_distance:
                print(json.dumps(row), flush=True)
            if dist <= args.stop_distance:
                break
            # Collision IDs are latched at takeoff; log but do not abort here.
            sim.advance(tuple(v_ned), float(yaw_rate), args.exec_duration)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"instruction": args.instruction, "marker": MARKER,
                                              "min_dist": min_dist, "success": min_dist <= args.stop_distance,
                                              "rows": rows}, indent=2))
        print(json.dumps({"min_dist": min_dist, "success": min_dist <= args.stop_distance, "steps": len(rows)}), flush=True)
    finally:
        try:
            sim.client.reset()
        except Exception:
            pass
        try:
            sim.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
