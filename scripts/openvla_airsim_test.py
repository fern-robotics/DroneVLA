#!/usr/bin/env python3
"""Closed-loop OpenVLA-UAV inference in actual AirSimNH."""
import argparse, json, math, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path("/workspace/thesis")
sys.path.insert(0, str(ROOT / "src"))
from dronevl.episodes import Episode
from dronevl.simulators.airsim import AirSimSimulator

def rot(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c*v[0]-s*v[1], s*v[0]+c*v[1], v[2]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/openvla-uav")
    ap.add_argument("--instruction", default="Move 5.0 meters at 0 degrees to the forward")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default="/workspace/thesis/runs/openvla-airsim-test.json")
    ap.add_argument("--episode", default="environments/airsimnh/episodes/square.json")
    args = ap.parse_args()
    print("loading processor/model", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).cuda().eval()
    print("model ready", flush=True)
    ep = Episode.load(ROOT / args.episode)
    sim = AirSimSimulator(scene="AirSimNH", server_family="microsoft-airsim")
    sim.prepare(ep)
    snap0 = sim.observe()
    start = np.array(snap0.state.position_ned_m, dtype=float)
    yaw0 = snap0.state.yaw_rad
    rows = []
    try:
        for step in range(args.steps):
            sim.pause()
            tmp = Path(tempfile.mkdtemp()) / "frame"
            snap = sim.observe(sensor_dir=tmp)
            pos = np.array(snap.state.position_ned_m, dtype=float)
            yaw = snap.state.yaw_rad
            rel = np.array([(pos[0]-start[0])*100.0, (pos[1]-start[1])*100.0, (pos[2]-start[2])*100.0, math.degrees((yaw-yaw0+math.pi)%(2*math.pi)-math.pi)])
            proprio = ",".join(str(round(float(x), 1)) for x in rel)
            prompt = f"In: Current State: {proprio}, What action should the uav take to {args.instruction}?\nOut:"
            image = Image.open(tmp / "rgb.png").convert("RGB")
            inputs = proc(prompt, image).to("cuda", dtype=torch.bfloat16)
            t0 = time.time()
            with torch.inference_mode():
                pred = model.predict_action(
                    input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                    unnorm_key="sim", do_sample=False,
                )
            dt_inf = time.time() - t0
            sim.resume()
            delta = np.array(pred, dtype=float)
            v_body = delta[:3] / 100.0 / 0.1
            v_ned = rot(v_body, yaw)
            yaw_rate = delta[3] / 0.1
            rows.append({"step": step, "proprio": rel.tolist(), "pred_cm_rad": delta.tolist(),
                         "v_ned": v_ned.tolist(), "yaw_rate": float(yaw_rate), "infer_s": dt_inf})
            print(json.dumps(rows[-1]), flush=True)
            sim.advance(tuple(v_ned), float(yaw_rate), 0.1)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"instruction": args.instruction, "rows": rows}, indent=2))
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
