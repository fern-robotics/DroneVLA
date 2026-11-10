#!/usr/bin/env python3
"""π0-FAST policy-server intent mission in AirSimNH."""
import argparse, dataclasses, json, math, sys, tempfile, time
from pathlib import Path
import numpy as np
from PIL import Image
from openpi_client import websocket_client_policy

ROOT = Path("/workspace/thesis")
sys.path.insert(0, str(ROOT / "src"))
from dronevl.episodes import Episode
from dronevl.simulators.airsim import AirSimSimulator

MARKER = (10.0, 0.0, -10.0)
ARENA = {
    "schema_version": "1.0", "family": "visual_grounding", "variant": "pi0-intent",
    "reference_frame": "world_NED", "initial_target_hidden": False,
    "markers": [{
        "marker_id": "C", "asset": "Cube", "position_ned_m": list(MARKER),
        "scale": [0.2, 1.5, 1.5],
        "texture": "environments/airsimnh/markers/red-circle.png",
        "texture_sha256": "4245b9c4b22ddb75c0b652660ba4bcdd7278b3db147172a2492d9178b1cf93be",
        "inspection_position_ned_m": [7.0, 0.0, -10.0], "inspection_yaw_rad": 0.0,
    }],
}

def rot(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c*v[0]-s*v[1], s*v[0]+c*v[1], v[2]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--instruction", default="Move to the red circle on the front side")
    ap.add_argument("--replans", type=int, default=20)
    ap.add_argument("--replan-steps", type=int, default=4)
    ap.add_argument("--exec-duration", type=float, default=0.5)
    ap.add_argument("--stop-distance", type=float, default=3.5)
    ap.add_argument("--out", default="/workspace/thesis/runs/pi0-intent.json")
    args = ap.parse_args()
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    ep = dataclasses.replace(Episode.load(ROOT / "environments/airsimnh/episodes/square.json"),
                             scene="AirSimNH", marker_arena=ARENA, instruction=args.instruction)
    sim = AirSimSimulator(scene="AirSimNH", server_family="microsoft-airsim")
    sim.prepare(ep)
    snap0 = sim.observe()
    start = np.array(snap0.state.position_ned_m, dtype=float)
    yaw0 = snap0.state.yaw_rad
    rows = []
    min_dist = 1e9
    try:
        for replan in range(args.replans):
            sim.pause()
            tmp = Path(tempfile.mkdtemp()) / "frame"
            snap = sim.observe(sensor_dir=tmp)
            pos = np.array(snap.state.position_ned_m, dtype=float)
            yaw = snap.state.yaw_rad
            dist = float(np.linalg.norm((pos - np.array(MARKER))[:2]))
            min_dist = min(min_dist, dist)
            rel = np.array([(pos[0]-start[0]), (pos[1]-start[1]), (pos[2]-start[2]),
                            (yaw-yaw0+math.pi)%(2*math.pi)-math.pi], dtype=np.float32)
            image = np.asarray(Image.open(tmp / "rgb.png").convert("RGB"))
            t0 = time.time()
            result = policy.infer({"observation/image": image, "observation/state": rel, "prompt": args.instruction})
            dt_inf = time.time() - t0
            sim.resume()
            actions = np.asarray(result["actions"], dtype=float)
            print(json.dumps({"replan": replan, "dist": round(dist,2), "infer_s": round(dt_inf,3),
                              "chunk": [[round(float(x),2) for x in a] for a in actions[:args.replan_steps]]}), flush=True)
            for a in actions[:args.replan_steps]:
                if dist <= args.stop_distance:
                    break
                v_body = a[:3] / args.exec_duration
                v_ned = rot(v_body, yaw)
                yaw_rate = a[3] / args.exec_duration
                sim.advance(tuple(v_ned), float(yaw_rate), args.exec_duration)
                snap2 = sim.observe()
                pos = np.array(snap2.state.position_ned_m, dtype=float)
                yaw = snap2.state.yaw_rad
                dist = float(np.linalg.norm((pos - np.array(MARKER))[:2]))
                min_dist = min(min_dist, dist)
                rows.append({"replan": replan, "dist": dist})
                if dist <= args.stop_distance:
                    break
            if dist <= args.stop_distance:
                break
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"instruction": args.instruction, "marker": MARKER,
                                              "min_dist": min_dist, "success": min_dist <= args.stop_distance,
                                              "rows": rows}, indent=2))
        print(json.dumps({"min_dist": min_dist, "success": min_dist <= args.stop_distance, "rows": len(rows)}), flush=True)
    finally:
        try: sim.client.reset()
        except Exception: pass
        try: sim.close()
        except Exception: pass

if __name__ == "__main__":
    main()
