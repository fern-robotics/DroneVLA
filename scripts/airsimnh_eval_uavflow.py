#!/usr/bin/env python3
"""Evaluate a UAV-Flow-Sim-trained DroneVL linear checkpoint in actual AirSimNH.

The Rust checkpoint uses the same 48-feature goal-conditioned policy that was
trained on UAV-Flow-Sim. This script reproduces the feature/planner contract in
Python, flies the intent in AirSimNH over RPC, and scores intent success from the
actual trajectory. It is an evaluation bridge, not additional training.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/workspace/thesis")
sys.path.insert(0, str(ROOT / "src"))
from dronevl.episodes import Episode
from dronevl.simulators.airsim import AirSimSimulator

DT = 0.1
MAX_T = 25.0
START = np.array([0.0, 0.0, -10.0])
INTENT_INDEX = {"approach": 4, "pass": 5, "orbit": 6, "hold_at": 7}

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def rot_inv(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])

def rot(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])

def fnv(s):
    h = 0xcbf29ce484222325
    for b in s.encode():
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h

def text_hash(text):
    counts = np.zeros(16)
    tokens = [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]
    for t in tokens:
        counts[fnv(t) % 16] += 1.0
    if tokens:
        counts /= len(tokens)
    return counts

class Policy:
    def __init__(self, path):
        d = json.load(open(path))
        self.w = np.array(d["w"], dtype=float)
        self.b = np.array(d["b"], dtype=float)
        self.scales = np.array(d.get("scales", [0.35, 0.22, 0.18, 0.12]))
    def features(self, pos, yaw, vel, goal, goal_yaw, intent, progress, t, prev):
        e = goal - pos
        body = rot_inv(e, yaw)
        f = np.zeros(48)
        f[0:3] = body / 5.0
        f[3] = np.linalg.norm(e) / 10.0
        f[4] = wrap(goal_yaw - yaw) / math.pi
        f[5] = math.sin(yaw); f[6] = math.cos(yaw)
        f[7] = pos[2] / 10.0
        f[8] = np.linalg.norm(vel) / 3.0
        idx = INTENT_INDEX[intent["kind"]]
        f[idx] = 1.0
        f[18] = intent.get("magnitude", 3.0) / 5.0
        f[19] = intent.get("side_sign", 0.0)
        f[20] = progress
        f[21] = 0.0
        f[22] = min(1.0, t / MAX_T)
        # f[23:27] landmark perception features stay zero, matching SFT.
        f[27:43] = text_hash(intent["instruction"])
        f[43:47] = prev / self.scales
        f[47] = 1.0
        return f
    def act(self, f):
        z = self.w @ f + self.b
        return self.scales * np.tanh(z)

def goal_for(intent, pos, yaw, state):
    L = intent["landmark"]
    kind = intent["kind"]
    if kind in ("approach", "hold_at"):
        d = L - np.array([START[0], START[1], L[2]])
        n = np.linalg.norm(d[:2]) or 1.0
        d = d / n
        goal = L - d * intent.get("standoff", 3.0)
        return goal, math.atan2(L[1] - pos[1], L[0] - pos[0])
    if kind == "pass":
        travel = L - np.array([START[0], START[1], L[2]])
        n = np.linalg.norm(travel[:2]) or 1.0
        travel = travel / n
        right = np.array([-travel[1], travel[0], 0.0])
        side = intent["side_sign"]
        pre = L - travel * 2.0 + right * side * 1.5
        post = L + travel * 2.0 + right * side * 1.5
        plane = float((pos - L) @ travel)
        goal = pre if plane < -1.5 else post
        return goal, math.atan2(travel[1], travel[0])
    if kind == "orbit":
        R = intent.get("radius", 3.0)
        bearing = math.atan2(pos[1] - L[1], pos[0] - L[0])
        radius = np.linalg.norm((pos - L)[:2])
        step = 0.22 * intent["orbit_sign"]
        tb = bearing if radius > R + 1.0 else bearing + step
        goal = np.array([L[0] + R * math.cos(tb), L[1] + R * math.sin(tb), L[2]])
        tangent = tb + intent["orbit_sign"] * math.pi / 2
        return goal, tangent
    raise ValueError(kind)

def progress_and_done(intent, pos, yaw, state, accum):
    L = intent["landmark"]; kind = intent["kind"]
    if kind == "approach":
        d = np.linalg.norm((pos - goal_for(intent, pos, yaw, state)[0])[:2])
        p = max(0.0, min(1.0, 1.0 - d / 6.0))
        return p, d <= 0.9, accum
    if kind == "hold_at":
        d = np.linalg.norm((pos - goal_for(intent, pos, yaw, state)[0])[:2])
        accum["hold"] = accum.get("hold", 0.0) + DT if d <= 1.0 else 0.0
        p = min(1.0, accum["hold"] / intent.get("duration", 2.0))
        return p, accum["hold"] >= intent.get("duration", 2.0), accum
    if kind == "pass":
        travel = L - np.array([START[0], START[1], L[2]])
        travel = travel / (np.linalg.norm(travel[:2]) or 1.0)
        right = np.array([-travel[1], travel[0], 0.0])
        plane = float((pos - L) @ travel)
        lateral = float((pos - L) @ right)
        crossed = plane > 0.0 and lateral * intent["side_sign"] > 0.8
        p = 0.5 + 0.5 * min(1.0, max(0.0, plane / 4.0)) if plane > 0 else max(0.0, 0.5 + plane / 8.0)
        return min(1.0, p), crossed, accum
    if kind == "orbit":
        bearing = math.atan2(pos[1] - L[1], pos[0] - L[0])
        radius = np.linalg.norm((pos - L)[:2])
        prev = accum.get("bearing")
        if prev is not None and 1.0 <= radius <= intent.get("radius", 3.0) + 2.5:
            accum["angle"] = accum.get("angle", 0.0) + wrap(bearing - prev) * intent["orbit_sign"]
        accum["bearing"] = bearing
        target = math.radians(intent.get("degrees", 180.0))
        p = min(1.0, accum.get("angle", 0.0) / target)
        return p, accum.get("angle", 0.0) >= target, accum
    raise ValueError(kind)

def run_episode(policy, intent, seed):
    ep = Episode.load(ROOT / "environments/airsimnh/episodes/square.json")
    sim = AirSimSimulator(host="127.0.0.1", port=41451, scene="AirSimNH", server_family="microsoft-airsim")
    sim.prepare(ep)
    accum = {}
    prev = np.zeros(4)
    t = 0.0
    collisions = set()
    traj = []
    success = False
    progress = 0.0
    try:
        while t < MAX_T:
            snap = sim.observe()
            st = snap.state
            pos = np.array(st.position_ned_m, dtype=float)
            yaw = st.yaw_rad
            vel = np.array(st.velocity_ned_mps, dtype=float)
            goal, goal_yaw = goal_for(intent, pos, yaw, st)
            progress, done, accum = progress_and_done(intent, pos, yaw, st, accum)
            if done:
                success = True
                break
            if snap.collision_id:
                collisions.add(snap.collision_id)
            f = policy.features(pos, yaw, vel, goal, goal_yaw, intent, progress, t, prev)
            delta = policy.act(f)
            delta = np.clip(delta, -1.2 * policy.scales, 1.2 * policy.scales)
            v_body = delta[:3] / DT
            v_ned = rot(v_body, yaw)
            yaw_rate = delta[3] / DT
            sim.advance(tuple(v_ned), yaw_rate, DT)
            prev = delta
            t += DT
            traj.append((round(t, 2), *[round(v, 3) for v in pos], round(yaw, 3)))
    finally:
        try:
            sim.client.reset()
        except Exception:
            pass
        try:
            sim.close()
        except Exception:
            pass
    return {
        "intent": intent["kind"], "landmark": intent["landmark"].tolist(),
        "seed": seed, "success": success, "progress": round(progress, 3),
        "collisions": len(collisions), "steps": len(traj),
        "final": [round(v, 2) for v in traj[-1][1:4]] if traj else None,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", default="/workspace/dronevl/m1_uavflow.json")
    ap.add_argument("--out", default="/workspace/thesis/runs/airsimnh-uavflow-eval")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    intents = [
        {"kind": "approach", "landmark": np.array([5.0, 0.0, -10.0]), "standoff": 3.0,
         "instruction": "Approach the red circle to within 3.0 metres.", "magnitude": 3.0},
        {"kind": "pass", "landmark": np.array([5.0, 0.0, -10.0]), "side_sign": -1.0,
         "instruction": "Pass the red circle on the left.", "magnitude": 1.5},
        {"kind": "orbit", "landmark": np.array([5.0, 5.0, -10.0]), "orbit_sign": 1.0,
         "radius": 3.0, "degrees": 180.0,
         "instruction": "Orbit the green cross clockwise for 180 degrees at 3.0 metres.", "magnitude": 180.0},
        {"kind": "hold_at", "landmark": np.array([0.0, 5.0, -10.0]), "standoff": 3.0,
         "duration": 2.0, "instruction": "Hold 3.0 metres from the yellow triangle for 2.0 seconds.", "magnitude": 3.0},
    ]
    # M0 = same architecture with only the small forward bias.
    m0 = Policy(args.m1)
    m0.w[:] = 0.0
    m0.b[:] = 0.0
    m0.b[0] = 0.01
    policies = {"M0": m0, "M1": Policy(args.m1)}
    rows = []
    for name in ("M0", "M1"):
        for i, intent in enumerate(intents):
            r = run_episode(policies[name], intent, i)
            r["condition"] = name
            rows.append(r)
            print(json.dumps(r), flush=True)
            with open(Path(args.out) / "results.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
    with open(Path(args.out) / "results.json", "w") as f:
        json.dump(rows, f, indent=2)
    for name in ("M0", "M1"):
        rs = [r for r in rows if r["condition"] == name]
        print(json.dumps({"condition": name, "success_rate": sum(r["success"] for r in rs) / len(rs),
                          "mean_progress": sum(r["progress"] for r in rs) / len(rs),
                          "collisions": sum(r["collisions"] for r in rs)}), flush=True)

if __name__ == "__main__":
    main()
