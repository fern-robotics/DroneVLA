#!/usr/bin/env python3
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("inputs", nargs="+")
args = ap.parse_args()
merged, seen = [], set()
for inp in args.inputs:
    p = Path(inp)
    if not p.exists():
        continue
    for g in json.loads(p.read_text()):
        if g["task"] in seen:
            continue
        seen.add(g["task"]); g = dict(g); g["group"] = len(merged); merged.append(g)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(merged, indent=2))
print(json.dumps({"groups": len(merged), "out": args.out}))
