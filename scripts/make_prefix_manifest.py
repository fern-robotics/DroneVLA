#!/usr/bin/env python3
"""Create a manifest of the first N sorted test tasks (legacy M0 subset)."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="/workspace/UAV-Flow/UAV-Flow-Eval/test_jsons")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=244)
    args = ap.parse_args()
    tasks = sorted(Path(args.test_dir).glob("*.json"))[:args.count]
    Path(args.out).write_text(json.dumps({"count": len(tasks), "tasks": [str(p) for p in tasks]}, indent=2))
    print(json.dumps({"count": len(tasks), "out": args.out}))


if __name__ == "__main__":
    main()
