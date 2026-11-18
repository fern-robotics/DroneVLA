#!/usr/bin/env python3
"""Combine per-task collector group files into one GRPO group file."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    groups = []
    for task in manifest["tasks"]:
        tid = Path(task).stem
        p = Path(args.run) / tid / "groups.json"
        if not p.exists():
            continue
        for g in json.loads(p.read_text()):
            groups.append(g)
    Path(args.out).write_text(json.dumps(groups, indent=2))
    print(json.dumps({"groups": len(groups), "out": args.out}))


if __name__ == "__main__":
    main()
