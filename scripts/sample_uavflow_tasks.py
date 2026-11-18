#!/usr/bin/env python3
"""Deterministically sample training task JSONs, excluding the test set."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    test_ids = {p.stem for p in Path(args.test_dir).glob("*.json")}
    train = sorted(p for p in Path(args.train_dir).glob("*.json") if p.stem not in test_ids)
    if len(train) < args.count:
        raise SystemExit(f"only {len(train)} train tasks, requested {args.count}")
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(train, args.count))
    manifest = {"seed": args.seed, "count": len(chosen), "test_count": len(test_ids),
                "train_pool": len(train), "tasks": [str(p) for p in chosen],
                "test_dir": str(Path(args.test_dir))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"selected": len(chosen), "manifest": args.out}))


if __name__ == "__main__":
    main()
