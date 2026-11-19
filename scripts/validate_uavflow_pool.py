#!/usr/bin/env python3
"""Validate the UAV-Flow train/test split and reference paths."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="/workspace/uavflow_train_tasks")
    ap.add_argument("--test-dir", default="/workspace/UAV-Flow/UAV-Flow-Eval/test_jsons")
    args = ap.parse_args()
    train = {p.stem: p for p in Path(args.train_dir).glob("*.json")}
    test = {p.stem: p for p in Path(args.test_dir).glob("*.json")}
    overlap = set(train) & set(test)
    bad = []
    for p in list(train.values()) + list(test.values()):
        t = json.loads(p.read_text())
        if not t.get("reference_path_raw") or len(t["reference_path_raw"]) < 2:
            bad.append(str(p))
    print(json.dumps({"train": len(train), "test": len(test), "overlap": len(overlap),
                      "bad_reference": len(bad)}))
    if overlap or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
