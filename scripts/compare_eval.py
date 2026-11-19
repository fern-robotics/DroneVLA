#!/usr/bin/env python3
"""Paired comparison of two UAV-Flow eval result directories."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def load(path: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in (path / "results").glob("*.json")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--grpo", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    a, b = load(Path(args.baseline)), load(Path(args.grpo))
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no common tasks")
    sa = np.array([bool(a[t]["success"]) for t in common])
    sb = np.array([bool(b[t]["success"]) for t in common])
    na = np.array([float(a[t]["ndtW"]) for t in common])
    nb = np.array([float(b[t]["ndtW"]) for t in common])
    rec = {
        "tasks": len(common),
        "baseline_success": float(sa.mean()),
        "grpo_success": float(sb.mean()),
        "success_delta_pp": float(100 * (sb.mean() - sa.mean())),
        "baseline_ndtW": float(na.mean()),
        "grpo_ndtW": float(nb.mean()),
        "ndtW_delta": float(nb.mean() - na.mean()),
        "grpo_wins": int(((~sa) & sb).sum()),
        "baseline_wins": int((sa & (~sb)).sum()),
        "ties": int((sa == sb).sum()),
    }
    print(json.dumps(rec, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
