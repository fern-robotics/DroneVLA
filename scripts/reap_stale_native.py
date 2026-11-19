#!/usr/bin/env python3
"""Kill native UAV-Flow runners/engines older than a timeout."""
from __future__ import annotations
import argparse, os, signal, time
from pathlib import Path


def age_seconds(pid: int) -> float:
    stat = Path(f"/proc/{pid}/stat").read_text().split()
    start_ticks = float(stat[21])
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    return uptime - start_ticks / os.sysconf("SC_CLK_TCK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age", type=float, default=900)
    ap.add_argument("--match", default="uavflow_native_policy.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    killed = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError):
            continue
        if args.match not in cmd and "Binaries/Linux/Collection" not in cmd:
            continue
        if "eval_uavflow_test" in cmd or "serve_openvla" in cmd:
            continue
        try:
            age = age_seconds(pid)
        except Exception:
            continue
        if age > args.max_age:
            killed.append((pid, round(age), cmd[:120]))
            if not args.dry_run:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    print({"killed": killed, "count": len(killed)}, flush=True)


if __name__ == "__main__":
    main()
