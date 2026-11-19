#!/usr/bin/env python3
"""Serialize UnrealCV port selection across parallel engine launches."""
from pathlib import Path
import sys

LAUNCHER = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/venv-uavflow/lib/python3.11/site-packages/unrealcv/launcher.py")


def main() -> None:
    s = LAUNCHER.read_text()
    if "unrealcv_port.lock" in s:
        print("already patched"); return
    s = s.replace("        port = self.read_port()",
                  "        import fcntl\n"
                  "        _lock = open('/tmp/unrealcv_port.lock', 'w')\n"
                  "        fcntl.flock(_lock, fcntl.LOCK_EX)\n"
                  "        port = self.read_port()", 1)
    s = s.replace("        time.sleep(sleep_time)\n        return env_ip, port",
                  "        time.sleep(sleep_time)\n"
                  "        fcntl.flock(_lock, fcntl.LOCK_UN)\n"
                  "        _lock.close()\n"
                  "        return env_ip, port", 1)
    LAUNCHER.write_text(s)
    print("patched", LAUNCHER)


if __name__ == "__main__":
    main()
