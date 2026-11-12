#!/usr/bin/env python3
"""Patch official OpenVLA LoRA config to the thesis recipe.

The official script uses target_modules="all-linear" and alpha=min(rank,16).
The thesis uses q_proj/v_proj and alpha=64 for rank 32.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/openvla")
    path = root / "vla-scripts/finetune.py"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    s = path.read_text()
    s = s.replace('target_modules="all-linear"', 'target_modules=["q_proj", "v_proj"]')
    s = s.replace("lora_alpha=min(cfg.lora_rank, 16)", "lora_alpha=64")
    path.write_text(s)
    print("patched", path)


if __name__ == "__main__":
    main()
