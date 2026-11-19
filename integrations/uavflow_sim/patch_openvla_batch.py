#!/usr/bin/env python3
"""Remove OpenVLA-UAV's batch-size=1 guard from the cached remote module."""
from __future__ import annotations
import sys
from pathlib import Path

GUARD = '''        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")
'''


def main() -> None:
    roots = list(Path.home().glob('.cache/huggingface/modules/transformers_modules/*/modeling_prismatic.py'))
    if len(sys.argv) > 1:
        roots = [Path(sys.argv[1])]
    if not roots:
        raise SystemExit('modeling_prismatic.py not found')
    for p in roots:
        s = p.read_text()
        if GUARD not in s:
            print('already patched', p)
            continue
        p.write_text(s.replace(GUARD, ''))
        print('patched', p)


if __name__ == '__main__':
    main()
