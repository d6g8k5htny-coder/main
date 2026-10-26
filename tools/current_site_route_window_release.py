#!/usr/bin/env python3
"""Rewrite exactly the atlas asset pin in the generated dependency page."""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

PIN = re.compile(rb"atlas\.mjs\?site-release=[0-9a-f]{64}")


def render(source: bytes, atlas: bytes) -> bytes:
    digest = hashlib.sha256(atlas).hexdigest().encode("ascii")
    rendered, count = PIN.subn(b"atlas.mjs?site-release=" + digest, source)
    if count != 1:
        raise ValueError(f"expected exactly one atlas release pin; found {count}")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = render(args.input.read_bytes(), args.atlas.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
