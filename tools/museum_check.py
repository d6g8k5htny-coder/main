"""Read pinned museum bytes once, then check export and actual UI integration.

Only local, reviewed display code is executed. Retrieved proof/review/notebook
bytes are test inputs, never imported or executed.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile

import museum_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-root", type=Path)
    args = parser.parse_args()
    sources = list(museum_data.SOURCES.values())
    with ThreadPoolExecutor(max_workers=8) as pool:
        inputs = list(pool.map(
            lambda source: museum_data.read_source(source, ROOT, args.math_root), sources))
    with tempfile.TemporaryDirectory(prefix="museum-check-") as directory:
        fixture = Path(directory)
        mapping = {}
        for source, raw in zip(sources, inputs):
            repository = source["repository"].split("/")[-1]
            target = fixture / repository / source["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            mapping[museum_data.descriptor(source["path"])["url"]] = base64.b64encode(raw).decode("ascii")
        expected = museum_data.build(fixture / "main", fixture / "Math-")
        museum_data.check_export(ROOT / "docs/site/museum.json", expected)
        manifest = fixture / "verified-inputs.json"
        manifest.write_text(json.dumps(mapping), encoding="utf-8")
        env = dict(os.environ, MUSEUM_FIXTURE=str(manifest))
        subprocess.run(["node", "--test", "tests/test_museum_frontend.mjs"],
                       cwd=ROOT, env=env, check=True)
    print("Museum source export and actual display integration verified. Scientific effect NONE.")


if __name__ == "__main__":
    main()
