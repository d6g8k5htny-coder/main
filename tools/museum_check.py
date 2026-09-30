"""Read pinned museum bytes once, then check export and actual UI integration.

Only local, reviewed display code is executed. Retrieved proof/review/notebook
bytes are test inputs, never imported or executed.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import museum_data

ROOT = Path(__file__).resolve().parents[1]


def check_local_identity():
    config = json.loads((ROOT / "docs/site/config.json").read_bytes())
    raw = (ROOT / "docs/site/museum.json").read_bytes()
    descriptor = config["museum_json"]
    if (descriptor != {"url": "museum.json", "bytes": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest()}):
        raise ValueError("Museum manifest descriptor differs from local bytes")
    manifest = json.loads(raw)
    readme = (ROOT / "docs/site/README.md").read_text(encoding="utf-8")
    math_pin = manifest["index_source"]["commit"]
    query_pin = config["query"]["commit"]
    if (config["query"]["math_pin"] != math_pin or
            f"git -C Math- checkout --detach {math_pin}" not in readme or
            f"git -C query- checkout --detach {query_pin}" not in readme):
        raise ValueError("Site reproduction guide Math/query checkouts differ from museum/config pins")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-root", type=Path)
    args = parser.parse_args()
    check_local_identity()
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
