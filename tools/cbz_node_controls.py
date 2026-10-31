#!/usr/bin/env python3
"""Run the bounded Node controls with source-selected child test imports."""
import argparse
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

_BOOTSTRAP = (
    "import sys, unittest\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "unittest.main(module=None, argv=['unittest', 'discover', '-s', 'tests', "
    "'-p', 'test_cbz_node_regression.py', '-v'])\n"
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not (root / "tests").is_dir():
        parser.error("root must contain the test directory")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "GIT_"))
           and key not in ("CBZ_NODE_RUNNER_SOURCE", "CBZ_NODE_MUTANT_NONCE",
                           "CBZ_NODE_MUTANT_RECEIPT", "CBZ_NODE_MUTANT_NAME",
                           "CBZ_NODE_CONTROL_LAUNCHER")}
    with tempfile.TemporaryDirectory(prefix="cbz-node-outer-pycache-") as cache:
        command = [sys.executable, *(["-O"] if sys.flags.optimize else []),
                   "-I", "-B", "-S", "-X", f"pycache_prefix={cache}",
                   "-c", _BOOTSTRAP, str(root)]
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    runs = re.findall(r"^Ran [^\n]*$", result.stderr, re.MULTILINE)
    clean_summary = re.findall(r"^OK$", result.stderr, re.MULTILINE)
    if (result.returncode != 0 or len(runs) != 1 or len(clean_summary) != 1
            or not re.fullmatch(r"Ran [1-9][0-9]* tests? in [0-9]+(?:\.[0-9]+)?s", runs[0])):
        print("Node control suite did not execute a nonempty unskipped passing suite", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
