"""Q4 limits lens -- FINDING probe (minor): museum_data.read_source's local paths
(tools/museum_data.py:466-475) have no size cap and no timeout, unlike the network path's
`response.read(identity["bytes"] + 1)`:

    result = subprocess.run(["git", "show", commit + ":" + path], cwd=root, capture_output=True)
    ...
    raw = (Path(root) / identity["path"]).read_bytes()      # when git show fails

The pinned byte count is known before the read, but the whole object/file is read first and only
then compared by verify_bytes. A FIFO at a pinned path blocks forever; a symlink to /dev/zero reads
until memory is exhausted. Reached with --math-root pointing at a plain fixture folder or at a
checkout that lacks the pinned commit (git show fails -> working-tree fallback).

No network. Each case runs in a child process with RLIMIT_AS = 1 GiB and a wall-clock limit so the
probe itself cannot hang or exhaust the host.
Usage: python3 -I -B p5_local_fallback_uncapped.py <path-to-cd32303-tree>
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

tree = Path(sys.argv[1]).resolve()
PIN = "frontiers/full_price_20260924/PROOF.md"   # a Math- source, read from math_root

CHILD = textwrap.dedent("""
    import resource, sys, time
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    sys.path.insert(0, sys.argv[1])
    import museum_data
    identity = museum_data.SOURCES[sys.argv[3]]
    t = time.monotonic()
    try:
        museum_data.read_source(identity, None, sys.argv[2])
        print("returned (unexpected)")
    except BaseException as exc:
        print(f"child: {type(exc).__name__}: {str(exc)[:80]!r} after {time.monotonic() - t:.1f}s; pinned bytes={identity['bytes']}")
""")


def run(label, math_root, limit):
    t = time.monotonic()
    try:
        proc = subprocess.run([sys.executable, "-I", "-B", "-c", CHILD, str(tree / "tools"), str(math_root), PIN],
                              capture_output=True, text=True, timeout=limit,
                              env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(math_root.parent)))
        print(f"{label}: exit={proc.returncode} {proc.stdout.strip()} {proc.stderr.strip()[-160:]}")
    except subprocess.TimeoutExpired:
        print(f"{label}: STILL BLOCKED after {time.monotonic() - t:.1f}s wall clock (killed by the probe)")


with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
    for name, make in (("fifo", lambda p: os.mkfifo(p)), ("devzero", lambda p: os.symlink("/dev/zero", p)),
                       ("bigfile", lambda p: p.write_bytes(b"x" * (300 << 20)))):
        math_root = Path(tmp) / name
        target = math_root / PIN
        target.parent.mkdir(parents=True)
        make(target)
        run(f"{name:8s} at math_root/{PIN}", math_root, 10)
