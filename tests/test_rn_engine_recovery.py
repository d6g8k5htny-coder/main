"""The recovered frozen RN engine stays byte-exact, and the recovery verifier's
own negative controls keep firing.

``engine/rn_engine/verify_recovery.py --selftest`` was written by the agent that
recovered the engine, which did not own ``tests/`` and so could not add a pytest
wrapper. This is that wrapper — plus one assertion the self-test does not make
for us: the eight declared digests are re-derived here from the archive member
index and compared to disk directly, so a regression in the verifier itself
cannot hide a regression in the bytes.

What this does not establish: anything mathematical. A byte-exact frozen engine
is provenance. ``D3-LEMMA-RN-UNIF`` Pieces 1 and 2 remain OPEN; the engine is
``mpmath`` floating point end to end and certifies nothing; and nothing here
runs it.
"""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFIER = os.path.join(ROOT, "engine", "rn_engine", "verify_recovery.py")
FROZEN = os.path.join(ROOT, "engine", "rn_engine", "frozen")
MEMBERS = os.path.join(ROOT, "drive", "source_map", "Archive_Members.csv")

# The engine itself, pinned by name: the one file this lane exists to recover.
ENGINE_REL = os.path.join("K3_SIDE24_LB", "UPPER2D", "D3_percolation", "d3_rn_unif.py")
ENGINE_SHA = "85d7725fab42eeb0e823226f44d17f142a5c57e5d89084b2b6edffe4a8f0c930"
ENGINE_BYTES = 103166


def sha256_of(path: str) -> tuple[str, int]:
    with open(path, "rb") as f:
        b = f.read()
    return hashlib.sha256(b).hexdigest(), len(b)


def member_index() -> dict[str, set[tuple[str, int]]]:
    """basename -> {(payload sha256, bytes)} from the accessibility source map."""
    idx: dict[str, set[tuple[str, int]]] = {}
    with open(MEMBERS, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mp = row["Member path"].split("!/")[-1]
            idx.setdefault(os.path.basename(mp), set()).add(
                (row["Payload SHA-256"], int(row["Bytes"] or 0))
            )
    return idx


def test_selftest_passes():
    proc = subprocess.run(
        [sys.executable, VERIFIER, "--selftest"], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 problems" in proc.stdout
    # The verifier must keep saying what it is not.
    assert "establishes no mathematical claim" in proc.stdout


def test_the_engine_itself_is_byte_exact():
    digest, n = sha256_of(os.path.join(FROZEN, ENGINE_REL))
    assert n == ENGINE_BYTES
    assert digest == ENGINE_SHA


def test_every_frozen_file_matches_the_archive_member_index():
    """Independent of the verifier: re-derive expectations from the source map."""
    idx = member_index()
    checked = 0
    for dirpath, _, files in os.walk(FROZEN):
        for fn in files:
            if fn.endswith(".pyc") or "__pycache__" in dirpath:
                continue
            path = os.path.join(dirpath, fn)
            digest, n = sha256_of(path)
            assert fn in idx, f"{fn} is under frozen/ but appears nowhere in the member index"
            assert (digest, n) in idx[fn], (
                f"{fn}: on disk {digest[:16]}… {n} B, index has "
                f"{[(s[:16], b) for s, b in idx[fn]]}"
            )
            checked += 1
    assert checked >= 8


def test_a_single_flipped_byte_is_caught(tmp_path):
    """Negative control: the verifier must reject a frozen tree with one byte
    changed. Run it against a modified copy, never against the real tree."""
    import shutil

    copy_root = tmp_path / "repo"
    shutil.copytree(
        ROOT, copy_root,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "drive", "registers", "recovery", "reviews", "research"),
    )
    # The verifier reads the source map, so bring the one CSV it needs.
    os.makedirs(copy_root / "drive" / "source_map", exist_ok=True)
    shutil.copy(MEMBERS, copy_root / "drive" / "source_map" / "Archive_Members.csv")

    target = copy_root / "engine" / "rn_engine" / "frozen" / ENGINE_REL
    b = bytearray(target.read_bytes())
    b[0] ^= 0x01
    target.write_bytes(bytes(b))

    proc = subprocess.run(
        [sys.executable, str(copy_root / "engine" / "rn_engine" / "verify_recovery.py"), "--selftest"],
        capture_output=True, text=True, cwd=str(copy_root),
    )
    assert proc.returncode != 0
