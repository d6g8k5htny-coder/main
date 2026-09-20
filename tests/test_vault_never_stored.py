"""Negative controls for the do-not-open vault guard in tools/quarantine_check.py.

CLAUDE.md rule 9: `99_DO_NOT_OPEN` is never opened for authority, proofs,
certificates or "latest" status unless an operator names a vault id for forensic
recovery.  Metadata only.

Until this guard existed the rule was kept by care alone.  Nothing in the tree
refused a manifest row that stored vault bytes, so a later port sweeping "every
remaining native Doc in the inventory" would have taken the vault with it and
passed every checker here.  The lane's five native Docs are the whole of the
repository's remaining native-Doc gap, which is exactly the shape of sweep that
would have done it.

Every control runs the checker through its CLI against a synthetic scan root, so
a tree path bound at import time cannot silently re-check the real repository.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "quarantine_check.py")

VAULT = ("01_ACTIVE_RESEARCH_PACKAGES/99_DO_NOT_OPEN — "
         "SUPERSEDED_MIRRORS_DEAD_ENDS_AND_TRAP_COPIES/A trap copy")


def run(scan_root):
    """Run the checker with the real registers but a synthetic tree to scan."""
    return subprocess.run(
        [sys.executable, CHECKER, "--scan-root", scan_root],
        capture_output=True, text=True, cwd=ROOT)


def write_manifest(tmp_path, rows):
    d = tmp_path / "drive" / "mirrors" / "LANE"
    d.mkdir(parents=True)
    with open(d / "_MANIFEST.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(tmp_path)


BYTES_ROW = {
    "id": "1VaultDriveIdAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "title": "A trap copy",
    "drive_path": VAULT,
    "dest": "trap.txt",
    "bytes": 11,
    "sha256": "0" * 64,
    "exact": True,
    "stored": True,
}


def test_a_byte_exact_vault_row_is_refused(tmp_path):
    out = run(write_manifest(tmp_path, [BYTES_ROW]))
    assert out.returncode != 0, out.stdout
    assert "is metadata only and is never opened" in out.stdout
    assert "vault_rows_storing_bytes=1" in out.stdout


def test_a_vault_reading_copy_is_refused(tmp_path):
    """The sweep that would actually have done it: native Docs, exact false."""
    row = dict(BYTES_ROW, exact=False, inventory_sha256=None,
               dest="trap.export.txt")
    out = run(write_manifest(tmp_path, [row]))
    assert out.returncode != 0, out.stdout
    assert "vault_rows_storing_bytes=1" in out.stdout


def test_an_index_row_naming_the_vault_is_allowed(tmp_path):
    """Metadata only is the rule, and an index row is metadata."""
    row = dict(BYTES_ROW, stored=False, dest=None, bytes=None, sha256=None,
               not_stored_reason="INDEX_ONLY")
    out = run(write_manifest(tmp_path, [row]))
    assert out.returncode == 0, out.stdout
    assert "vault_rows_storing_bytes=0" in out.stdout


def test_a_stored_row_outside_the_vault_is_allowed(tmp_path):
    row = dict(BYTES_ROW, drive_path="01_ACTIVE_RESEARCH_PACKAGES/16_THEMATIC/x")
    out = run(write_manifest(tmp_path, [row]))
    assert out.returncode == 0, out.stdout
    assert "vault_rows_storing_bytes=0" in out.stdout


def test_the_real_tree_stores_no_vault_bytes():
    """The standing invariant, over the repository as it is."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import quarantine_check as Q

    assert Q.vault_rows(ROOT) == []
