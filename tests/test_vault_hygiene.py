"""Negative controls for tools/vault_hygiene_check.py.

The real tree must match inventory metadata and store no vault bytes. A
synthetic root is used wherever a mutation would otherwise edit the tip.
`DO_NOT_OPEN_BEFORE_HASH_FREEZE` is not the vault and must not be refused.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "vault_hygiene_check.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import vault_hygiene_check as H  # noqa: E402

VAULT_ID = "1VaultDocId________________________________"
EXTERNAL_ID = "1ExternalManifestId______________________"
FREEZE_ID = "1HashFreezeId______________________________"
VAULT_PATH = ("01_ACTIVE_RESEARCH_PACKAGES/99_DO_NOT_OPEN — "
              "SUPERSEDED_MIRRORS_DEAD_ENDS_AND_TRAP_COPIES/ZZ_SUPERSEDED — trap")
EXTERNAL_PATH = ("01_ACTIVE_RESEARCH_PACKAGES/00_DO_NOT_OPEN_MANIFEST — "
                 "what was vaulted and why")
FREEZE_PATH = ("01_ACTIVE_RESEARCH_PACKAGES/02_RESEARCH_CARRY_FORWARD_CANON/"
               "99_EC019 — DO_NOT_OPEN_BEFORE_HASH_FREEZE/object.json")


def run(root):
    return subprocess.run(
        [sys.executable, CHECKER, "--root", root],
        capture_output=True, text=True, cwd=ROOT)


def write_tree(root, rows):
    path = root / "drive" / "vault_tree.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# metadata only", ""]
    for row in rows:
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_inventory(root, records):
    path = root / "drive" / "inventory.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def write_manifest(root, rel, rows):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def base(tmp_path):
    records = [
        {"id": VAULT_ID, "path": VAULT_PATH, "access_status": "DIRECT_NATIVE",
         "bytes": 1024, "mimeType": "application/vnd.google-apps.document", "sha256": None},
        {"id": EXTERNAL_ID, "path": EXTERNAL_PATH, "access_status": "DIRECT_NATIVE",
         "bytes": 3376, "mimeType": "application/vnd.google-apps.document", "sha256": None},
        {"id": FREEZE_ID, "path": FREEZE_PATH, "access_status": "TEXT_READING_COPY",
         "bytes": 12, "mimeType": "application/json", "sha256": "ab" * 32},
    ]
    write_inventory(tmp_path, records)
    write_tree(tmp_path, [
        (VAULT_ID, "DIRECT_NATIVE", "1024", VAULT_PATH),
        (EXTERNAL_ID, "DIRECT_NATIVE", "3376", EXTERNAL_PATH),
    ])
    return tmp_path


def test_the_real_tree_matches_and_activates_nothing():
    out = run(ROOT)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "vault_tree_rows=7" in out.stdout
    assert "vault_ids=6" in out.stdout
    assert "external_manifest_stored=1" in out.stdout
    assert "vault_bytes_stored=0" in out.stdout
    assert "held_no_open_vault=5" in out.stdout
    assert "held_no_open_external=1" in out.stdout
    assert "held_no_open_hash_freeze=32" in out.stdout
    assert "held_no_open_other=2" in out.stdout
    assert "problems=0" in out.stdout
    assert "quarantine ≠ SoT" in out.stdout
    assert "discharges nothing" in out.stdout


def test_a_digest_column_is_refused(tmp_path):
    base(tmp_path)
    tree = tmp_path / "drive" / "vault_tree.txt"
    tree.write_text(
        tree.read_text(encoding="utf-8") + VAULT_ID + "\tDIRECT_NATIVE\t1024\t" + ("a" * 64) + "\n",
        encoding="utf-8")
    out = run(tmp_path)
    assert out.returncode != 0, out.stdout
    assert "digest" in out.stdout


def test_a_vault_id_stored_under_another_path_is_refused(tmp_path):
    """quarantine_check.py keys the vault on drive_path. This keys it on id."""
    base(tmp_path)
    write_manifest(tmp_path, "drive/mirrors/LANE/_MANIFEST.jsonl", [{
        "id": VAULT_ID,
        "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/16_THEMATIC/renamed.export.txt",
        "stored": True,
        "exact": False,
        "bytes": 10,
        "sha256": "b" * 64,
    }])
    out = run(tmp_path)
    assert out.returncode != 0, out.stdout
    assert "vault material" in out.stdout


def test_a_hash_freeze_store_is_not_the_vault(tmp_path):
    base(tmp_path)
    write_manifest(tmp_path, "drive/mirrors/FREEZE/_MANIFEST.jsonl", [{
        "id": FREEZE_ID,
        "drive_path": FREEZE_PATH,
        "stored": True,
        "exact": True,
        "bytes": 12,
        "sha256": "ab" * 32,
    }])
    out = run(tmp_path)
    assert out.returncode == 0, out.stdout
    assert "vault_bytes_stored=0" in out.stdout


def test_copying_the_quarantine_lane_into_research_is_refused(tmp_path):
    base(tmp_path)
    dest = tmp_path / "research" / "90_QUARANTINE_AND_TRIAGE" / "note.md"
    dest.parent.mkdir(parents=True)
    dest.write_text("copied\n", encoding="utf-8")
    out = run(tmp_path)
    assert out.returncode != 0, out.stdout
    assert "does not activate" in out.stdout


def test_an_exact_external_manifest_is_refused(tmp_path):
    base(tmp_path)
    write_manifest(tmp_path, "drive/mirrors/ROOT/_MANIFEST.jsonl", [{
        "id": EXTERNAL_ID,
        "drive_path": EXTERNAL_PATH,
        "stored": True,
        "exact": True,
        "bytes": 3376,
        "sha256": "c" * 64,
    }])
    out = run(tmp_path)
    assert out.returncode != 0, out.stdout
    assert "exact:true" in out.stdout


def test_a_matching_listing_passes(tmp_path):
    base(tmp_path)
    out = run(tmp_path)
    assert out.returncode == 0, out.stdout
    assert "vault_tree_rows=2" in out.stdout
    assert "problems=0" in out.stdout


def test_classifier_does_not_call_the_hash_freeze_folder_the_vault():
    assert H.coverage_class(FREEZE_PATH) == "hash_freeze"
    assert H.coverage_class(VAULT_PATH) == "vault"
    assert H.coverage_class(EXTERNAL_PATH) == "external_manifest"
    assert H.is_vault_path(FREEZE_PATH) is False
