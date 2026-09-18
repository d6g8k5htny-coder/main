"""Invariants for the content-addressed carrier binding.

The point of these tests is the NEGATIVE CONTROLS: each one takes a working
copy of `engine/carriers/`, breaks one specific thing, and asserts that
`tools/carriers_verify.py` exits non-zero and names the defect. A checker that
cannot fail proves nothing.

None of these tests runs a bound carrier, and none of them asserts anything
mathematical. Binding is not review, not replay and not endorsement.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY = os.path.join(ROOT, "tools", "carriers_verify.py")
MANIFEST = os.path.join(ROOT, "engine", "carriers", "MANIFEST.json")
BLOBS = os.path.join(ROOT, "engine", "carriers", "blobs")
EXCLUSIONS = os.path.join(ROOT, "quarantine", "EXCLUSIONS.json")


def load() -> dict:
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def run_verifier(root: str) -> subprocess.CompletedProcess:
    """Run tools/carriers_verify.py against a (possibly mutated) tree copy."""
    return subprocess.run(
        [sys.executable, os.path.join(root, "tools", "carriers_verify.py")],
        capture_output=True, text=True,
    )


@pytest.fixture()
def sandbox(tmp_path):
    """A copy of just the trees the verifier reads."""
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "drive").mkdir()
    (root / "quarantine").mkdir()
    shutil.copy(VERIFY, root / "tools" / "carriers_verify.py")
    shutil.copy(os.path.join(ROOT, "docs", "OPEN_PROBLEMS.md"), root / "docs")
    shutil.copy(os.path.join(ROOT, "drive", "inventory.jsonl"), root / "drive")
    shutil.copy(EXCLUSIONS, root / "quarantine")
    shutil.copytree(os.path.join(ROOT, "engine", "carriers"),
                    root / "engine" / "carriers")
    return root


def mutate(root, fn):
    path = root / "engine" / "carriers" / "MANIFEST.json"
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    fn(m)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1, ensure_ascii=False)


# --------------------------------------------------------------- positive side

def test_verifier_passes_on_the_repository():
    proc = run_verifier(ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "problems=0" in proc.stdout


def test_sandbox_copy_passes_before_mutation(sandbox):
    proc = run_verifier(sandbox)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_every_stored_blob_rehashes_to_its_content_address():
    for rec in load()["carriers"]:
        if not rec["blob_stored"]:
            continue
        path = os.path.join(ROOT, rec["blob_path"])
        raw = open(path, "rb").read()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == rec["sha256"], rec["carrier_id"]
        assert len(raw) == rec["bytes"], rec["carrier_id"]
        assert os.path.basename(path).startswith(digest[:16] + "__")


def test_no_carrier_claims_to_be_certifying():
    # Binding records bytes. Nothing in this manifest is a certified bound.
    for rec in load()["carriers"]:
        assert rec["certifying"] is False, rec["carrier_id"]


def test_no_bound_carrier_comes_from_the_legacy_archive():
    for rec in load()["carriers"]:
        assert not rec["drive_path"].startswith("02_LEGACY_Q0_ARCHIVE"), rec["carrier_id"]


def test_excluded_payloads_are_never_stored():
    with open(EXCLUSIONS, encoding="utf-8") as f:
        ex = json.load(f)["exclusions"]
    digests = {(e.get("payload_sha256") or "").lower() for e in ex} - {""}
    objects = {e.get("carrier_id") for e in ex if e.get("kind") == "drive_object"}
    for rec in load()["carriers"]:
        if rec["sha256"].lower() in digests or rec["drive_id"] in objects:
            assert rec["blob_stored"] is False, rec["carrier_id"]


def test_quarantine_check_still_passes():
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "quarantine_check.py")],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ------------------------------------------------------------ negative controls

def test_negative_control_corrupted_blob_is_caught(sandbox):
    rec = next(r for r in load()["carriers"] if r["blob_stored"])
    target = sandbox / rec["blob_path"]
    data = bytearray(target.read_bytes())
    data[0] ^= 0x01                      # flip one bit of one byte
    target.write_bytes(bytes(data))
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "hashes to" in proc.stdout


def test_negative_control_truncated_blob_is_caught(sandbox):
    rec = next(r for r in load()["carriers"] if r["blob_stored"])
    target = sandbox / rec["blob_path"]
    target.write_bytes(target.read_bytes()[:-1])
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "hashes to" in proc.stdout or "bytes, manifest says" in proc.stdout


def test_negative_control_quarantined_digest_admitted_is_caught(sandbox):
    """Admit an excluded archive payload as a stored blob: must be rejected."""
    with open(EXCLUSIONS, encoding="utf-8") as f:
        ex = json.load(f)["exclusions"]
    victim = next(e for e in ex if e.get("payload_sha256"))
    digest = victim["payload_sha256"].lower()
    blob = f"engine/carriers/blobs/{digest[:16]}__smuggled.py"
    (sandbox / blob).write_bytes(b"# smuggled\n")

    def admit(m):
        m["carriers"].append({
            "carrier_id": "CR-SMUGGLED", "title": victim["member_path"],
            "drive_id": m["carriers"][0]["drive_id"],
            "drive_path": m["carriers"][0]["drive_path"],
            "bytes": len(b"# smuggled\n"), "sha256": digest,
            "blob_stored": True, "blob_path": blob, "lane": "A1",
            "computes": "smuggled", "arithmetic": "unknown", "certifying": False,
            "certifying_note": "smuggled", "authority_tier": "active",
            "source_status": "TEXT_READING_COPY",
        })
        m["carriers_bound"] += 1
        m["blobs_stored"] += 1

    mutate(sandbox, admit)
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "excluded by" in proc.stdout and victim["key"] in proc.stdout


def test_negative_control_quarantined_drive_object_stored_is_caught(sandbox):
    """Flip the SUPERSEDED RN carrier to stored: must be rejected."""
    rec = next(r for r in load()["carriers"]
               if r.get("not_stored_reason") == "QUARANTINE_EXCLUSION")
    blob = f"engine/carriers/blobs/{rec['sha256'][:16]}__restored.py"
    (sandbox / blob).write_bytes(b"# restored\n")

    def restore(m):
        for r in m["carriers"]:
            if r["carrier_id"] == rec["carrier_id"]:
                r["blob_stored"] = True
                r["blob_path"] = blob
                r.pop("not_stored_reason", None)
        m["blobs_stored"] += 1

    mutate(sandbox, restore)
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "excluded by" in proc.stdout


def test_negative_control_certifying_true_on_mpmath_carrier_is_caught(sandbox):
    rec = next(r for r in load()["carriers"] if r["arithmetic"] == "mpmath_float")

    def promote(m):
        for r in m["carriers"]:
            if r["carrier_id"] == rec["carrier_id"]:
                r["certifying"] = True

    mutate(sandbox, promote)
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "certifying true with arithmetic 'mpmath_float'" in proc.stdout


def test_negative_control_certifying_true_on_float_carrier_is_caught(sandbox):
    rec = next(r for r in load()["carriers"] if r["arithmetic"] == "float")

    def promote(m):
        for r in m["carriers"]:
            if r["carrier_id"] == rec["carrier_id"]:
                r["certifying"] = True

    mutate(sandbox, promote)
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "certifying true" in proc.stdout


def test_negative_control_invented_lane_is_caught(sandbox):
    def invent(m):
        m["carriers"][0]["lane"] = "A9"

    mutate(sandbox, invent)
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "is not a section of docs/OPEN_PROBLEMS.md" in proc.stdout


def test_negative_control_sha_drift_from_the_inventory_is_caught(sandbox):
    def drift(m):
        rec = next(r for r in m["carriers"] if r["blob_stored"])
        rec["sha256"] = "0" * 64

    mutate(sandbox, drift)
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "sha256 differs from the inventory" in proc.stdout


def test_negative_control_byte_count_drift_is_caught(sandbox):
    def drift(m):
        rec = next(r for r in m["carriers"] if r["blob_stored"])
        rec["bytes"] = rec["bytes"] + 1

    mutate(sandbox, drift)
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "!= inventory" in proc.stdout


def test_negative_control_missing_required_field_is_caught(sandbox):
    def strip(m):
        m["carriers"][0].pop("certifying_note")

    mutate(sandbox, strip)
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "missing required field 'certifying_note'" in proc.stdout


def test_negative_control_orphan_blob_is_caught(sandbox):
    (sandbox / "engine" / "carriers" / "blobs" / "deadbeefdeadbeef__orphan.py"
     ).write_bytes(b"# orphan\n")
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "no manifest record references it" in proc.stdout


def test_negative_control_unexplained_missing_blob_is_caught(sandbox):
    rec = next(r for r in load()["carriers"] if not r["blob_stored"])

    def blank(m):
        for r in m["carriers"]:
            if r["carrier_id"] == rec["carrier_id"]:
                r.pop("not_stored_reason", None)

    mutate(sandbox, blank)
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "not_stored_reason" in proc.stdout


def test_negative_control_legacy_archive_binding_is_caught(sandbox):
    """Bind a 02_LEGACY_Q0_ARCHIVE carrier: zero authority, must be rejected."""
    legacy = None
    with open(os.path.join(ROOT, "drive", "inventory.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if (row["path"].startswith("02_LEGACY_Q0_ARCHIVE")
                    and row.get("sha256")
                    and row.get("access_status") == "TEXT_READING_COPY"):
                legacy = row
                break
    assert legacy is not None, "inventory has no legacy text carrier to test with"

    def bind(m):
        m["carriers"].append({
            "carrier_id": "CR-LEGACY", "title": legacy["title"],
            "drive_id": legacy["id"], "drive_path": legacy["path"],
            "bytes": legacy["bytes"], "sha256": legacy["sha256"],
            "blob_stored": False, "lane": "B", "computes": "legacy",
            "arithmetic": "unknown", "certifying": False,
            "certifying_note": "legacy", "authority_tier": "legacy",
            "source_status": legacy["access_status"],
            "not_stored_reason": "SIZE",
        })
        m["carriers_bound"] += 1

    mutate(sandbox, bind)
    proc = run_verifier(sandbox)
    assert proc.returncode != 0
    assert "02_LEGACY_Q0_ARCHIVE" in proc.stdout
