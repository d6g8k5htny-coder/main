"""Negative controls for tools/frozen_check.py.

The checker cross-checks the Drive's frozen-object register against the
accessibility source map, offline. A checker that cannot fail is decoration:
each control below breaks the inputs in exactly one way and asserts the checker
refuses. Every control runs on temporary copies; nothing under registers/ or
drive/ is modified, and a final control asserts that.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "frozen_check.py")
FROZEN = os.path.join(ROOT, "registers", "json", "frozen_objects.json")
INVENTORY = os.path.join(ROOT, "drive", "inventory.jsonl")
PAYLOADS = os.path.join(ROOT, "drive", "source_map", "Payloads.csv")
MEMBERS = os.path.join(ROOT, "drive", "source_map", "Archive_Members.csv")

sys.path.insert(0, os.path.join(ROOT, "tools"))
import frozen_check as FC  # noqa: E402


class Inputs:
    """Writable copies of every input, plus an empty allowlist."""

    def __init__(self, tmp_path):
        self.frozen = str(tmp_path / "frozen_objects.json")
        shutil.copyfile(FROZEN, self.frozen)
        self.inventory = str(tmp_path / "inventory.jsonl")
        shutil.copyfile(INVENTORY, self.inventory)
        self.known = str(tmp_path / "KNOWN_FINDINGS.json")
        with open(self.known, "w", encoding="utf-8") as f:
            json.dump({"_comment": "empty allowlist for the controls"}, f)

    def frozen_rows(self):
        with open(self.frozen, encoding="utf-8") as f:
            return json.load(f)

    def write_frozen(self, d):
        with open(self.frozen, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)

    def edit_inventory(self, drive_id, **fields):
        rows = [json.loads(l) for l in open(self.inventory, encoding="utf-8") if l.strip()]
        hit = False
        out = []
        for r in rows:
            if r["id"] == drive_id:
                hit = True
                if fields.get("__delete__"):
                    continue
                r.update(fields)
            out.append(r)
        assert hit, drive_id
        with open(self.inventory, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def run(self, *extra):
        return subprocess.run(
            [sys.executable, CHECKER, "--frozen", self.frozen, "--inventory", self.inventory,
             "--payloads", PAYLOADS, "--members", MEMBERS, "--known", self.known, *extra],
            capture_output=True, text=True)


@pytest.fixture
def inputs(tmp_path):
    return Inputs(tmp_path)


def rows_by_class(d):
    h = d["header"]
    c_cls = h.index("Binding Class")
    whole = [r for r in d["rows"] if r[c_cls][:1] in "AD"]
    body = [r for r in d["rows"] if r[c_cls][:1] in "BC"]
    return whole, body


# ---------------------------------------------------------------------------
# the repository as it stands
# ---------------------------------------------------------------------------

# Where the pinned numbers come from (register export 2026-09-18, the xlsx):
#   rows=195        registers/json/frozen_objects.json data rows: 188 in the
#                   2026-09-17 export plus the seven RN5 objects frozen on
#                   2026-09-17T17:19-17:24Z (RN5-PROOF, RN5-LM004-ERRATUM,
#                   RN5-SCOPE-HOLDS, RN5-MANIFEST, RN5-BUNDLE, RN5-CUSTODY,
#                   RN5-OPS-EVIDENCE), all class "D — EXACT RAW FILE".
#   comparable=62   rows of class A or D: 55 before the refresh + the 7 RN5 rows.
#   match=62        each of the 62 agrees with drive/inventory.jsonl on SHA-256
#                   and byte count; the 7 RN5 Drive ids all resolve there.
#   body_present=17 / not_comparable=116  the 133 class-B/C rows, unchanged by
#                   the refresh (no RN5 row is a body-class row).
RN5_OBJECTS = ["RN5-PROOF", "RN5-LM004-ERRATUM", "RN5-SCOPE-HOLDS", "RN5-MANIFEST",
               "RN5-BUNDLE", "RN5-CUSTODY", "RN5-OPS-EVIDENCE"]


def test_checker_passes_on_the_repository_and_reports_the_partition():
    out = subprocess.run([sys.executable, CHECKER], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    line = [l for l in out.stdout.splitlines() if l.startswith("frozen_check:")][0]
    assert "rows=195" in line and "mismatch=0" in line and "problems=0" in line
    # the partition is a fact about the register: 62 whole-file rows, 133 body rows
    assert "comparable=62" in line and "match=62" in line
    assert "body_present=17" in line and "not_comparable=116" in line


def test_the_seven_rn5_objects_are_class_d_and_match_the_inventory():
    """The refresh added seven class-D rows; each resolves in the inventory and
    agrees with it. A MATCH is agreement between two Drive-side records of the
    same bytes — it re-freezes nothing and moves no status."""
    report, problems = FC.check(FROZEN, INVENTORY, PAYLOADS, MEMBERS)
    assert problems == []
    rows = {e["object_id"]: e for e in report}
    for oid in RN5_OBJECTS:
        assert rows[oid]["binding_class"].startswith("D"), oid
        assert rows[oid]["status"] == FC.MATCH, (oid, rows[oid])
        assert rows[oid]["register_drift_status"] == "PASS — RAW READBACK"


def test_every_row_gets_exactly_one_status_and_the_register_drift_status_is_carried():
    report, problems = FC.check(FROZEN, INVENTORY, PAYLOADS, MEMBERS)
    assert problems == []
    assert len(report) == 195
    assert all(e["status"] for e in report)
    d = json.load(open(FROZEN, encoding="utf-8"))
    c_drift = d["header"].index("Drift Status")
    for e, r in zip(report, d["rows"]):
        assert e["register_drift_status"] == (r[c_drift].strip() if c_drift < len(r) else "")


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------

def test_control_a_flipped_inventory_digest_is_a_mismatch(inputs):
    d = inputs.frozen_rows()
    whole, _ = rows_by_class(d)
    oid, did = whole[0][0], whole[0][1].strip()
    inputs.edit_inventory(did, sha256="0" * 64)
    out = inputs.run()
    assert out.returncode != 0
    assert oid in out.stdout and "disagrees" in out.stdout
    assert "mismatch=1" in out.stdout


def test_control_a_changed_expected_byte_count_is_a_mismatch(inputs):
    d = inputs.frozen_rows()
    whole, _ = rows_by_class(d)
    c_bytes = d["header"].index("Expected Bytes")
    whole[0][c_bytes] = str(int(whole[0][c_bytes]) + 1)
    inputs.write_frozen(d)
    out = inputs.run()
    assert out.returncode != 0
    assert whole[0][0] in out.stdout and "mismatch=1" in out.stdout


def test_control_a_missing_inventory_row_is_a_structural_problem(inputs):
    d = inputs.frozen_rows()
    whole, _ = rows_by_class(d)
    did = whole[1][1].strip()
    inputs.edit_inventory(did, __delete__=True)
    out = inputs.run()
    assert out.returncode != 0
    assert "not in the inventory" in out.stdout and whole[1][0] in out.stdout


def test_control_an_unknown_binding_class_is_refused(inputs):
    d = inputs.frozen_rows()
    c_cls = d["header"].index("Binding Class")
    d["rows"][0][c_cls] = "Z — SOMETHING NEW"
    inputs.write_frozen(d)
    out = inputs.run()
    assert out.returncode != 0
    assert "unknown binding class" in out.stdout


def test_control_a_register_row_without_a_digest_is_refused(inputs):
    d = inputs.frozen_rows()
    c_hex = d["header"].index("Expected SHA-256 / Identity")
    d["rows"][3][c_hex] = "see notes"
    inputs.write_frozen(d)
    out = inputs.run()
    assert out.returncode != 0
    assert "no 64-hex digest" in out.stdout


def test_a_body_class_row_is_never_compared_against_the_whole_file_digest(inputs):
    """The category firewall. A class-B/C digest is of an extracted body; the
    inventory measures the containing object. Even a body digest that matches
    nothing must read NOT_COMPARABLE_OFFLINE, never MISMATCH — a checker that
    compared it to the whole-file digest would report 116 false drifts."""
    d = inputs.frozen_rows()
    _, body = rows_by_class(d)
    c_hex = d["header"].index("Expected SHA-256 / Identity")
    body[0][c_hex] = "f" * 64
    inputs.write_frozen(d)
    out = inputs.run("--json")
    assert out.returncode == 0, out.stdout
    rep = json.loads(out.stdout.split("\nfrozen_check:")[0])
    row = [e for e in rep["rows"] if e["object_id"] == body[0][0]][0]
    assert row["status"] == FC.NOT_COMPARABLE_OFFLINE


def test_the_known_body_whole_file_disagreement_is_not_reported_as_drift():
    """GP-DER-118-v1.10 is a class-B row whose Drive ID is a raw markdown file
    30,933 B long with a 29,293 B marker-delimited body: the whole-file digest
    differs from the body digest by construction. It must not be a MISMATCH."""
    report, _ = FC.check(FROZEN, INVENTORY, PAYLOADS, MEMBERS)
    row = [e for e in report if e["object_id"] == "GP-DER-118-v1.10"][0]
    assert row["binding_class"].startswith("B")
    assert row["status"] in (FC.NOT_COMPARABLE_OFFLINE, FC.BODY_PRESENT_AS_PAYLOAD)


def test_allowlist_matches_exact_strings_only(inputs):
    d = inputs.frozen_rows()
    whole, _ = rows_by_class(d)
    oid, did = whole[0][0], whole[0][1].strip()
    inputs.edit_inventory(did, sha256="0" * 64)
    out = inputs.run()
    problem = [l for l in out.stdout.splitlines() if l.startswith("frozen_objects:")][0]
    # a near miss does not allowlist
    with open(inputs.known, "w", encoding="utf-8") as f:
        json.dump({problem + " ": "near miss"}, f)
    assert inputs.run().returncode != 0
    # the exact string does, and the problem is still printed as allowlisted
    with open(inputs.known, "w", encoding="utf-8") as f:
        json.dump({problem: "recorded for the owner"}, f)
    out = inputs.run()
    assert out.returncode == 0
    assert "(allowlisted)" in out.stdout and oid in out.stdout


def test_the_checker_never_writes_its_inputs():
    before = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (FROZEN, INVENTORY)}
    subprocess.run([sys.executable, CHECKER, "--json"], capture_output=True, text=True, check=True)
    after = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (FROZEN, INVENTORY)}
    assert before == after
    src = open(CHECKER, encoding="utf-8").read()
    assert 'open(' in src and '"w"' not in src and "'w'" not in src
