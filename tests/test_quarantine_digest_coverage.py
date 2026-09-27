"""Negative controls for invariant 3's coverage accounting in
`tools/quarantine_check.py`.

Invariant 3 is "no excluded payload digest appears in any repository manifest".
It can only test a record that carries a `payload_sha256`.  Six of the twenty-two
exclusions do not: three folders, a native Google document, a superseded source
recorded by byte count, and an EXACT_DUPLICATE whose digest lives in a free-text
`identity` field.  The loop skipped all six in silence while the summary line
printed `exclusions=22`, so a reader auditing quarantine coverage from that line
counted twenty-two comparisons where sixteen had been made.

The fix is accounting, not more comparison.  Promoting those six `identity`
strings into `payload_sha256` would be wrong: an EXACT_DUPLICATE shares its bytes
with a retained keeper *by definition*, so invariant 3 would fire on a correct
tree.  Instead the checker now counts what it compared, requires a record it
cannot compare to say why, and refuses a run in which it compared nothing.

A new exclusion states its reason on the record, in a
`payload_digest_not_compared` field.  These six are declared in the checker's own
`DIGEST_NOT_COMPARED` table because `quarantine/EXCLUSIONS.json` is pinned as
source identity by two RN certificates, and a label is not worth breaking a pin
for.  Both routes are controlled below, and so is the pin itself.

Every control runs the checker through its CLI against synthetic inputs, so a
path bound at import time cannot silently re-check the real repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "quarantine_check.py")
ARCHIVES = os.path.join(ROOT, "drive", "source_map", "Archive_Members.csv")

LEAKED = "a" * 64
CLEAN = "b" * 64

REGISTER_HEADER = [
    "Quarantine key", "Class", "Object / file ID", "Relative path or title",
    "Reason / affected scope", "Keeper or repair", "Original parent",
    "Current destination", "SHA-256 / identity", "Restoration test",
    "Disposition UTC",
]


def exclusion(key, **kw):
    """A minimally valid drive_object exclusion; `kw` overrides any field."""
    rec = {
        "key": key,
        "class": "EXACT_DUPLICATE",
        "scope": "no certification claim rests on these bytes",
        "restoration_test": "establish a distinct needed role first",
        "kind": "drive_object",
        "carrier_id": "1Synthetic" + key,
        "identity": "synthetic record for a negative control",
    }
    rec.update(kw)
    return rec


def write_case(tmp_path, exclusions, manifest_rows=()):
    """Lay out a synthetic run and return the CLI arguments that point at it.

    The register is generated from the exclusions so invariant 1 is satisfied by
    construction: these controls are about invariant 3, and a register mismatch
    would fail the run for the wrong reason.  Binding and manifest indexes are
    empty, so invariant 5 has nothing to say either.
    """
    (tmp_path / "quarantine").mkdir(parents=True, exist_ok=True)
    ex_path = tmp_path / "quarantine" / "EXCLUSIONS.json"
    ex_path.write_text(json.dumps({"exclusions": list(exclusions)}, indent=1),
                       encoding="utf-8")

    rows = [[e["key"], e["class"], e.get("carrier_id", ""), "", "", "", "", "",
             e.get("identity", ""), e.get("restoration_test", ""), ""]
            for e in exclusions]
    reg_path = tmp_path / "quarantine_index.json"
    reg_path.write_text(json.dumps({"header": REGISTER_HEADER, "rows": rows}),
                        encoding="utf-8")

    empty = tmp_path / "empty_index.json"
    empty.write_text(json.dumps({"carriers": []}), encoding="utf-8")

    scan = tmp_path / "scan"
    lane = scan / "drive" / "mirrors" / "LANE"
    lane.mkdir(parents=True, exist_ok=True)
    with open(lane / "_MANIFEST.jsonl", "w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row) + "\n")

    return ["--exclusions", str(ex_path), "--register", str(reg_path),
            "--archives", ARCHIVES, "--binding", str(empty),
            "--manifest", str(empty), "--scan-root", str(scan)]


def run(argv):
    return subprocess.run([sys.executable, CHECKER] + list(argv),
                          capture_output=True, text=True, cwd=ROOT)


def stored_row(sha):
    return {"id": "1SyntheticRow", "title": "a stored object",
            "drive_path": "01_ACTIVE_RESEARCH_PACKAGES/16_THEMATIC/x",
            "dest": "x.md", "bytes": 12, "sha256": sha, "exact": True,
            "stored": True}


# --- the gap itself -------------------------------------------------------

def test_a_record_the_comparison_cannot_see_must_say_why(tmp_path):
    """The whole point: silence is what the run used to permit."""
    out = run(write_case(tmp_path, [
        exclusion("Q-SYN-DIGEST", payload_sha256=CLEAN),
        exclusion("Q-SYN-SILENT"),
    ]))
    assert out.returncode != 0, out.stdout
    assert "Q-SYN-SILENT: no `payload_sha256`" in out.stdout
    assert "payload_digest_not_compared" in out.stdout
    assert "digest_comparable=1 digest_not_compared=1" in out.stdout


def test_a_stated_reason_is_accepted(tmp_path):
    out = run(write_case(tmp_path, [
        exclusion("Q-SYN-DIGEST", payload_sha256=CLEAN),
        exclusion("Q-SYN-FOLDER",
                  payload_digest_not_compared="names a folder, which has no payload"),
    ]))
    assert out.returncode == 0, out.stdout
    assert "digest_comparable=1 digest_not_compared=1" in out.stdout


def test_a_blank_reason_is_not_a_reason(tmp_path):
    """Whitespace would satisfy a truthiness test and say nothing."""
    out = run(write_case(tmp_path, [
        exclusion("Q-SYN-DIGEST", payload_sha256=CLEAN),
        exclusion("Q-SYN-BLANK", payload_digest_not_compared="   \n  "),
    ]))
    assert out.returncode != 0, out.stdout
    assert "Q-SYN-BLANK: no `payload_sha256`" in out.stdout


# --- the vacuity floor ----------------------------------------------------

def test_a_run_that_compared_nothing_is_refused(tmp_path):
    """Every record excused is a green run in which invariant 3 did no work."""
    out = run(write_case(tmp_path, [
        exclusion("Q-SYN-A", payload_digest_not_compared="a folder"),
        exclusion("Q-SYN-B", payload_digest_not_compared="a native document"),
    ]))
    assert out.returncode != 0, out.stdout
    assert "VACUOUS RUN" in out.stdout
    assert "digest_comparable=0 digest_not_compared=2" in out.stdout


def test_one_comparable_record_clears_the_floor(tmp_path):
    out = run(write_case(tmp_path, [
        exclusion("Q-SYN-A", payload_digest_not_compared="a folder"),
        exclusion("Q-SYN-B", payload_sha256=CLEAN),
    ]))
    assert out.returncode == 0, out.stdout
    assert "VACUOUS RUN" not in out.stdout


# --- invariant 3 still bites ----------------------------------------------

def test_a_leaked_payload_is_still_caught(tmp_path):
    """The accounting must not have cost the check it accounts for."""
    out = run(write_case(tmp_path,
                         [exclusion("Q-SYN-LEAK", payload_sha256=LEAKED)],
                         manifest_rows=[stored_row(LEAKED)]))
    assert out.returncode != 0, out.stdout
    assert "Q-SYN-LEAK: excluded payload" in out.stdout
    assert "is consumed by manifest" in out.stdout


def test_a_reason_does_not_excuse_a_record_that_does_carry_a_digest(tmp_path):
    """`payload_digest_not_compared` must not become an opt-out.

    A record that carries a digest is comparable, so the reason field is inert
    on it -- the comparison runs anyway.  If the reason short-circuited the
    loop, an exclusion could be made invisible to invariant 3 by annotating it.
    """
    out = run(write_case(
        tmp_path,
        [exclusion("Q-SYN-BOTH", payload_sha256=LEAKED,
                   payload_digest_not_compared="please do not look at this one")],
        manifest_rows=[stored_row(LEAKED)]))
    assert out.returncode != 0, out.stdout
    assert "Q-SYN-BOTH: excluded payload" in out.stdout
    assert "digest_comparable=1 digest_not_compared=0" in out.stdout


def test_a_clean_digest_is_not_reported_as_a_leak(tmp_path):
    out = run(write_case(tmp_path,
                         [exclusion("Q-SYN-CLEAN", payload_sha256=CLEAN)],
                         manifest_rows=[stored_row(LEAKED)]))
    assert out.returncode == 0, out.stdout
    assert "digest_comparable=1 digest_not_compared=0" in out.stdout


# --- the repository as it stands ------------------------------------------

def summary(stdout):
    line = [l for l in stdout.splitlines() if l.startswith("exclusions=")]
    assert len(line) == 1, stdout
    return dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", line[0]))


def test_the_real_run_reports_sixteen_of_twenty_two():
    out = run([])
    assert out.returncode == 0, out.stdout
    s = summary(out.stdout)
    assert s["exclusions"] == 22
    assert s["digest_comparable"] == 16
    assert s["digest_not_compared"] == 6


def test_the_two_counters_partition_the_register():
    """No record may be counted twice, and none may fall out of both counts."""
    out = run([])
    s = summary(out.stdout)
    assert s["digest_comparable"] + s["digest_not_compared"] == s["exclusions"]


def real_exclusions():
    with open(os.path.join(ROOT, "quarantine", "EXCLUSIONS.json"),
              encoding="utf-8") as handle:
        return json.load(handle)


def real_case(tmp_path, exclusions):
    """The real archives, bindings and tree, with a mutated exclusion list.

    The register is regenerated so invariant 1 stays satisfied and the run fails
    -- or passes -- on the reason accounting rather than on a key mismatch.
    """
    ex_path = tmp_path / "EXCLUSIONS_mutated.json"
    ex_path.write_text(json.dumps({"exclusions": exclusions}, indent=1),
                       encoding="utf-8")
    with open(os.path.join(ROOT, "registers", "json", "quarantine_index.json"),
              encoding="utf-8") as handle:
        reg = json.load(handle)
    h = reg["header"]
    ki, ci = h.index("Quarantine key"), h.index("Class")
    rows = {r[ki]: list(r) for r in reg["rows"] if r and r[ki]}
    out = []
    for e in exclusions:
        row = rows.get(e["key"])
        if row is None:
            row = [""] * len(h)
            row[ki], row[ci] = e["key"], e["class"]
        out.append(row)
    reg_path = tmp_path / "quarantine_index_mutated.json"
    reg_path.write_text(json.dumps({"header": h, "rows": out}), encoding="utf-8")
    return ["--exclusions", str(ex_path), "--register", str(reg_path)]


def test_a_new_uncompared_exclusion_cannot_join_the_unchecked_set_silently(tmp_path):
    """The property the whole change exists for, over the real register."""
    doc = real_exclusions()
    doc["exclusions"].append(exclusion("Q-SYN-NEWCOMER"))
    out = run(real_case(tmp_path, doc["exclusions"]))
    assert out.returncode != 0, out.stdout
    assert "Q-SYN-NEWCOMER: no `payload_sha256`" in out.stdout
    assert "digest_comparable=16 digest_not_compared=7" in out.stdout


def test_the_newcomer_is_accepted_once_it_says_why(tmp_path):
    doc = real_exclusions()
    doc["exclusions"].append(exclusion(
        "Q-SYN-NEWCOMER",
        payload_digest_not_compared="names a folder, which has no payload"))
    out = run(real_case(tmp_path, doc["exclusions"]))
    assert out.returncode == 0, out.stdout
    assert "digest_comparable=16 digest_not_compared=7" in out.stdout


# --- the declared table may not outlive the records it describes ----------

def test_a_declared_reason_for_a_record_that_gained_a_digest_is_refused(tmp_path):
    """`Q-R17-VAULT` is not bound by any index, so only the table speaks for it."""
    doc = real_exclusions()
    for e in doc["exclusions"]:
        if e["key"] == "Q-R17-VAULT":
            e["payload_sha256"] = CLEAN
    out = run(real_case(tmp_path, doc["exclusions"]))
    assert out.returncode != 0, out.stdout
    assert "DIGEST_NOT_COMPARED names Q-R17-VAULT" in out.stdout
    assert "has stopped being true" in out.stdout


def test_a_declared_reason_for_a_record_that_is_gone_is_refused(tmp_path):
    doc = real_exclusions()
    kept = [e for e in doc["exclusions"] if e["key"] != "Q-R17-LOCAL-TB"]
    out = run(real_case(tmp_path, kept))
    assert out.returncode != 0, out.stdout
    assert "DIGEST_NOT_COMPARED names 'Q-R17-LOCAL-TB'" in out.stdout
    assert "is not an exclusion" in out.stdout


def test_every_uncompared_record_has_a_reason_from_one_route_or_the_other():
    """Read off the data and the table directly, not through the checker, so a
    checker weakened to stop asking is not the only thing standing here."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import quarantine_check as Q

    silent = [e["key"] for e in real_exclusions()["exclusions"]
              if not (e.get("payload_sha256") or "").strip()
              and not Q.uncompared_reason(e)]
    assert silent == [], silent


def test_the_six_declared_keys_are_the_six_uncompared_records():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import quarantine_check as Q

    uncompared = {e["key"] for e in real_exclusions()["exclusions"]
                  if not (e.get("payload_sha256") or "").strip()}
    assert set(Q.DIGEST_NOT_COMPARED) == uncompared


# --- the pin that put the reasons in the checker --------------------------

EXCLUSIONS_PINNED_SHA256 = (
    "8a5a89012dcd0fece1b3ea882ea2551f8952d333d22a847e06bd7935c255d1ed")
EXCLUSIONS_PINNED_BYTES = 19555


def test_exclusions_json_still_matches_the_certificate_pin():
    """`research/rn/candidates/inner_wedge_20260920_v1.json` binds these bytes
    twice as source identity, and four replay checkers fail closed on a
    mismatch.  This control exists so the next person to reach for
    EXCLUSIONS.json learns that here instead of from a red CI run.

    If this fails because the file was changed on purpose, the remedy is a
    re-pin on the lane that owns those certificates.  Do not simply update the
    constants below; the constants are not the authority, the certificate is.
    """
    with open(os.path.join(ROOT, "quarantine", "EXCLUSIONS.json"), "rb") as h:
        raw = h.read()
    assert len(raw) == EXCLUSIONS_PINNED_BYTES
    assert hashlib.sha256(raw).hexdigest() == EXCLUSIONS_PINNED_SHA256


def test_the_certificate_still_pins_what_this_file_says_it_pins():
    """And the constants above are read back from the certificate, so the two
    cannot drift apart in silence."""
    with open(os.path.join(ROOT, "research", "rn", "candidates",
                           "inner_wedge_20260920_v1.json"), encoding="utf-8") as h:
        cert = json.load(h)
    sites = [
        cert["majorant"]["source_binding"]["authenticated_identities"][
            "quarantine/EXCLUSIONS.json"],
        cert["source_identities"]["quarantine/EXCLUSIONS.json"],
    ]
    for site in sites:
        assert site["sha256"] == EXCLUSIONS_PINNED_SHA256, site
        assert site["bytes"] == EXCLUSIONS_PINNED_BYTES, site


# --- the one uncompared record whose digest does exist --------------------

DUP_001_DIGEST = "cbc52f0b27a50624b0d73e592e544acd2ec6230aea91ef4cc743996f9c941363"
DUP_001_EXCLUDED_OBJECT = "1Y_3zFonLsFIAHP5KSkUfsJXqHZXIUAL2"
DUP_001_KEEPER = "1Hc8dJvdh504xKBBHXswU5Ly-_8uYp_Sv"


def manifest_rows(root=ROOT):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", ".pytest_cache")]
        for fn in filenames:
            if fn not in ("_MANIFEST.jsonl", "MANIFEST.jsonl"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield path, json.loads(line)
                    except json.JSONDecodeError:
                        continue


def test_the_exact_duplicate_digest_is_stored_only_for_the_keeper():
    """`Q-R17-DUP-001` is the reason invariant 3 must not simply be widened.

    Its digest is legitimately in the tree, because an exact duplicate shares
    its bytes with a retained keeper.  What must never happen is the *excluded*
    object being stored.  The prose in quarantine/README.md says so; this is the
    same statement, read off the manifests.
    """
    storers = [(p, r["id"]) for p, r in manifest_rows()
               if r.get("stored") and str(r.get("sha256") or "").lower() == DUP_001_DIGEST]
    assert storers, "the keeper's bytes are gone; the duplicate finding no longer holds"
    assert {i for _p, i in storers} == {DUP_001_KEEPER}, storers


def surplus_copy_storers(root=ROOT):
    return [p for p, r in manifest_rows(root)
            if r.get("id") == DUP_001_EXCLUDED_OBJECT and r.get("stored")]


def test_the_excluded_surplus_copy_stores_no_bytes():
    assert surplus_copy_storers() == []


def test_negative_control_a_stored_surplus_copy_is_seen(tmp_path):
    """The predicate above is not vacuous: it finds the row it is looking for."""
    lane = tmp_path / "drive" / "mirrors" / "LANE"
    lane.mkdir(parents=True)
    (lane / "_MANIFEST.jsonl").write_text(json.dumps({
        "id": DUP_001_EXCLUDED_OBJECT, "stored": True, "dest": "surplus.md",
        "bytes": 4088, "sha256": DUP_001_DIGEST}) + "\n", encoding="utf-8")
    assert len(surplus_copy_storers(str(tmp_path))) == 1
