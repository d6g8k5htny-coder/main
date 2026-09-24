"""Negative controls for tools/pinned_sources_check.py.

The tool exists because a certificate's byte bindings were invisible until you
broke one.  Its own failure mode is the mirror image: an inventory that looks
complete while silently skipping a container, a resolution rule that compares
the wrong file, or a green run that compared nothing.  Every control below runs
the tool through its CLI against a synthetic root, so a path bound at import
time cannot quietly re-check the real repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "pinned_sources_check.py")

INDEX_REL = os.path.join("research", "PINNED_SOURCES.md")


def run(root, *extra):
    return subprocess.run(
        [sys.executable, TOOL, "--root", str(root)] + list(extra),
        capture_output=True, text=True, cwd=ROOT)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build(tmp_path, sources, certificate, cert_rel="research/c/candidate.json"):
    """Lay out a synthetic tree: `sources` is {path: bytes}, plus one certificate."""
    for rel, raw in sources.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    cert = tmp_path / cert_rel
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text(json.dumps(certificate, indent=1), encoding="utf-8")
    return tmp_path


def pinned(raw):
    return {"bytes": len(raw), "sha256": sha(raw)}


BODY = b"def f():\n    return 1\n"
OTHER = b"def f():\n    return 2\n"


def simple(tmp_path, body=BODY):
    return build(tmp_path, {"lib/mod.py": body},
                 {"source_identities": {"lib/mod.py": pinned(body)}})


# --- the inventory is generated, and drift fails --------------------------

def test_a_matching_pin_passes_and_writes_the_index(tmp_path):
    root = simple(tmp_path)
    assert run(root, "--write").returncode == 0
    out = run(root)
    assert out.returncode == 0, out.stdout
    assert "pinned_files=1" in out.stdout
    assert "digest_matches=1" in out.stdout
    assert "problems=0" in out.stdout
    assert "lib/mod.py" in (root / INDEX_REL).read_text(encoding="utf-8")


def test_a_missing_index_is_refused(tmp_path):
    out = run(simple(tmp_path))
    assert out.returncode != 0, out.stdout
    assert "is missing; run --write" in out.stdout


def test_an_edited_index_is_refused_and_the_row_is_named(tmp_path):
    root = simple(tmp_path)
    run(root, "--write")
    index = root / INDEX_REL
    index.write_text(index.read_text(encoding="utf-8").replace("lib/mod.py", "lib/other.py"),
                     encoding="utf-8")
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "the index says" in out.stdout and "lib/other.py" in out.stdout
    assert "the certificates say" in out.stdout and "lib/mod.py" in out.stdout
    assert "regenerate it with --write rather than editing it" in out.stdout


# --- the binding itself ---------------------------------------------------

def test_a_changed_pinned_file_is_reported_with_the_remedy(tmp_path):
    """The whole point: say it once, here, instead of four replay rejections."""
    root = simple(tmp_path)
    run(root, "--write")
    (root / "lib/mod.py").write_bytes(OTHER)
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "lib/mod.py: pinned at" in out.stdout
    assert "and this tree holds" in out.stdout
    assert "re-pin on the lane that owns the certificate" in out.stdout
    assert "never editing the expected digest to match" in out.stdout


def test_a_deleted_pinned_file_is_reported(tmp_path):
    root = simple(tmp_path)
    run(root, "--write")
    os.remove(root / "lib/mod.py")
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "this tree holds nothing there" in out.stdout


def test_a_digest_that_matches_with_a_wrong_byte_count_is_refused(tmp_path):
    """An internally inconsistent certificate, which no replay would catch as one."""
    root = build(tmp_path, {"lib/mod.py": BODY},
                 {"source_identities": {"lib/mod.py":
                                        {"bytes": len(BODY) + 1, "sha256": sha(BODY)}}})
    out = run(root, "--write")
    assert out.returncode == 0
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "the digest matches and the byte count does not" in out.stdout


def test_the_bare_digest_shape_is_recognised(tmp_path):
    """`path -> "<sha256>"`, which is how the H3 candidate binds ladder.py.

    A tool that knows only the {bytes, sha256} shape reports zero bindings here
    and reads as a clean run.
    """
    root = build(tmp_path, {"lib/mod.py": BODY},
                 {"payload": {"computational_sources": {"lib/mod.py": sha(BODY)}}})
    run(root, "--write")
    assert "pinned_files=1" in run(root).stdout
    (root / "lib/mod.py").write_bytes(OTHER)
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "lib/mod.py: pinned at" in out.stdout


def test_list_record_siblings_are_not_dropped(tmp_path):
    """`[{path, bytes, sha256}, ...]` names the file in a sibling field.

    The LPW candidate binds its sources this way, and so does
    `DEPENDENCIES.json`.  A walker that only understands map keys, bare
    digests and archive members drops every record in the list.  With no
    other shape present, that drop reads as a clean run that compared nothing.
    """
    root = build(tmp_path,
                 {"lib/mod.py": BODY, "lib/other.py": OTHER},
                 {"sources": [
                     {"path": "lib/mod.py", "bytes": len(BODY), "sha256": sha(BODY),
                      "extraction": "complete local dependency bytes"},
                     {"path": "lib/other.py", "bytes": len(OTHER), "sha256": sha(OTHER),
                      "module": "lib.other"},
                 ]})
    written = run(root, "--write")
    assert written.returncode == 0, written.stdout
    out = run(root)
    assert out.returncode == 0, out.stdout
    assert "pinned_files=2" in out.stdout
    assert "digest_matches=2" in out.stdout
    assert "certificates=1" in out.stdout
    index = (root / INDEX_REL).read_text(encoding="utf-8")
    assert "`lib/mod.py`" in index
    assert "`lib/other.py`" in index
    assert "`sources[]`" in index
    (root / "lib/mod.py").write_bytes(b"changed\n")
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "lib/mod.py: pinned at" in out.stdout
    assert "lib/other.py: pinned at" not in out.stdout


# --- resolution: the defect this repository already fixed once ------------

def test_a_bundle_relative_container_is_not_compared_at_the_root(tmp_path):
    """The `README.md` case.

    A bundle member that shares a name with a repository file must not be
    compared against it.  Resolving by last component is exactly the seal-checker
    bug this repository already found; a tool that repeats it reports a mismatch
    on a file the certificate never named.
    """
    root = build(tmp_path, {"README.md": b"the repository's own readme\n"},
                 {"members": {"README.md": pinned(b"a different readme, inside a bundle\n"),
                              "verification/build.log": pinned(b"log\n")}})
    out = run(root, "--write")
    assert out.returncode == 0, out.stdout
    out = run(root)
    assert "README.md: pinned at" not in out.stdout
    assert "keys_not_repository_relative=2" in out.stdout
    # and with nothing compared, the run is refused rather than read as clean
    assert "VACUOUS RUN" in out.stdout
    assert out.returncode != 0


def test_a_repository_container_is_still_compared_when_a_bundle_one_is_present(tmp_path):
    root = build(tmp_path, {"README.md": b"repo readme\n", "lib/mod.py": BODY},
                 {"members": {"README.md": pinned(b"bundle readme\n")},
                  "source_identities": {"lib/mod.py": pinned(BODY)}})
    run(root, "--write")
    out = run(root)
    assert out.returncode == 0, out.stdout
    assert "pinned_files=1" in out.stdout
    assert "keys_not_repository_relative=1" in out.stdout


def test_a_container_whose_every_pin_drifted_is_still_compared(tmp_path):
    """The hole the evidence-based rule leaves on its own, closed by the index.

    `classify()` calls a container repository-relative when at least one key
    resolves to the bytes it pins.  If *every* pin drifts, that evidence is gone
    and a self-deriving tool would reclassify the container as bundle-relative,
    stop comparing it, and report a clean run over two changed files.  So the
    index records the classification and later runs use what was recorded.
    """
    root = build(tmp_path, {"lib/a.py": BODY, "lib/b.py": BODY},
                 {"source_identities": {"lib/a.py": pinned(BODY),
                                        "lib/b.py": pinned(BODY)}})
    run(root, "--write")
    (root / "lib/a.py").write_bytes(OTHER)
    (root / "lib/b.py").write_bytes(OTHER)
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "lib/a.py: pinned at" in out.stdout
    assert "lib/b.py: pinned at" in out.stdout
    assert "records 'source_identities' as repository and the evidence now reads other" in out.stdout
    assert "VACUOUS RUN" not in out.stdout


def test_a_new_container_must_be_declared_by_regenerating(tmp_path):
    """A certificate that grows a binding surface cannot add it in silence."""
    root = build(tmp_path, {"lib/a.py": BODY},
                 {"source_identities": {"lib/a.py": pinned(BODY)}})
    run(root, "--write")
    cert = root / "research/c/candidate.json"
    doc = json.loads(cert.read_text(encoding="utf-8"))
    doc["payload"] = {"computational_sources": {"lib/a.py": sha(BODY)}}
    cert.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "does not record that container" in out.stdout
    assert "regenerate the index with --write" in out.stdout


# --- content sealed inside a carrier --------------------------------------

def member_zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for name, raw in members.items():
            z.writestr(name, raw)
    return path.read_bytes()


def test_an_archive_member_is_resolved_inside_its_carrier(tmp_path):
    (tmp_path / "carriers").mkdir(parents=True)
    arc = tmp_path / "carriers" / "bundle.zip"
    member_zip(arc, {"inner/mod.py": BODY})
    root = build(tmp_path, {},
                 {"source_identities": {"carriers/bundle.zip::inner/mod.py": pinned(BODY)}})
    run(root, "--write")
    out = run(root)
    assert out.returncode == 0, out.stdout
    assert "pinned_archive_members=1" in out.stdout
    assert "pinned_files=0" in out.stdout

    member_zip(arc, {"inner/mod.py": OTHER})
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "carriers/bundle.zip::inner/mod.py: pinned at" in out.stdout


# --- the vacuity floor ----------------------------------------------------

def test_a_run_with_no_certificates_is_refused(tmp_path):
    (tmp_path / "research").mkdir(parents=True)
    out = run(tmp_path, "--write")
    assert out.returncode == 0
    out = run(tmp_path)
    assert out.returncode != 0, out.stdout
    assert "VACUOUS RUN" in out.stdout
    assert "certificates=0" in out.stdout


def test_a_certificate_with_no_pins_does_not_count_as_one(tmp_path):
    root = build(tmp_path, {}, {"notes": "no digests here"})
    out = run(root)
    assert out.returncode != 0, out.stdout
    assert "certificates=0" in out.stdout
    assert "VACUOUS RUN" in out.stdout


# --- the repository as it stands ------------------------------------------

def summary(stdout):
    line = [l for l in stdout.splitlines() if l.startswith("pinned_sources_check: certificates=")]
    assert len(line) == 1, stdout
    return {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line[0])}


def test_the_real_tree_verifies_every_binding():
    out = run(ROOT)
    assert out.returncode == 0, out.stdout
    s = summary(out.stdout)
    assert s["certificates"] == 10
    assert s["pinned_files"] == 37
    assert s["pinned_archive_members"] == 6
    assert s["digest_matches"] == s["pinned_files"] + s["pinned_archive_members"]
    assert s["unresolved"] == 0
    assert s["problems"] == 0


# The three LPW sources that no map-shaped pin names.  The interval modules in
# the same `sources` list are already bound by other certificates; these drive
# mirrors are present only as `{path, bytes, sha256}` siblings, and their
# digests match the bytes in this tree.
LPW_LIST_RECORD_PATHS = (
    "drive/mirrors/02_RESEARCH_CARRY_FORWARD_CANON/LPW — RAW ARCHIVE INTAKE R05/"
    "03_RAYLEIGH_REPAIR_AND_INTERVAL_CERTIFICATE.md",
    "drive/mirrors/02_RESEARCH_CARRY_FORWARD_CANON/LPW — RAW ARCHIVE INTAKE R05/"
    "checks/interval_repair.py",
    "drive/mirrors/02_RESEARCH_CARRY_FORWARD_CANON/LPW — RAW ARCHIVE INTAKE R05/"
    "raw_reports/lpw_constant.py",
)


def test_the_lpw_list_record_paths_are_present_against_this_tree():
    """Those three paths are listed when the check is run on this tree.

    Dropping list-record siblings removes them from the generated inventory
    and from the comparison, which is the gap this control exists to close.
    """
    out = run(ROOT)
    assert out.returncode == 0, out.stdout
    index = open(os.path.join(ROOT, INDEX_REL), encoding="utf-8").read()
    for path in LPW_LIST_RECORD_PATHS:
        assert f"`{path}`" in index, path
        assert f"{path}: pinned at" not in out.stdout


def test_the_index_names_the_bindings_nobody_would_guess():
    """Three checkers and a document are bound by certificate. That is the fact
    this page exists to publish, so it is asserted rather than left to the
    generator."""
    index = open(os.path.join(ROOT, INDEX_REL), encoding="utf-8").read()
    for surprising in ("tools/drive_coverage.py",
                       "tools/hermite_envelope_report.py",
                       "tools/manifest_integrity_check.py",
                       "docs/HERMITE_GAUSSIAN_ENVELOPE.md",
                       "quarantine/EXCLUSIONS.json"):
        assert f"`{surprising}`" in index, surprising


def test_the_index_is_what_the_certificates_say():
    """Belt and braces: the CLI comparison above, asserted from the test too, so
    a tool weakened to stop comparing is not the only thing standing here."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import pinned_sources_check as P

    files, containers, problems, counts = P.survey(ROOT, "research")
    assert problems == []
    want = P.render(files, containers, counts)
    assert open(os.path.join(ROOT, INDEX_REL), encoding="utf-8").read() == want
