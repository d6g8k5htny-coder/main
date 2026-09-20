"""Custody mutations only; these tests do not run Lean or earn review credit."""
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import stat
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import lean_receipt as L  # noqa: E402


@pytest.fixture
def bundle():
    return L.archive((ROOT / L.BUNDLE_FILES[0]).read_bytes())


def receipt():
    return L.strict_json((ROOT / L.BUNDLE_FILES[2]).read_bytes())


def replace_report(bundle, mutate):
    report = L.strict_json(bundle["verification/report.json"])
    mutate(report)
    payload = {k: v for k, v in report.items() if k != "payload_sha256"}
    report["payload_sha256"] = L.identity(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())["sha256"]
    bundle["verification/report.json"] = json.dumps(report).encode()
    r = receipt()
    r["verification_payload_sha256"] = report["payload_sha256"]
    return r


def test_exact_bundle_reports_recorded_custody_only():
    result = L.verify(ROOT)
    assert result["status"] == "RECORDED_BUILD_EVIDENCE_CUSTODY_CHECKED"
    assert result["replayed_here"] is False
    assert result["scientific_promotions"] == result["independence_credit"] == 0
    assert result["recorded_theorems"] == 13 and result["recorded_steps"] == 10
    assert set(result["bundle_files"]) == set(L.BUNDLE_FILES)


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r["steps"][-1].update(returncode=0), "expected outcome"),
    (lambda r: r["steps"][-2].update(timed_out=True), "expected outcome"),
    (lambda r: r["steps"].pop(), "step set"),
    (lambda r: r["declarations"].pop(), "theorem declarations"),
    (lambda r: r.update(scientific_promotions=1), "boundary"),
    (lambda r: r.update(independence_credit=False), "boundary"),
    (lambda r: r["source_inputs"]["ResearchFormalCoreR1.lean"].update(sha256="0" * 64), "source inputs"),
    (lambda r: r["steps"][0]["log_identity"].update(bytes=1), "log identity"),
    (lambda r: r.update(axioms_observed=["sorryAx"]), "axiom audit"),
])
def test_report_controls_survive_rehashing(bundle, mutate, match):
    # Deliberately bypass outer pins to exercise each inner consistency check.
    r = replace_report(bundle, mutate)
    with pytest.raises(ValueError, match=match):
        L.contents_check(bundle, r)


def test_proof_body_mutation_is_not_hidden_by_unchanged_statement(bundle):
    name = "recovered/ResearchFormalCoreR1/Algebra.lean"
    bundle[name] = bundle[name].replace(b"  ring\n", b"  sorry\n", 1)
    with pytest.raises(ValueError, match="beyond the permitted"):
        L.contents_check(bundle, receipt())


@pytest.mark.parametrize("mutation", ("extra", "missing", "corrupt"))
def test_every_member_and_exact_member_set_are_checked(bundle, mutation):
    manifest_bytes = (ROOT / L.BUNDLE_FILES[1]).read_bytes()
    manifest = L.strict_json(manifest_bytes)
    readme = (ROOT / L.BUNDLE_FILES[3]).read_bytes()
    if mutation == "extra":
        bundle["extra.txt"] = b"extra"
    elif mutation == "missing":
        del bundle["verification/counterexamples.log"]
    else:
        bundle["verification/build.log"] += b"changed"
    with pytest.raises(ValueError, match="member"):
        L.manifest_check(bundle, manifest, manifest_bytes, readme)


@pytest.mark.parametrize("kind", ("duplicate", "traversal", "absolute", "symlink", "oversized"))
def test_hostile_archives_refused_before_extraction(kind):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if kind == "duplicate":
            z.writestr("same", b"a")
            with pytest.warns(UserWarning):
                z.writestr("same", b"b")
        elif kind == "symlink":
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, b"destination")
        else:
            name = {"traversal": "../escape", "absolute": "/escape", "oversized": "big"}[kind]
            z.writestr(name, b"x" * (L.LIMIT + 1) if kind == "oversized" else b"x")
    with pytest.raises(ValueError):
        L.archive(target.getvalue())


def test_pinned_custody_and_external_receipt_changes_fail(tmp_path):
    for name in L.BUNDLE_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    r = receipt()
    r["verification_steps"] = 9
    (tmp_path / L.BUNDLE_FILES[2]).write_text(json.dumps(r))
    with pytest.raises(ValueError, match="delivery receipt"):
        L.verify(tmp_path)
    shutil.copyfile(ROOT / L.BUNDLE_FILES[2], tmp_path / L.BUNDLE_FILES[2])
    with (tmp_path / L.BUNDLE_FILES[0]).open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="pinned Lean bundle"):
        L.verify(tmp_path)


def test_duplicate_json_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        L.strict_json(b'{"scientific_promotions":0,"scientific_promotions":1}')
