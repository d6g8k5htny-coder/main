"""CLI negative controls; filesystem integrity only, no scientific verdict."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

CHECKER = Path(__file__).resolve().parents[1] / "tools/manifest_integrity_check.py"


def run(root, *args):
    result = subprocess.run([sys.executable, str(CHECKER), "--root", str(root), "--json", *args],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode in (0, 1), result.stderr
    return result.returncode, json.loads(result.stdout)


def payload(tmp_path, name="payload.txt", body=b"evidence\n"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"dest": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def manifest(tmp_path, rows):
    path = tmp_path / "_MANIFEST.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_valid_jsonl_and_explicit_metadata_exception(tmp_path):
    row = payload(tmp_path)
    manifest(tmp_path, [row, {"stored": False, "note": "tree-only: folder metadata"}])
    code, report = run(tmp_path)
    assert code == 0, report
    assert report["verified_entries"] == report["unique_payload_paths"] == 1
    assert report["excluded_entries"] == 1
    assert "folder metadata" in report["exclusions"][0]


@pytest.mark.parametrize("field", ["dest", "bytes", "sha256"])
def test_missing_required_fields_fail(tmp_path, field):
    row = payload(tmp_path)
    del row[field]
    manifest(tmp_path, [row])
    code, report = run(tmp_path)
    assert code == 1 and report["verified_entries"] == 0


@pytest.mark.parametrize("size", [True, False, -1, 1.5, "9", None, [], {}])
def test_bytes_type_is_strict(tmp_path, size):
    row = payload(tmp_path)
    row["bytes"] = size
    manifest(tmp_path, [row])
    assert run(tmp_path)[0] == 1


@pytest.mark.parametrize("digest", [None, "", "a" * 63, "g" * 64, 123, []])
def test_digest_shape_is_strict(tmp_path, digest):
    row = payload(tmp_path)
    row["sha256"] = digest
    manifest(tmp_path, [row])
    assert run(tmp_path)[0] == 1


def test_wrong_well_formed_digest_fails(tmp_path):
    row = payload(tmp_path)
    row["sha256"] = "a" * 64
    manifest(tmp_path, [row])
    code, report = run(tmp_path)
    assert code == 1 and any("SHA MISMATCH" in p for p in report["problems"])


def test_wrong_size_fails(tmp_path):
    row = payload(tmp_path)
    row["bytes"] += 1
    manifest(tmp_path, [row])
    code, report = run(tmp_path)
    assert code == 1 and any("SIZE MISMATCH" in p for p in report["problems"])


def test_bad_byte_metadata_fails(tmp_path):
    row = payload(tmp_path)
    row["bytes_stored"] = row["bytes"] + 1
    manifest(tmp_path, [row])
    assert run(tmp_path)[0] == 1


@pytest.mark.parametrize("raw", ['[]', 'null', '123', '"text"', '{bad',
                                  '{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}'])
def test_invalid_json_shapes_fail_cleanly(tmp_path, raw):
    (tmp_path / "_MANIFEST.jsonl").write_text(raw + "\n")
    assert run(tmp_path)[0] == 1


def test_duplicate_json_keys_rejected_even_when_final_value_valid(tmp_path):
    row = payload(tmp_path)
    body = json.dumps(row)
    (tmp_path / "_MANIFEST.jsonl").write_text('{"sha256":"' + "a" * 64 + '",' + body[1:] + "\n")
    code, report = run(tmp_path)
    assert code == 1 and any("duplicate JSON key" in p for p in report["problems"])


def test_malformed_utf8_fails(tmp_path):
    (tmp_path / "_MANIFEST.jsonl").write_bytes(b'{"bad":"\xff"}\n')
    assert run(tmp_path)[0] == 1


def test_missing_payload_fails(tmp_path):
    row = payload(tmp_path)
    (tmp_path / row["dest"]).unlink()
    manifest(tmp_path, [row])
    assert run(tmp_path)[0] == 1


def test_basename_substitution_is_not_verification(tmp_path):
    row = payload(tmp_path)
    row["dest"] = "absent/path/payload.txt"
    manifest(tmp_path, [row])
    assert run(tmp_path)[0] == 1


@pytest.mark.parametrize("destination", ["../outside.txt", "/tmp/outside.txt", "C:/outside.txt",
                                         "C:outside.txt", "foo\\bar", "foo//bar", "./payload.txt",
                                         "payload.txt\n", "payload.txt\x00", ""])
def test_unsafe_destinations_fail(tmp_path, destination):
    row = payload(tmp_path)
    row["dest"] = destination
    manifest(tmp_path, [row])
    assert run(tmp_path)[0] == 1


def test_parent_traversal_to_existing_matching_payload_rejected(tmp_path):
    row = payload(tmp_path)
    child = tmp_path / "child"
    child.mkdir()
    row["dest"] = "../payload.txt"
    manifest(child, [row])
    assert run(tmp_path)[0] == 1


def test_absolute_existing_payload_rejected(tmp_path):
    row = payload(tmp_path)
    row["dest"] = str(tmp_path / row["dest"])
    manifest(tmp_path, [row])
    assert run(tmp_path)[0] == 1


def test_symlink_payload_rejected(tmp_path):
    row = payload(tmp_path)
    (tmp_path / "alias.txt").symlink_to(tmp_path / row["dest"])
    row["dest"] = "alias.txt"
    manifest(tmp_path, [row])
    assert run(tmp_path)[0] == 1


def test_symlink_manifest_rejected(tmp_path):
    row = payload(tmp_path)
    backing = tmp_path / "rows.txt"
    backing.write_text(json.dumps(row) + "\n")
    (tmp_path / "_MANIFEST.jsonl").symlink_to(backing)
    assert run(tmp_path)[0] == 1


def test_symlink_scan_directory_rejected(tmp_path):
    row = payload(tmp_path)
    manifest(tmp_path, [row])
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    assert run(tmp_path)[0] == 1


def test_directory_instead_of_file_rejected(tmp_path):
    row = payload(tmp_path)
    (tmp_path / "directory").mkdir()
    row["dest"] = "directory"
    manifest(tmp_path, [row])
    assert run(tmp_path)[0] == 1


def test_missing_scan_root_is_error(tmp_path):
    assert run(tmp_path / "does-not-exist")[0] == 1


def test_zero_manifests_not_a_pass(tmp_path):
    code, report = run(tmp_path)
    assert code == 1 and "no manifests discovered" in report["problems"]


def test_empty_manifest_not_a_pass(tmp_path):
    manifest(tmp_path, [])
    assert run(tmp_path)[0] == 1


def test_exceptions_only_not_a_pass(tmp_path):
    manifest(tmp_path, [{"stored": False, "note": "tree-only: a folder"}])
    code, report = run(tmp_path)
    assert code == 1 and report["verified_entries"] == 0 and report["excluded_entries"] == 1


def test_missing_required_manifest_fails_even_with_other_coverage(tmp_path):
    manifest(tmp_path, [payload(tmp_path)])
    code, report = run(tmp_path, "--require-manifest", "absent/_MANIFEST.jsonl")
    assert code == 1 and report["verified_entries"] == 1


def test_present_required_manifest_passes(tmp_path):
    manifest(tmp_path, [payload(tmp_path)])
    assert run(tmp_path, "--require-manifest", "_MANIFEST.jsonl")[0] == 0


def test_required_manifest_must_be_scanned(tmp_path):
    manifest(tmp_path, [payload(tmp_path)])
    cache = tmp_path / "node_modules"
    cache.mkdir()
    manifest(cache, [payload(cache)])
    assert run(tmp_path, "--require-manifest", "node_modules/_MANIFEST.jsonl")[0] == 1


def test_duplicate_destination_is_error(tmp_path):
    row = payload(tmp_path)
    manifest(tmp_path, [row, row])
    assert run(tmp_path)[0] == 1


def test_stored_and_excluded_is_contradiction(tmp_path):
    row = payload(tmp_path)
    row.update(stored=True, note="skipped: claimed exception")
    manifest(tmp_path, [row])
    assert run(tmp_path)[0] == 1


def test_stored_false_needs_an_exclusion_reason(tmp_path):
    row = payload(tmp_path)
    row["stored"] = False
    manifest(tmp_path, [row])
    assert run(tmp_path)[0] == 1


@pytest.mark.parametrize("note", ["skipped", "tree-only:", "failedSomething", None, 123])
def test_incomplete_exclusion_cannot_hide_missing_payload(tmp_path, note):
    manifest(tmp_path, [{"note": note}])
    assert run(tmp_path)[0] == 1


@pytest.mark.parametrize("name", ["with spaces.txt", "*star.txt", "Unicode-λ.txt", "nested/payload.txt"])
@pytest.mark.parametrize("mode", [" ", "*"])
def test_standard_sha256sum_text_and_binary_modes(tmp_path, name, mode):
    row = payload(tmp_path, name)
    (tmp_path / "MANIFEST.sha256").write_text(f'{row["sha256"].upper()} {mode}{name}\n', encoding="utf-8")
    code, report = run(tmp_path)
    assert code == 0 and report["verified_entries"] == 1


@pytest.mark.parametrize("line", ["nonsense", "a"*63 + "  payload.txt", "g"*64 + "  payload.txt",
                                 "a"*64 + " payload.txt"])
def test_malformed_sha256sum_lines_not_skipped(tmp_path, line):
    row = payload(tmp_path)
    (tmp_path / "MANIFEST.sha256").write_text(f'{row["sha256"]}  payload.txt\n{line}\n')
    assert run(tmp_path)[0] == 1


def test_comment_only_sha256_file_not_a_pass(tmp_path):
    (tmp_path / "MANIFEST.sha256").write_text("# comment\n\n")
    assert run(tmp_path)[0] == 1


def test_overlapping_manifests_count_unique_paths_separately(tmp_path):
    row = payload(tmp_path)
    manifest(tmp_path, [row])
    (tmp_path / "MANIFEST.sha256").write_text(f'{row["sha256"]}  payload.txt\n')
    code, report = run(tmp_path)
    assert code == 0 and report["verified_entries"] == 2 and report["unique_payload_paths"] == 1


def pin_coverage(root, paths):
    config = {"schema": "q0.manifest-coverage/v1", "base_commit": "a" * 40,
              "scope": "Fixture manifest identities only; no research authority.",
              "manifests": [{"path": p, "sha256": hashlib.sha256((root / p).read_bytes()).hexdigest()}
                            for p in paths]}
    (root / "coverage.json").write_text(json.dumps(config))
    return config


def test_pinned_manifest_deletion_fails_with_other_valid_coverage(tmp_path):
    manifest(tmp_path, [payload(tmp_path)])
    child = tmp_path / "second"
    child.mkdir()
    manifest(child, [payload(child)])
    pin_coverage(tmp_path, ["_MANIFEST.jsonl", "second/_MANIFEST.jsonl"])
    assert run(tmp_path, "--coverage", "coverage.json")[0] == 0
    (child / "_MANIFEST.jsonl").unlink()
    code, report = run(tmp_path, "--coverage", "coverage.json")
    assert code == 1 and report["verified_entries"] == 1


def test_pinned_manifest_cannot_turn_payload_into_exclusion(tmp_path):
    manifest(tmp_path, [payload(tmp_path), payload(tmp_path, "other.txt")])
    pin_coverage(tmp_path, ["_MANIFEST.jsonl"])
    manifest(tmp_path, [payload(tmp_path), {"stored": False, "note": "skipped: deliberately removed"}])
    code, report = run(tmp_path, "--coverage", "coverage.json")
    assert code == 1 and any("pinned manifest changed" in p for p in report["problems"])


def test_new_manifest_is_checked_beside_pinned_set(tmp_path):
    manifest(tmp_path, [payload(tmp_path)])
    pin_coverage(tmp_path, ["_MANIFEST.jsonl"])
    child = tmp_path / "new"
    child.mkdir()
    row = payload(child)
    manifest(child, [row])
    code, report = run(tmp_path, "--coverage", "coverage.json")
    assert code == 0 and report["pinned_manifests"] == 1 and report["manifests"] == 2
    (child / row["dest"]).write_bytes(b"corruption")
    assert run(tmp_path, "--coverage", "coverage.json")[0] == 1


@pytest.mark.parametrize("kind", ["missing", "empty", "duplicate", "symlink", "bad_digest", "duplicate_keys"])
def test_invalid_coverage_config_fails(tmp_path, kind):
    manifest(tmp_path, [payload(tmp_path)])
    config = pin_coverage(tmp_path, ["_MANIFEST.jsonl"])
    path = tmp_path / "coverage.json"
    if kind == "missing":
        path.unlink()
    elif kind == "empty":
        config["manifests"] = []
        path.write_text(json.dumps(config))
    elif kind == "duplicate":
        config["manifests"] *= 2
        path.write_text(json.dumps(config))
    elif kind == "symlink":
        path.rename(tmp_path / "config.txt")
        path.symlink_to(tmp_path / "config.txt")
    elif kind == "bad_digest":
        config["manifests"][0]["sha256"] = "g" * 64
        path.write_text(json.dumps(config))
    else:
        path.write_text('{"schema":"wrong",' + json.dumps(config)[1:])
    assert run(tmp_path, "--coverage", "coverage.json")[0] == 1


@pytest.mark.parametrize("flag", ["stored", "exact"])
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_boolean_metadata_cannot_be_ambiguous(tmp_path, flag, value):
    row = payload(tmp_path)
    row[flag] = value
    manifest(tmp_path, [row])
    assert run(tmp_path)[0] == 1


def test_frozen_tree_only_null_exactness_is_an_exclusion_not_verified(tmp_path):
    row = payload(tmp_path)
    manifest(tmp_path, [row, {"stored": False, "exact": None, "bytes": None,
                              "sha256": None, "note": "tree-only: metadata, not downloaded"}])
    code, report = run(tmp_path)
    assert code == 0
    assert report["verified_entries"] == 1
    assert report["excluded_entries"] == 1
