#!/usr/bin/env python3
"""Accept staged architecture evidence and emit a non-scientific receipt.

Run on the trusted disposable host after sandbox disposal and output staging.
This module reads inert data only: it imports no project, gate, or test code,
executes no staged content, and changes no evidence files. Receipt retention is
the caller's separate stdout redirection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Iterator


FILE_LIMIT = 16 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]
GRAPH_SOURCE = {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "7858329974e28be79f29b22644370084ff43da4f",
    "captured_at": "2026-10-03T15:07:03Z",
    "graph_sha256": "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09",
    "gate_sha256": "a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8",
}
IMAGE_DIGEST = "python@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48"
MEMBERS = (
    "generated/graph.json", "checked-commit.txt", "run-id.txt", "run-attempt.txt",
    "container-exit-status.txt", "tests-normal.log", "tests-optimized.log",
    "runtime-image.json",
)
REVIEW_FIELDS = (
    "review_source", "review_issue", "review_basis", "review_provider",
    "review_providers", "review_disposition",
)
REQUIRED_METHODS = {
    "test_architecture_graph_export.ArchitectureGraphExportContract": (
        "test_pinned_49_node_55_edge_export_is_deterministic_and_preserves_records",
        "test_exact_source_bound_hard_gate_validator_is_called",
        "test_dimensions_describe_only_explicit_recorded_metadata",
        "test_classification_and_proof_review_text_cannot_manufacture_evidence",
        "test_graph_identity_is_checked_before_gate_import",
        "test_gate_identity_is_checked_before_untrusted_gate_executes",
        "test_same_size_gate_change_is_rejected_by_sha_before_import",
        "test_source_byte_counts_are_validated_even_when_sha_matches",
        "test_missing_required_dependency_is_rejected_by_pinned_hard_gate",
        "test_required_dependency_cycle_is_rejected_by_pinned_hard_gate",
        "test_context_only_cycle_is_allowed_without_changing_gate_semantics",
        "test_default_trust_rejects_self_consistent_changed_graph_and_provenance",
        "test_caller_pinned_provenance_cannot_authorize_different_gate_code",
        "test_duplicate_graph_json_keys_are_rejected_after_identity_validation",
        "test_duplicate_provenance_json_keys_are_rejected_before_gate_import",
        "test_symlink_source_files_are_rejected_before_gate_import",
        "test_symlink_source_directory_is_rejected",
        "test_failed_export_preserves_an_earlier_valid_export",
    ),
    "test_architecture_binding.ArchitectureBindingContract": (
        "test_valid_current_run_has_exact_stable_non_scientific_output",
        "test_attempt_ten_is_a_positive_decimal_not_a_single_digit",
        "test_other_valid_native_identities_are_accepted_and_preserved",
        "test_valid_json_at_exact_stdin_byte_limit_is_accepted",
        "test_each_non_success_result_is_refused",
        "test_each_top_level_field_is_required",
        "test_extra_top_level_fields_are_refused",
        "test_non_object_top_level_values_are_refused",
        "test_outputs_must_be_an_object",
        "test_each_of_the_eight_outputs_is_required",
        "test_extra_output_fields_are_refused",
        "test_each_output_rejects_non_string_identities_without_coercion",
        "test_each_output_rejects_an_empty_string",
        "test_stale_checked_commit_is_refused",
        "test_repository_substitution_and_case_drift_are_refused",
        "test_stale_run_is_refused_even_with_a_matching_stale_artifact_name",
        "test_stale_attempt_is_refused_even_with_a_matching_stale_artifact_name",
        "test_each_current_native_argument_is_used_for_identity_binding",
        "test_commit_format_cannot_be_authorized_by_an_equally_invalid_native_commit",
        "test_receipt_and_artifact_sha256_require_lowercase_64_hex_digits",
        "test_artifact_id_requires_positive_ascii_decimal_without_leading_zero",
        "test_run_and_attempt_decimal_format_cannot_be_authorized_by_invalid_native_args",
        "test_artifact_name_is_exactly_bound_to_current_run_and_attempt",
        "test_duplicate_top_level_keys_are_refused_even_when_last_value_is_valid",
        "test_each_duplicate_output_key_is_refused_even_when_last_value_is_valid",
        "test_non_finite_json_numbers_are_refused",
        "test_valid_json_over_stdin_byte_limit_is_refused",
        "test_deeply_nested_json_is_refused_without_a_success_envelope",
        "test_invalid_or_multiple_json_documents_are_refused",
        "test_each_expected_native_identity_argument_is_required",
    ),
    "test_architecture_sandbox.ArchitectureSandbox": (
        "test_nonroot_and_no_new_privileges",
        "test_source_mount_is_read_only_and_environment_has_no_publishing_secret",
        "test_network_namespace_has_no_external_interface_or_route",
        "test_kernel_enforces_cpu_memory_and_process_limits",
        "test_output_quota_stops_disk_exhaustion_and_cleanup_recovers_space",
    ),
}
REQUIRED_IDS = frozenset(
    owner + "." + method
    for owner, methods in REQUIRED_METHODS.items()
    for method in methods
)
HEX40 = re.compile(r"[0-9a-f]{40}")
POSITIVE_DECIMAL = re.compile(r"[1-9][0-9]*")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
TEST_ID = re.compile(
    r"test_architecture_[A-Za-z0-9_]+\.[A-Za-z_][A-Za-z0-9_]*\.test[A-Za-z0-9_]*"
)
TEST_RESULT = re.compile(r"(\S+) \(([^()]*)\) \.\.\. (.+)")
TEST_SUMMARY = re.compile(r"Ran ([1-9][0-9]*) tests in [0-9]+(?:\.[0-9]+)?s")


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Once(argparse.Action):
    """Reject repeated options instead of choosing an ambiguous native value."""

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is not None:
            parser.error(f"{option_string} may be supplied only once")
        setattr(namespace, self.dest, values)


@contextmanager
def directory(path: Path) -> Iterator[int]:
    """Open the entire absolute directory chain without following symlinks."""
    # Retain any '..' components: lexical normalization could erase an earlier
    # symlink component before its O_NOFOLLOW check.
    path = path if path.is_absolute() else Path.cwd() / path
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            following = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        yield descriptor
    finally:
        os.close(descriptor)


def read_member(root_descriptor: int, name: str) -> bytes:
    """Anchor nested parents and read one bounded, nonblocking regular leaf."""
    components = name.split("/")
    descriptor = os.dup(root_descriptor)
    try:
        for component in components[:-1]:
            following = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = following
        leaf = os.open(
            components[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=descriptor,
        )
        with os.fdopen(leaf, "rb") as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode), f"{name} must be a regular file")
            require(info.st_size <= FILE_LIMIT, f"{name} exceeds 16 MiB")
            raw = stream.read(FILE_LIMIT + 1)
            require(len(raw) <= FILE_LIMIT, f"{name} exceeds 16 MiB")
            require(len(raw) == info.st_size, f"{name} changed length while reading")
            return raw
    finally:
        os.close(descriptor)


def strict_json(raw: bytes, name: str) -> object:
    def unique(pairs):
        document = {}
        for key, value in pairs:
            require(key not in document, f"duplicate JSON key in {name}")
            document[key] = value
        return document

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant in {name}: {value}")

    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), f"non-finite JSON number in {name}")
        return number

    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=unique,
        parse_constant=reject_constant, parse_float=finite_float,
    )


def canonical(document: object) -> str:
    """Retain JSON types: false differs from 0, and an integer from a float."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def graph_data(raw: bytes) -> None:
    with directory(ROOT / "docs/site/dependency-source") as descriptor:
        source_raw = read_member(descriptor, "GRAPH.json")
    require(
        hashlib.sha256(source_raw).hexdigest() == GRAPH_SOURCE["graph_sha256"],
        "pinned source GRAPH.json SHA256 mismatch",
    )
    source = strict_json(source_raw, "source GRAPH.json")
    graph = strict_json(raw, "generated/graph.json")
    require(type(graph) is dict, "generated graph must be an object")
    require(
        set(graph) == {
            "schema_version", "source", "nodes", "edges", "dimensions",
            "scientific_effect", "scientific_status_authority",
        },
        "generated graph has invalid root fields",
    )
    require(type(graph["schema_version"]) is int and graph["schema_version"] == 1,
            "generated graph has invalid schema_version")
    require(graph["scientific_effect"] == "NONE" and graph["scientific_status_authority"] is False,
            "generated graph cannot authorize scientific status")
    require(canonical(graph["source"]) == canonical(GRAPH_SOURCE), "graph source identity differs")
    for field in ("nodes", "edges"):
        require(canonical(graph[field]) == canonical(source[field]),
                f"graph {field} differs from pinned source records")
    dimensions = {
        identity: {
            "source": "recorded" if node.get("source") else "not_recorded",
            "review": "recorded" if any(node.get(field) for field in REVIEW_FIELDS) else "not_recorded",
            "kernel": "not_recorded", "computation": "not_recorded", "alignment": "not_recorded",
        }
        for identity, node in source["nodes"].items()
    }
    require(canonical(graph["dimensions"]) == canonical(dimensions),
            "graph evidence dimensions differ from recorded metadata policy")


def inspected_image(raw: bytes) -> dict[str, str]:
    document = strict_json(raw, "runtime-image.json")
    require(type(document) is list and len(document) == 1, "inspect must contain exactly one image")
    image = document[0]
    require(type(image) is dict, "inspected image must be an object")
    identity = image.get("Id")
    require(type(identity) is str and IMAGE_ID.fullmatch(identity), "invalid inspected image ID")
    require(image.get("RepoDigests") == [IMAGE_DIGEST], "inspected image RepoDigests differs")
    require(image.get("Os") == "linux" and image.get("Architecture") == "amd64",
            "inspected image must be linux amd64")
    return {"id": identity, "repo_digest": IMAGE_DIGEST}


def test_log(raw: bytes, mode: str) -> dict[str, object]:
    lines = raw.decode("utf-8").splitlines()
    nonempty = [line for line in lines if line.strip()]
    require(nonempty and nonempty[-1] == "OK", f"{mode} log must end with terminal OK")
    require(sum(line == "OK" for line in lines) == 1, f"{mode} log contains multiple terminal results")
    identities = set()
    footer_index = None
    reported_count = None
    last_test_index = -1
    for index, line in enumerate(lines):
        if line.startswith("Ran "):
            summary = TEST_SUMMARY.fullmatch(line)
            require(summary is not None and footer_index is None, f"{mode} log has invalid or duplicate run footer")
            reported_count = int(summary.group(1))
            footer_index = index
        elif " ... " in line:
            result = TEST_RESULT.fullmatch(line)
            require(result is not None, f"{mode} log has malformed unittest result")
            short_name, identity, disposition = result.groups()
            require(TEST_ID.fullmatch(identity), f"{mode} log has foreign or malformed test identity")
            require(short_name == identity.rsplit(".", 1)[1], f"{mode} log has mismatched test short name")
            require(disposition == "ok", f"{mode} test did not pass: {identity}")
            require(identity not in identities, f"{mode} log repeats a test identity")
            identities.add(identity)
            last_test_index = index
        elif line.startswith(("FAIL:", "ERROR:", "FAILED", "OK (")):
            raise ValueError(f"{mode} log contains a failed, skipped, or non-success result")
    require(footer_index is not None and last_test_index < footer_index, f"{mode} log lacks a final run footer")
    require(REQUIRED_IDS <= identities, f"{mode} log is missing required baseline controls")
    require(reported_count == len(identities), f"{mode} log count differs from discovered test identities")
    return {"mode": mode, "test_count": len(identities), "test_ids": sorted(identities)}


def native_identities(arguments) -> dict[str, str]:
    expected = {
        "checked_commit": arguments.expected_commit,
        "repository": arguments.expected_repository,
        "run_id": arguments.expected_run_id,
        "run_attempt": arguments.expected_run_attempt,
    }
    patterns = {
        "checked_commit": HEX40, "repository": REPOSITORY,
        "run_id": POSITIVE_DECIMAL, "run_attempt": POSITIVE_DECIMAL,
    }
    for field, value in expected.items():
        require(type(value) is str and patterns[field].fullmatch(value), f"invalid native {field}")
    require(all(part not in (".", "..") for part in expected["repository"].split("/")),
            "invalid native repository")
    return expected


def receipt(evidence_dir: Path, expected: dict[str, str]) -> dict[str, object]:
    with directory(evidence_dir) as descriptor:
        members = {name: read_member(descriptor, name) for name in MEMBERS}
    for name, field in (
        ("checked-commit.txt", "checked_commit"), ("run-id.txt", "run_id"),
        ("run-attempt.txt", "run_attempt"),
    ):
        require(members[name] == (expected[field] + "\n").encode("utf-8"),
                f"{name} differs from current native identity")
    require(members["container-exit-status.txt"] == b"0\n", "container did not report exact zero exit status")
    graph_data(members["generated/graph.json"])
    image = inspected_image(members["runtime-image.json"])
    modes = [test_log(members["tests-" + mode + ".log"], mode) for mode in ("normal", "optimized")]
    require(modes[0]["test_ids"] == modes[1]["test_ids"], "normal and optimized test inventories differ")
    return {
        "schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
        **expected, "runtime_image": image, "test_modes": modes, "graph_source": GRAPH_SOURCE,
        "evidence_file_sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--evidence-dir", required=True, type=Path, action=Once)
    for option in ("commit", "repository", "run-id", "run-attempt"):
        parser.add_argument("--expected-" + option, required=True, action=Once)
    arguments = parser.parse_args(argv)
    try:
        document = receipt(arguments.evidence_dir, native_identities(arguments))
        encoded = canonical(document)
    except (ValueError, UnicodeError, OSError, RecursionError) as exc:
        parser.exit(1, f"architecture receipt refused: {exc}\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
