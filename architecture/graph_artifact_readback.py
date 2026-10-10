"""Strict byte-only readback of a bounded architecture artifact.

Caller pins establish declared comparison context, never authenticated custody.
No captured code, producer helper, gate, or test module is imported or executed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import struct
import zlib
from typing import Any


MIB = 1024 * 1024
RAW_LIMIT = 64 * MIB
INFLATED_LIMIT = 64 * MIB
OUTPUT_LIMIT = 16 * MIB
MAX_DEPTH = 64
SOURCE_BYTES = 38753
GRAPH_SOURCE = {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "7858329974e28be79f29b22644370084ff43da4f",
    "captured_at": "2026-10-03T15:07:03Z",
    "graph_sha256": "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09",
    "gate_sha256": "a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8",
}
IMAGE_DIGEST = "python@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48"
MEMBER_LIMITS = {
    "ARTIFACT_SHA256SUMS": 16 * 1024,
    "checked-commit.txt": 128,
    "run-id.txt": 128,
    "run-attempt.txt": 128,
    "container-exit-status.txt": 128,
    "exporter.stdout": 64 * 1024,
    "exporter.stderr": 64 * 1024,
    "runtime-image.json": MIB,
    "generated/graph.json": 4 * MIB,
    "architecture-run-receipt.json": 4 * MIB,
    "console.log": 4 * MIB,
    "tests-normal.log": 16 * MIB,
    "tests-optimized.log": 16 * MIB,
}
EVIDENCE_MEMBERS = frozenset({
    "generated/graph.json", "checked-commit.txt", "run-id.txt",
    "run-attempt.txt", "container-exit-status.txt", "tests-normal.log",
    "tests-optimized.log", "runtime-image.json",
})
BINDING_KEYS = frozenset({
    "checked_commit", "repository", "run_id", "run_attempt",
    "receipt_sha256", "artifact_id", "artifact_sha256", "artifact_name",
})
REVIEW_FIELDS = (
    "review_source", "review_issue", "review_basis", "review_provider",
    "review_providers", "review_disposition",
)
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
POSITIVE_DECIMAL = re.compile(r"[1-9][0-9]*")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
TEST_ID = re.compile(r"test_architecture_[A-Za-z0-9_]+\.[A-Za-z_][A-Za-z0-9_]*\.test[A-Za-z0-9_]*")
TEST_RESULT = re.compile(r"(\S+) \(([^()]*)\) \.\.\. (.+)")
TEST_RESULT_LIKE = re.compile(r"^\s*test|\([^()\n]*\.test[^()\n]*\)|\)\s*\.\.\.", re.IGNORECASE)
TEST_SUMMARY = re.compile(r"Ran ([1-9][0-9]*) tests in ([0-9]+(?:\.[0-9]+)?)s")
REQUIRED_IDS = frozenset("""
test_architecture_binding.ArchitectureBindingContract.test_artifact_id_requires_positive_ascii_decimal_without_leading_zero
test_architecture_binding.ArchitectureBindingContract.test_artifact_name_is_exactly_bound_to_current_run_and_attempt
test_architecture_binding.ArchitectureBindingContract.test_attempt_ten_is_a_positive_decimal_not_a_single_digit
test_architecture_binding.ArchitectureBindingContract.test_commit_format_cannot_be_authorized_by_an_equally_invalid_native_commit
test_architecture_binding.ArchitectureBindingContract.test_deeply_nested_json_is_refused_without_a_success_envelope
test_architecture_binding.ArchitectureBindingContract.test_duplicate_top_level_keys_are_refused_even_when_last_value_is_valid
test_architecture_binding.ArchitectureBindingContract.test_each_current_native_argument_is_used_for_identity_binding
test_architecture_binding.ArchitectureBindingContract.test_each_duplicate_output_key_is_refused_even_when_last_value_is_valid
test_architecture_binding.ArchitectureBindingContract.test_each_expected_native_identity_argument_is_required
test_architecture_binding.ArchitectureBindingContract.test_each_non_success_result_is_refused
test_architecture_binding.ArchitectureBindingContract.test_each_of_the_eight_outputs_is_required
test_architecture_binding.ArchitectureBindingContract.test_each_output_rejects_an_empty_string
test_architecture_binding.ArchitectureBindingContract.test_each_output_rejects_non_string_identities_without_coercion
test_architecture_binding.ArchitectureBindingContract.test_each_top_level_field_is_required
test_architecture_binding.ArchitectureBindingContract.test_extra_output_fields_are_refused
test_architecture_binding.ArchitectureBindingContract.test_extra_top_level_fields_are_refused
test_architecture_binding.ArchitectureBindingContract.test_invalid_or_multiple_json_documents_are_refused
test_architecture_binding.ArchitectureBindingContract.test_non_finite_json_numbers_are_refused
test_architecture_binding.ArchitectureBindingContract.test_non_object_top_level_values_are_refused
test_architecture_binding.ArchitectureBindingContract.test_other_valid_native_identities_are_accepted_and_preserved
test_architecture_binding.ArchitectureBindingContract.test_outputs_must_be_an_object
test_architecture_binding.ArchitectureBindingContract.test_receipt_and_artifact_sha256_require_lowercase_64_hex_digits
test_architecture_binding.ArchitectureBindingContract.test_repository_substitution_and_case_drift_are_refused
test_architecture_binding.ArchitectureBindingContract.test_run_and_attempt_decimal_format_cannot_be_authorized_by_invalid_native_args
test_architecture_binding.ArchitectureBindingContract.test_stale_attempt_is_refused_even_with_a_matching_stale_artifact_name
test_architecture_binding.ArchitectureBindingContract.test_stale_checked_commit_is_refused
test_architecture_binding.ArchitectureBindingContract.test_stale_run_is_refused_even_with_a_matching_stale_artifact_name
test_architecture_binding.ArchitectureBindingContract.test_valid_current_run_has_exact_stable_non_scientific_output
test_architecture_binding.ArchitectureBindingContract.test_valid_json_at_exact_stdin_byte_limit_is_accepted
test_architecture_binding.ArchitectureBindingContract.test_valid_json_over_stdin_byte_limit_is_refused
test_architecture_graph_export.ArchitectureGraphExportContract.test_caller_pinned_provenance_cannot_authorize_different_gate_code
test_architecture_graph_export.ArchitectureGraphExportContract.test_classification_and_proof_review_text_cannot_manufacture_evidence
test_architecture_graph_export.ArchitectureGraphExportContract.test_context_only_cycle_is_allowed_without_changing_gate_semantics
test_architecture_graph_export.ArchitectureGraphExportContract.test_default_trust_rejects_self_consistent_changed_graph_and_provenance
test_architecture_graph_export.ArchitectureGraphExportContract.test_dimensions_describe_only_explicit_recorded_metadata
test_architecture_graph_export.ArchitectureGraphExportContract.test_duplicate_graph_json_keys_are_rejected_after_identity_validation
test_architecture_graph_export.ArchitectureGraphExportContract.test_duplicate_provenance_json_keys_are_rejected_before_gate_import
test_architecture_graph_export.ArchitectureGraphExportContract.test_exact_source_bound_hard_gate_validator_is_called
test_architecture_graph_export.ArchitectureGraphExportContract.test_failed_export_preserves_an_earlier_valid_export
test_architecture_graph_export.ArchitectureGraphExportContract.test_gate_identity_is_checked_before_untrusted_gate_executes
test_architecture_graph_export.ArchitectureGraphExportContract.test_graph_identity_is_checked_before_gate_import
test_architecture_graph_export.ArchitectureGraphExportContract.test_missing_required_dependency_is_rejected_by_pinned_hard_gate
test_architecture_graph_export.ArchitectureGraphExportContract.test_pinned_49_node_55_edge_export_is_deterministic_and_preserves_records
test_architecture_graph_export.ArchitectureGraphExportContract.test_required_dependency_cycle_is_rejected_by_pinned_hard_gate
test_architecture_graph_export.ArchitectureGraphExportContract.test_same_size_gate_change_is_rejected_by_sha_before_import
test_architecture_graph_export.ArchitectureGraphExportContract.test_source_byte_counts_are_validated_even_when_sha_matches
test_architecture_graph_export.ArchitectureGraphExportContract.test_symlink_source_directory_is_rejected
test_architecture_graph_export.ArchitectureGraphExportContract.test_symlink_source_files_are_rejected_before_gate_import
test_architecture_receipt.ArchitectureReceiptContract.test_attempt_ten_and_other_native_repository_are_preserved
test_architecture_receipt.ArchitectureReceiptContract.test_changed_node_or_edge_is_refused_even_when_graph_source_pins_are_unchanged
test_architecture_receipt.ArchitectureReceiptContract.test_directory_or_fifo_member_is_refused_without_waiting_for_a_writer
test_architecture_receipt.ArchitectureReceiptContract.test_duplicate_json_keys_are_refused_even_when_last_value_is_valid
test_architecture_receipt.ArchitectureReceiptContract.test_duplicate_test_identity_or_multiple_runs_are_refused
test_architecture_receipt.ArchitectureReceiptContract.test_each_expected_host_identity_argument_is_used_and_required
test_architecture_receipt.ArchitectureReceiptContract.test_each_native_identity_file_must_match_current_host_arguments
test_architecture_receipt.ArchitectureReceiptContract.test_each_required_sandbox_control_rejects_skipped_failed_or_error_status
test_architecture_receipt.ArchitectureReceiptContract.test_each_staged_member_is_required
test_architecture_receipt.ArchitectureReceiptContract.test_each_staged_member_must_be_regular_and_cannot_be_a_symlink
test_architecture_receipt.ArchitectureReceiptContract.test_each_staged_member_over_16_mib_is_refused
test_architecture_receipt.ArchitectureReceiptContract.test_empty_or_zero_discovered_test_logs_are_refused_in_each_mode
test_architecture_receipt.ArchitectureReceiptContract.test_every_graph_source_pin_is_required_and_cannot_be_substituted
test_architecture_receipt.ArchitectureReceiptContract.test_every_required_identity_is_needed_even_when_53_other_tests_are_ok
test_architecture_receipt.ArchitectureReceiptContract.test_evidence_directory_or_generated_parent_symlink_is_refused
test_architecture_receipt.ArchitectureReceiptContract.test_graph_and_runtime_inspect_require_strict_json
test_architecture_receipt.ArchitectureReceiptContract.test_graph_root_schema_and_non_scientific_flags_are_exact
test_architecture_receipt.ArchitectureReceiptContract.test_invalid_matching_host_identity_does_not_authorize_bad_commit_or_decimal
test_architecture_receipt.ArchitectureReceiptContract.test_later_all_ok_test_methods_are_counted_and_bound
test_architecture_receipt.ArchitectureReceiptContract.test_malformed_extra_result_lines_cannot_be_ignored_before_the_success_footer
test_architecture_receipt.ArchitectureReceiptContract.test_malformed_foreign_or_shortname_mismatched_test_identities_are_refused
test_architecture_receipt.ArchitectureReceiptContract.test_missing_extra_or_forged_dimension_cannot_manufacture_scientific_evidence
test_architecture_receipt.ArchitectureReceiptContract.test_native_repository_argument_must_have_owner_and_repository_components
test_architecture_receipt.ArchitectureReceiptContract.test_nested_record_types_cannot_change_even_when_python_values_compare_equal
test_architecture_receipt.ArchitectureReceiptContract.test_non_finite_numbers_are_refused_even_in_additional_inspect_fields
test_architecture_receipt.ArchitectureReceiptContract.test_nonunderscore_unittest_names_are_accepted_as_additional_controls
test_architecture_receipt.ArchitectureReceiptContract.test_nonunderscore_unittest_prefix_malformed_results_are_not_diagnostics
test_architecture_receipt.ArchitectureReceiptContract.test_nonzero_or_malformed_container_exit_is_refused
test_architecture_receipt.ArchitectureReceiptContract.test_normal_and_optimized_must_discover_the_same_additional_passing_tests
test_architecture_receipt.ArchitectureReceiptContract.test_runtime_inspect_is_one_linux_amd64_image_with_the_fixed_digest
test_architecture_receipt.ArchitectureReceiptContract.test_skip_or_failure_in_additional_discovered_test_is_also_refused
test_architecture_receipt.ArchitectureReceiptContract.test_summary_must_report_exact_discovered_count_and_terminal_ok
test_architecture_receipt.ArchitectureReceiptContract.test_valid_53_test_receipt_is_exact_deterministic_and_preserves_staged_bytes
test_architecture_receipt.ArchitectureReceiptContract.test_valid_inspected_image_id_is_preserved_instead_of_hardcoded
test_architecture_receipt.ArchitectureReceiptContract.test_valid_log_at_exact_16_mib_limit_is_accepted
test_architecture_receipt.ArchitectureReceiptContract.test_valid_log_larger_than_64_kib_is_accepted_and_raw_bytes_are_hashed
test_architecture_sandbox.ArchitectureSandbox.test_kernel_enforces_cpu_memory_and_process_limits
test_architecture_sandbox.ArchitectureSandbox.test_network_namespace_has_no_external_interface_or_route
test_architecture_sandbox.ArchitectureSandbox.test_nonroot_and_no_new_privileges
test_architecture_sandbox.ArchitectureSandbox.test_output_quota_stops_disk_exhaustion_and_cleanup_recovers_space
test_architecture_sandbox.ArchitectureSandbox.test_source_mount_is_read_only_and_environment_has_no_publishing_secret
""".split())


def _require(condition: object) -> None:
    if not condition:
        raise ValueError("architecture graph readback refused")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _encoded(value: Any) -> bytes:
    return (_canonical(value) + "\n").encode("ascii")


def _object(value: Any, fields: set[str] | frozenset[str]) -> dict:
    _require(type(value) is dict and set(value) == fields)
    return value


def _science(document: dict) -> None:
    _require(type(document["schema_version"]) is int and document["schema_version"] == 1)
    _require(document["scientific_effect"] == "NONE" and document["scientific_status_authority"] is False)


def _json_types(value: Any, depth: int = 0) -> None:
    kind = type(value)
    if kind in (dict, list):
        depth += 1
        _require(depth <= MAX_DEPTH)
        if kind is dict:
            for key, child in value.items():
                _require(type(key) is str)
                key.encode("utf-8")
                _json_types(child, depth)
        else:
            for child in value:
                _json_types(child, depth)
    elif kind is str:
        value.encode("utf-8")
    elif kind is float:
        _require(math.isfinite(value))
    else:
        _require(value is None or kind in (bool, int))


def _json(raw: bytes) -> Any:
    def unique(pairs):
        value = {}
        for key, child in pairs:
            _require(key not in value)
            value[key] = child
        return value

    def no_constant(_value):
        raise ValueError("architecture graph readback refused")

    def finite_float(value):
        number = float(value)
        _require(math.isfinite(number))
        return number

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=no_constant, parse_float=finite_float)
        _json_types(value)
        return value
    except (UnicodeError, RecursionError, OverflowError) as exc:
        raise ValueError("architecture graph readback refused") from exc


def validate_expected(expected: dict[str, str]) -> dict[str, str]:
    _object(expected, BINDING_KEYS)
    _require(all(type(value) is str for value in expected.values()))
    _require(sum(len(value) for value in expected.values()) <= OUTPUT_LIMIT)
    for field, pattern in (
        ("checked_commit", HEX40), ("repository", REPOSITORY),
        ("run_id", POSITIVE_DECIMAL), ("run_attempt", POSITIVE_DECIMAL),
        ("receipt_sha256", HEX64), ("artifact_id", POSITIVE_DECIMAL),
        ("artifact_sha256", HEX64),
    ):
        _require(pattern.fullmatch(expected[field]) is not None)
    _require(all(part not in (".", "..") for part in expected["repository"].split("/")))
    _require(expected["artifact_name"] == "research-architecture-" + expected["run_id"] + "-" + expected["run_attempt"])
    return dict(expected)


def _fields(layout: str, raw: bytes, offset: int, boundary: int) -> tuple:
    _require(0 <= offset and offset + struct.calcsize(layout) <= boundary <= len(raw))
    return struct.unpack_from(layout, raw, offset)


def _archive_geometry(raw: bytes) -> list[dict[str, Any]]:
    """Inspect the complete closed inventory and every span before inflation."""
    footer_start = len(raw) - 22
    footer = _fields("<IHHHHIIH", raw, footer_start, len(raw))
    signature, disk, cd_disk, disk_count, count, cd_size, cd_start, comment = footer
    _require(signature == 0x06054B50 and disk == cd_disk == 0 and comment == 0)
    _require(disk_count == count == len(MEMBER_LIMITS))
    _require(cd_size != 0xFFFFFFFF and cd_start != 0xFFFFFFFF)
    _require(cd_start + cd_size == footer_start)
    rows = []
    names = set()
    folded = set()
    position = cd_start
    declared_total = 0
    for _ in range(count):
        central = _fields("<IHHHHHHIIIHHHHHII", raw, position, footer_start)
        (signature, creator, needed, flags, method, _time, _date, crc, compressed,
         expanded, name_size, extra_size, comment_size, entry_disk, _internal,
         attributes, local_offset) = central
        _require(signature == 0x02014B50 and method in (0, 8))
        _require(10 <= needed <= 20 and entry_disk == 0)
        allowed = (1 << 3) | (1 << 11) | (6 if method == 8 else 0)
        _require(flags & ~allowed == 0 and extra_size == comment_size == 0)
        _require(compressed != 0xFFFFFFFF and expanded != 0xFFFFFFFF and local_offset != 0xFFFFFFFF)
        record_end = position + 46 + name_size
        _require(record_end <= footer_start and name_size > 0)
        name_bytes = raw[position + 46:record_end]
        try:
            name = name_bytes.decode("ascii")
        except UnicodeError as exc:
            raise ValueError("architecture graph readback refused") from exc
        _require(name.encode("ascii") == name_bytes and name in MEMBER_LIMITS)
        _require(name not in names and name.casefold() not in folded)
        names.add(name)
        folded.add(name.casefold())
        file_type = stat.S_IFMT(attributes >> 16)
        _require(file_type in (0, stat.S_IFREG) and not attributes & (0x08 | 0x10 | 0x40))
        if file_type == 0:
            _require(creator >> 8 == 0)
        _require(expanded <= MEMBER_LIMITS[name])
        declared_total += expanded
        _require(declared_total <= INFLATED_LIMIT)
        rows.append({"name": name, "name_bytes": name_bytes, "needed": needed,
                     "flags": flags, "method": method, "crc": crc, "compressed": compressed,
                     "expanded": expanded, "local_offset": local_offset})
        position = record_end
    _require(position == footer_start and names == set(MEMBER_LIMITS))
    rows.sort(key=lambda row: row["local_offset"])
    cursor = 0
    for index, row in enumerate(rows):
        _require(row["local_offset"] == cursor)
        local = _fields("<IHHHHHIIIHH", raw, cursor, cd_start)
        signature, needed, flags, method, _time, _date, crc, compressed, expanded, name_size, extra_size = local
        _require(signature == 0x04034B50 and 10 <= needed <= 20)
        _require(flags == row["flags"] and method == row["method"] and extra_size == 0)
        _require(name_size == len(row["name_bytes"]))
        data_start = cursor + 30 + name_size
        _require(data_start <= cd_start and raw[cursor + 30:data_start] == row["name_bytes"])
        data_end = data_start + row["compressed"]
        boundary = rows[index + 1]["local_offset"] if index + 1 < len(rows) else cd_start
        _require(data_end <= boundary <= cd_start)
        declared = (crc, compressed, expanded)
        central_declared = (row["crc"], row["compressed"], row["expanded"])
        if flags & 8:
            _require(all(value in (0, expected) for value, expected in zip(declared, central_declared)))
            descriptor_size = boundary - data_end
            # The exact span disambiguates an unsigned descriptor whose CRC
            # happens to equal the optional descriptor signature.
            if descriptor_size == 12:
                descriptor = _fields("<III", raw, data_end, boundary)
            else:
                _require(descriptor_size == 16)
                signed = _fields("<IIII", raw, data_end, boundary)
                _require(signed[0] == 0x08074B50)
                descriptor = signed[1:]
            _require(descriptor == central_declared)
        else:
            _require(declared == central_declared and data_end == boundary)
        row["data_start"] = data_start
        row["data_end"] = data_end
        cursor = boundary
    _require(cursor == cd_start)
    return rows


def _member(raw: bytes, row: dict[str, Any], remaining: int) -> bytes:
    limit = min(MEMBER_LIMITS[row["name"]], remaining)
    _require(row["expanded"] <= limit)
    parts = []
    total = 0
    crc = 0
    decoder = zlib.decompressobj(-15) if row["method"] == 8 else None
    start, end = row["data_start"], row["data_end"]
    for position in range(start, end, 64 * 1024):
        block = memoryview(raw)[position:min(position + 64 * 1024, end)]
        if decoder is None:
            part = bytes(block)
        else:
            try:
                part = decoder.decompress(block, min(limit, row["expanded"]) - total + 1)
            except zlib.error as exc:
                raise ValueError("architecture graph readback refused") from exc
            _require(not decoder.unconsumed_tail and not decoder.unused_data)
        total += len(part)
        _require(total <= limit and total <= row["expanded"])
        parts.append(part)
        crc = zlib.crc32(part, crc)
        if decoder is not None and decoder.eof:
            _require(position + len(block) == end)
    if decoder is not None:
        _require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail)
    _require(total == row["expanded"] and crc & 0xFFFFFFFF == row["crc"])
    return b"".join(parts)


def _members(raw: bytes) -> dict[str, bytes]:
    rows = _archive_geometry(raw)
    members = {}
    total = 0
    for row in rows:
        member = _member(raw, row, INFLATED_LIMIT - total)
        total += len(member)
        _require(total <= INFLATED_LIMIT)
        members[row["name"]] = member
    return members


def _manifest(members: dict[str, bytes]) -> None:
    raw = members["ARTIFACT_SHA256SUMS"]
    _require(raw.endswith(b"\n"))
    lines = raw[:-1].split(b"\n")
    _require(len(lines) == 12)
    expected_names = set(MEMBER_LIMITS) - {"ARTIFACT_SHA256SUMS"}
    seen = set()
    for line in lines:
        _require(re.fullmatch(rb"[0-9a-f]{64}  [A-Za-z0-9_./-]+", line) is not None)
        digest, name_bytes = line.split(b"  ", 1)
        name = name_bytes.decode("ascii")
        _require(name in expected_names and name not in seen)
        _require(digest.decode("ascii") == _sha(members[name]))
        seen.add(name)
    _require(seen == expected_names)


def _source_graph(raw: bytes) -> dict:
    _require(type(raw) is bytes and len(raw) == SOURCE_BYTES and _sha(raw) == GRAPH_SOURCE["graph_sha256"])
    source = _json(raw)
    _require(type(source) is dict and type(source.get("nodes")) is dict and len(source["nodes"]) == 49)
    _require(type(source.get("edges")) is list and len(source["edges"]) == 55)
    return source


def _graph(raw: bytes, source: dict) -> dict:
    graph = _object(_json(raw), {
        "schema_version", "source", "nodes", "edges", "dimensions",
        "scientific_effect", "scientific_status_authority",
    })
    _science(graph)
    _require(_canonical(graph["source"]) == _canonical(GRAPH_SOURCE))
    for field in ("nodes", "edges"):
        _require(_canonical(graph[field]) == _canonical(source[field]))
    dimensions = {
        identity: {
            "source": "recorded" if node.get("source") else "not_recorded",
            "review": "recorded" if any(node.get(field) for field in REVIEW_FIELDS) else "not_recorded",
            "kernel": "not_recorded", "computation": "not_recorded", "alignment": "not_recorded",
        } for identity, node in source["nodes"].items()
    }
    _require(_canonical(graph["dimensions"]) == _canonical(dimensions))
    return graph


def _image(raw: bytes) -> dict[str, str]:
    inspected = _json(raw)
    _require(type(inspected) is list and len(inspected) == 1 and type(inspected[0]) is dict)
    image = inspected[0]
    identity = image.get("Id")
    _require(type(identity) is str and IMAGE_ID.fullmatch(identity) is not None)
    _require(_canonical(image.get("RepoDigests")) == _canonical([IMAGE_DIGEST]))
    _require(image.get("Os") == "linux" and image.get("Architecture") == "amd64")
    return {"id": identity, "repo_digest": IMAGE_DIGEST}


def _log(raw: bytes, mode: str) -> dict[str, Any]:
    text = raw.decode("utf-8")
    _require(text.endswith("\n") and "\r" not in text and "\x00" not in text)
    lines = text.split("\n")
    nonempty = [line for line in lines if line.strip()]
    _require(bool(nonempty) and nonempty[-1] == "OK" and lines.count("OK") == 1)
    identities = set()
    footer_index = None
    reported_count = None
    last_result = -1
    for index, line in enumerate(lines):
        if line.lstrip().casefold().startswith("ran "):
            summary = TEST_SUMMARY.fullmatch(line)
            _require(summary is not None and footer_index is None)
            reported_count = int(summary.group(1))
            duration = float(summary.group(2))
            _require(math.isfinite(duration) and duration >= 0 and reported_count <= 4096)
            footer_index = index
        elif " ... " in line or TEST_RESULT_LIKE.search(line):
            result = TEST_RESULT.fullmatch(line)
            _require(result is not None)
            short, identity, status = result.groups()
            _require(TEST_ID.fullmatch(identity) is not None and short == identity.rsplit(".", 1)[1])
            _require(status == "ok" and identity not in identities and len(identities) < 4096)
            identities.add(identity)
            last_result = index
        elif (re.match(r"(?:FAIL|ERROR|FAILED)(?:$|[:\s(])", line.lstrip(), re.IGNORECASE)
              or line.lstrip().upper().startswith("OK (")):
            raise ValueError("architecture graph readback refused")
    _require(footer_index is not None and last_result < footer_index)
    _require(REQUIRED_IDS <= identities and reported_count == len(identities))
    return {"mode": mode, "test_count": len(identities), "test_ids": sorted(identities)}


def _receipt(members: dict[str, bytes], expected: dict[str, str], image: dict[str, str]) -> dict:
    raw = members["architecture-run-receipt.json"]
    _require(_sha(raw) == expected["receipt_sha256"])
    receipt = _object(_json(raw), {
        "schema_version", "scientific_effect", "scientific_status_authority", "checked_commit",
        "repository", "run_id", "run_attempt", "runtime_image", "test_modes", "graph_source",
        "evidence_file_sha256",
    })
    _science(receipt)
    for field in ("checked_commit", "repository", "run_id", "run_attempt"):
        _require(type(receipt[field]) is str and receipt[field] == expected[field])
    _require(_canonical(receipt["runtime_image"]) == _canonical(image))
    _require(_canonical(receipt["graph_source"]) == _canonical(GRAPH_SOURCE))
    hashes = _object(receipt["evidence_file_sha256"], EVIDENCE_MEMBERS)
    for name, digest in hashes.items():
        _require(type(digest) is str and HEX64.fullmatch(digest) is not None and digest == _sha(members[name]))
    modes = [_log(members["tests-" + mode + ".log"], mode) for mode in ("normal", "optimized")]
    _require(modes[0]["test_ids"] == modes[1]["test_ids"])
    _require(_canonical(receipt["test_modes"]) == _canonical(modes))
    return receipt


def read_graph_archive(raw_zip: bytes, expected: dict[str, str], source_graph_bytes: bytes) -> dict:
    """Compare declared pins and preserve validated original records; perform no I/O."""
    binding = validate_expected(expected)
    _require(type(raw_zip) is bytes and 22 <= len(raw_zip) <= RAW_LIMIT)
    _require(_sha(raw_zip) == binding["artifact_sha256"])
    source = _source_graph(source_graph_bytes)
    members = _members(raw_zip)
    _manifest(members)
    for name, field in (
        ("checked-commit.txt", "checked_commit"), ("run-id.txt", "run_id"),
        ("run-attempt.txt", "run_attempt"),
    ):
        _require(members[name] == (binding[field] + "\n").encode("ascii"))
    _require(members["container-exit-status.txt"] == b"0\n")
    _require(members["exporter.stderr"] == b"" and members["exporter.stdout"] == b"/output/generated/graph.json\n")
    graph = _graph(members["generated/graph.json"], source)
    producer = _receipt(members, binding, _image(members["runtime-image.json"]))
    metadata = {"schema_version": 1, "scientific_effect": "NONE",
                "scientific_status_authority": False, "custody": "unknown"}
    unit = {
        **metadata, "graph": graph,
        "receipt": {
            **metadata, "verification_scope": "declared-native-pins-and-archive-content",
            "binding": binding,
            "archive": {
                "sha256": _sha(raw_zip), "bytes": len(raw_zip),
                "members": [{"path": name, "bytes": len(members[name]), "sha256": _sha(members[name])}
                            for name in sorted(members)],
            },
            "graph_member": {"path": "generated/graph.json", "bytes": len(members["generated/graph.json"]),
                             "sha256": _sha(members["generated/graph.json"])},
            "producer_receipt_sha256": _sha(members["architecture-run-receipt.json"]),
            "producer_receipt": producer,
        },
    }
    _require(len(_encoded(unit)) <= OUTPUT_LIMIT)
    return unit
