"""External CLI controls for graph archive readback; run only in hosted isolation.

All native identities, logs, console text and runtime-image declarations below
are synthetic TEST fixtures. No fixture log establishes a native run or custody.
The graph records come from the already-public, independently pinned source.
This targeted audit observer is a test instrument, not an operating-system sandbox.
"""

import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from urllib.request import urlopen
import zlib


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools/architecture_graph_artifact_readback.py"
SOURCE = ROOT / "docs/site/dependency-source/GRAPH.json"
SOURCE_SHA = "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09"
SOURCE_BYTES = 38753
IMAGE_DIGEST = "python@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48"
REFUSAL = b"ARCHITECTURE_GRAPH_READBACK_REFUSED\n"
MIB = 1024 * 1024
GRAPH_SOURCE = {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "7858329974e28be79f29b22644370084ff43da4f",
    "captured_at": "2026-10-03T15:07:03Z",
    "graph_sha256": SOURCE_SHA,
    "gate_sha256": "a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8",
}
NATIVE = {"checked_commit": "a" * 40, "repository": "TEST-owner/TEST-repository",
          "run_id": "123", "run_attempt": "1", "artifact_id": "456"}
MEMBERS = (
    "runtime-image.json", "checked-commit.txt", "run-id.txt", "run-attempt.txt",
    "container-exit-status.txt", "console.log", "tests-normal.log", "tests-optimized.log",
    "exporter.stdout", "exporter.stderr", "generated/graph.json", "architecture-run-receipt.json",
)
EVIDENCE = (
    "generated/graph.json", "checked-commit.txt", "run-id.txt", "run-attempt.txt",
    "container-exit-status.txt", "tests-normal.log", "tests-optimized.log", "runtime-image.json",
)
REQUIRED_IDS = tuple("""
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
REQUIRED_IDS_SHA = "b51eaf6cfaede42105ad377c066c4b18789e320ccf542b6817a6850c350fc9ef"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


def document(raw):
    return json.loads(raw)


def synthetic_log(ids=REQUIRED_IDS):
    rows = ["Synthetic TEST fixture only; no executed tests or authenticated custody."]
    rows.extend(identity.rsplit(".", 1)[1] + " (" + identity + ") ... ok" for identity in ids)
    rows.extend(["", "----------------------------------------------------------------------",
                 "Ran " + str(len(ids)) + " tests in 0.001s", "", "OK", ""])
    return "\n".join(rows).encode("ascii")


def padded_log(raw, length):
    footer = raw.index(b"\n----------------------------------------------------------------------\n")
    padding = length - len(raw)
    if padding < 0:
        raise ValueError("TEST requested padding below fixture length")
    return raw[:footer] + b"\n" + b"x" * max(0, padding - 1) + raw[footer:] if padding else raw


def classic_zip(entries, options=None, *, comment=b"", prefix=b"", trailing=b"", zip64=False,
                disk=0):
    """Construct classic ZIP fixtures independently; this is not an acceptance parser."""
    options = {} if options is None else options
    body, central = bytearray(prefix), bytearray()
    for index, (name, raw) in enumerate(entries):
        option = options.get(index, {})
        filename = option.get("name", name)
        filename = filename.encode("utf-8") if isinstance(filename, str) else filename
        method, flags = option.get("method", 8), option.get("flags", 0)
        compressor = zlib.compressobj(level=6, wbits=-15)
        compressed = raw if method == 0 else compressor.compress(raw) + compressor.flush()
        compressed = option.get("compressed", compressed)
        crc, size, expanded = zlib.crc32(raw) & 0xffffffff, len(compressed), len(raw)
        crc, size, expanded = option.get("crc", crc), option.get("size", size), option.get("expanded", expanded)
        extra, member_comment = option.get("extra", b""), option.get("comment", b"")
        body.extend(option.get("before", b""))
        offset = len(body)
        local = {"flags": flags, "method": method, "crc": 0 if flags & 8 else crc,
                 "size": 0 if flags & 8 else size, "expanded": 0 if flags & 8 else expanded}
        local.update(option.get("local", {}))
        local_name = option.get("local_name", filename)
        body.extend(struct.pack("<IHHHHHIIIHH", 0x04034b50, 20, local["flags"], local["method"],
                                0, 33, local["crc"], local["size"], local["expanded"],
                                len(local_name), len(extra)))
        body.extend(local_name + extra + compressed)
        if flags & 8:
            descriptor = struct.pack("<IIII", 0x08074b50, crc, size, expanded)
            body.extend(option.get("descriptor", descriptor))
        creator = option.get("creator", (3 << 8) | 20)
        attributes = option.get("attributes", (stat.S_IFREG | 0o644) << 16)
        central.extend(struct.pack("<IHHHHHHIIIHHHHHII", 0x02014b50, creator, 20, flags, method,
                                   0, 33, crc, size, expanded, len(filename), len(extra),
                                   len(member_comment), 0, 0, attributes, option.get("offset", offset)))
        central.extend(filename + extra + member_comment)
    central_offset = len(body)
    body.extend(central)
    if zip64:
        zip64_offset = len(body)
        body.extend(struct.pack("<IQHHIIQQQQ", 0x06064b50, 44, 45, 45, 0, 0,
                                len(entries), len(entries), len(central), central_offset))
        body.extend(struct.pack("<IIQI", 0x07064b50, 0, zip64_offset, 1))
    body.extend(struct.pack("<IHHHHIIH", 0x06054b50, disk, disk,
                            0xffff if zip64 else len(entries), 0xffff if zip64 else len(entries),
                            0xffffffff if zip64 else len(central),
                            0xffffffff if zip64 else central_offset, len(comment)))
    body.extend(comment + trailing)
    return bytes(body)


AUDIT_LAUNCHER = r'''
import json, os, runpy, socket, sys
cwd, archive, secret, report_path, cli = sys.argv[1:6]
observer_port = int(sys.argv[6])
arguments = sys.argv[7:]
report_fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
labels = ("read", "write", "mkdir", "remove", "rename", "rmdir", "list", "network")
counts = dict.fromkeys(labels, 0)
active = False
opening_path = None
original_open = os.open
def actual_path(value, dir_fd=None):
    if isinstance(value, int):
        try:
            value = os.readlink("/proc/self/fd/" + str(value))
        except OSError:
            return None
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    path = os.fsdecode(os.fspath(value))
    if not os.path.isabs(path):
        if dir_fd not in (None, -1):
            try:
                parent = os.readlink("/proc/self/fd/" + str(dir_fd))
            except OSError:
                return cwd + "/TEST-unresolved-dir-fd"
        else:
            parent = cwd
        path = os.path.join(parent, path)
    return os.path.normpath(path)
def fixture_path(value, dir_fd=None):
    path = actual_path(value, dir_fd)
    if path is None:
        return False
    return path == cwd or path.startswith(cwd + os.sep)
def instrumented_open(path, flags, mode=0o777, *, dir_fd=None):
    global opening_path
    previous = opening_path
    opening_path = actual_path(path, dir_fd)
    try:
        return original_open(path, flags, mode, dir_fd=dir_fd)
    finally:
        opening_path = previous
os.open = instrumented_open
def observed(label):
    counts[label] += 1
    if active:
        raise RuntimeError("TEST fixture observer refused an attempt")
def audit(event, args):
    if event == "open":
        path, mode, flags = args
        path = opening_path if opening_path is not None else actual_path(path)
        if fixture_path(path):
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                observed("write")
            elif path not in (archive, cwd):
                observed("read")
    elif event in ("os.listdir", "os.scandir") and fixture_path(args[0]):
        observed("list")
    elif event in ("os.mkdir", "os.remove", "os.rmdir"):
        label = {"os.mkdir": "mkdir", "os.remove": "remove", "os.rmdir": "rmdir"}[event]
        if fixture_path(args[0], args[-1] if len(args) > 1 else None):
            observed(label)
    elif event in ("os.rename", "os.replace"):
        if fixture_path(args[0], args[2] if len(args) > 2 else None) or fixture_path(args[1], args[3] if len(args) > 3 else None):
            observed("rename")
    elif event == "socket.connect":
        observed("network")
sys.addaudithook(audit)
try:
    with open(secret, "rb") as stream:
        stream.read()
    calibration = os.path.join(cwd, "TEST-calibration")
    with open(calibration, "wb") as stream:
        stream.write(b"TEST calibration only")
    moved = calibration + "-moved"
    os.rename(calibration, moved)
    os.remove(moved)
    os.mkdir(calibration)
    os.listdir(cwd)
    os.rmdir(calibration)
    with socket.socket() as probe:
        probe.settimeout(5)
        probe.connect(("127.0.0.1", observer_port))
    calibration_counts = dict(counts)
    counts = dict.fromkeys(labels, 0)
    active = True
    sys.argv = [cli, *arguments]
    sys.path.insert(0, os.path.dirname(cli))
    runpy.run_path(cli, run_name="__main__")
finally:
    raw = (json.dumps({"calibration": calibration_counts, "attempts": counts,
                      "optimize": sys.flags.optimize}, sort_keys=True) + "\n").encode("ascii")
    os.write(report_fd, raw)
    os.close(report_fd)
'''


class ArchitectureGraphArtifactReadbackContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)
        source_raw = SOURCE.read_bytes()
        self.assertEqual(len(source_raw), SOURCE_BYTES)
        self.assertEqual(sha(source_raw), SOURCE_SHA, "public source fixture changed")
        self.assertEqual(len(REQUIRED_IDS), 89)
        self.assertEqual(sha(("\n".join(sorted(REQUIRED_IDS)) + "\n").encode("ascii")), REQUIRED_IDS_SHA)
        source = document(source_raw)
        self.graph = {
            "schema_version": 1, "source": GRAPH_SOURCE, "nodes": source["nodes"], "edges": source["edges"],
            "dimensions": {identity: {
                "source": "recorded" if node.get("source") else "not_recorded",
                "review": "recorded" if any(node.get(field) for field in (
                    "review_source", "review_issue", "review_basis", "review_provider",
                    "review_providers", "review_disposition")) else "not_recorded",
                "kernel": "not_recorded", "computation": "not_recorded", "alignment": "not_recorded",
            } for identity, node in source["nodes"].items()},
            "scientific_effect": "NONE", "scientific_status_authority": False,
        }
        self.base = self.fixture()

    def fixture(self, context=None, ids=REQUIRED_IDS):
        context = NATIVE if context is None else context
        image = {"Id": "sha256:" + "f" * 64, "RepoDigests": [IMAGE_DIGEST],
                 "Os": "linux", "Architecture": "amd64", "TEST_annotation": "synthetic image declaration"}
        members = {
            "generated/graph.json": encoded(self.graph),
            "checked-commit.txt": (context["checked_commit"] + "\n").encode("ascii"),
            "run-id.txt": (context["run_id"] + "\n").encode("ascii"),
            "run-attempt.txt": (context["run_attempt"] + "\n").encode("ascii"),
            "container-exit-status.txt": b"0\n", "runtime-image.json": encoded([image]),
            "tests-normal.log": synthetic_log(ids), "tests-optimized.log": synthetic_log(ids),
            "console.log": b"Synthetic TEST fixture, inert diagnostics; no executed native run.\n",
            "exporter.stdout": b"/output/generated/graph.json\n", "exporter.stderr": b"",
        }
        receipt = {"schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
                   **{key: context[key] for key in ("checked_commit", "repository", "run_id", "run_attempt")},
                   "runtime_image": {"id": image["Id"], "repo_digest": IMAGE_DIGEST},
                   "test_modes": [{"mode": mode, "test_count": len(ids), "test_ids": sorted(ids)}
                                  for mode in ("normal", "optimized")],
                   "graph_source": GRAPH_SOURCE,
                   "evidence_file_sha256": {name: sha(members[name]) for name in EVIDENCE}}
        members["architecture-run-receipt.json"] = encoded(receipt)
        return members

    def rebind(self, members):
        receipt = document(members["architecture-run-receipt.json"])
        receipt["evidence_file_sha256"] = {name: sha(members[name]) for name in EVIDENCE}
        members["architecture-run-receipt.json"] = encoded(receipt)

    def change_json(self, members, name, change, *, rebind=True):
        value = document(members[name])
        change(value)
        members[name] = encoded(value)
        if rebind:
            self.rebind(members)

    def entries(self, members=None, manifest=None, order=None):
        members = self.base if members is None else members
        if manifest is None:
            manifest = "".join(sha(members[name]) + "  " + name + "\n" for name in sorted(members)).encode("ascii")
        names = list(members) if order is None else order
        return [(name, members[name]) for name in names] + [("ARTIFACT_SHA256SUMS", manifest)]

    def expected(self, raw, members=None, context=None):
        members, context = (self.base if members is None else members), (NATIVE if context is None else context)
        return {**context, "receipt_sha256": sha(members.get("architecture-run-receipt.json", self.base["architecture-run-receipt.json"])),
                "artifact_sha256": sha(raw), "artifact_name": "research-architecture-" + context["run_id"] + "-" + context["run_attempt"]}

    def arguments(self, path, expected):
        aliases = {"checked_commit": "commit", "repository": "repository", "run_id": "run-id",
                   "run_attempt": "run-attempt", "receipt_sha256": "receipt-sha256", "artifact_id": "artifact-id",
                   "artifact_sha256": "artifact-sha256", "artifact_name": "artifact-name"}
        return ["--archive", str(path), *[part for field, flag in aliases.items()
                                        for part in ("--expected-" + flag, expected[field])]]

    def invoke(self, raw=None, *, members=None, expected=None, arguments=None, path=None,
               audit=False, observer_port=None):
        self.assertTrue(CLI.is_file(), "required graph readback CLI tools/architecture_graph_artifact_readback.py is absent")
        members = self.base if members is None else members
        raw = classic_zip(self.entries(members)) if raw is None else raw
        path = self.cwd / "TEST-input.zip" if path is None else path
        if arguments is None and not path.exists() and not path.is_symlink():
            path.write_bytes(raw)
        elif arguments is None and path.is_file() and not path.is_symlink():
            path.write_bytes(raw)
        expected = self.expected(raw, members) if expected is None else expected
        arguments = self.arguments(path, expected) if arguments is None else arguments
        flags = ["-B", "-S"]
        if sys.flags.optimize:
            flags.append("-" + "O" * sys.flags.optimize)
        command = [sys.executable, *flags, str(CLI), *arguments]
        if audit:
            command = [sys.executable, *flags, "-c", AUDIT_LAUNCHER, str(self.cwd), str(path),
                       str(self.cwd / "TEST-private-secret"), str(self.cwd / "TEST-observer.json"), str(CLI),
                       str(observer_port), *arguments]
        environment = os.environ.copy()
        environment.pop("PYTHONOPTIMIZE", None)
        try:
            return subprocess.run(command, cwd=self.cwd, env=environment, capture_output=True,
                                  timeout=20, check=False)
        except subprocess.TimeoutExpired:
            self.fail("graph archive CLI exceeded the bounded 20-second fixture run")

    def accepted(self, raw=None, *, manifest=None, **kwargs):
        members = kwargs.get("members", self.base)
        manifest = self.entries(members)[-1][1] if manifest is None else manifest
        raw = classic_zip(self.entries(members)) if raw is None else raw
        expected = kwargs.get("expected", self.expected(raw, members))
        result = self.invoke(raw, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr)
        self.assertEqual(result.stderr, b"")
        self.assertLessEqual(len(result.stdout), 16 * MIB)
        report = document(result.stdout)
        self.assertEqual(set(report), {"schema_version", "scientific_effect", "scientific_status_authority",
                                      "custody", "graph", "receipt"})
        self.assertIs(type(report["schema_version"]), int)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["scientific_effect"], "NONE")
        self.assertIs(report["scientific_status_authority"], False)
        self.assertEqual(report["custody"], "unknown")
        self.assertEqual(encoded(report["graph"]), encoded(document(members["generated/graph.json"])))
        receipt = report["receipt"]
        self.assertEqual(set(receipt), {"schema_version", "scientific_effect", "scientific_status_authority",
                                       "custody", "verification_scope", "binding", "archive", "graph_member",
                                       "producer_receipt_sha256", "producer_receipt"})
        self.assertIs(type(receipt["schema_version"]), int)
        self.assertEqual(receipt["schema_version"], 1)
        self.assertEqual(receipt["scientific_effect"], "NONE")
        self.assertIs(receipt["scientific_status_authority"], False)
        self.assertEqual(receipt["custody"], "unknown")
        self.assertEqual(receipt["verification_scope"], "declared-native-pins-and-archive-content")
        self.assertEqual(receipt["binding"], expected)
        self.assertEqual(receipt["producer_receipt_sha256"], sha(members["architecture-run-receipt.json"]))
        self.assertEqual(encoded(receipt["producer_receipt"]), encoded(document(members["architecture-run-receipt.json"])))
        archive = receipt["archive"]
        self.assertEqual(set(archive), {"sha256", "bytes", "members"})
        self.assertEqual(archive["sha256"], sha(raw))
        self.assertIs(type(archive["bytes"]), int)
        self.assertEqual(archive["bytes"], len(raw))
        rows = archive["members"]
        self.assertEqual(len(rows), 13)
        self.assertEqual([row["path"] for row in rows], sorted([*members, "ARTIFACT_SHA256SUMS"]))
        for row in rows:
            self.assertEqual(set(row), {"path", "bytes", "sha256"})
            self.assertIs(type(row["bytes"]), int)
            raw_member = manifest if row["path"] == "ARTIFACT_SHA256SUMS" else members[row["path"]]
            self.assertEqual(row, {"path": row["path"], "bytes": len(raw_member), "sha256": sha(raw_member)})
        self.assertIs(type(receipt["graph_member"]["bytes"]), int)
        self.assertEqual(receipt["graph_member"], {"path": "generated/graph.json",
                         "bytes": len(members["generated/graph.json"]), "sha256": sha(members["generated/graph.json"])})
        self.assertEqual(result.stdout, encoded(report), "stdout is not one canonical ASCII/LF unit")
        return report, result

    def refused(self, raw=None, **kwargs):
        result = self.invoke(raw, **kwargs)
        self.assertNotEqual(result.returncode, 0, "invalid archive was accepted")
        self.assertEqual(result.stdout, b"", "refusal published partial or success JSON")
        self.assertEqual(result.stderr, REFUSAL, "refusal exposed paths/member data or variable diagnostics")

    def test_valid_archive_is_one_stable_non_scientific_unit_with_original_receipt(self):
        first, one = self.accepted()
        _, two = self.accepted(path=self.cwd / "TEST-second-input.zip")
        self.assertEqual(one.stdout, two.stdout)
        self.assertNotIn("kernel_verified", first)
        self.assertNotIn("lemma_exists", first)
        self.assertNotIn(str(self.cwd).encode(), one.stdout)
        manifest = self.entries()[-1][1]
        self.assertEqual(next(row for row in first["receipt"]["archive"]["members"]
                              if row["path"] == "ARTIFACT_SHA256SUMS"),
                         {"path": "ARTIFACT_SHA256SUMS", "bytes": len(manifest), "sha256": sha(manifest)})

    def test_other_native_context_attempt_ten_and_declared_artifact_id_are_preserved(self):
        context = {**NATIVE, "checked_commit": "d" * 40, "repository": "TEST-other/TEST-repo",
                   "run_id": "987", "run_attempt": "10", "artifact_id": "7654321"}
        members = self.fixture(context)
        raw = classic_zip(self.entries(members))
        self.accepted(raw, members=members, expected=self.expected(raw, members, context))
        expected = self.expected(classic_zip(self.entries()))
        expected["artifact_id"] = "999999"
        report, _ = self.accepted(expected=expected)
        self.assertEqual(report["receipt"]["binding"]["artifact_id"], "999999")
        self.assertEqual(report["custody"], "unknown", "an artifact label became authenticated custody")

    def test_stored_deflated_descriptors_and_member_or_manifest_order_are_valid(self):
        reversed_rows = b"".join(reversed(self.entries()[-1][1].splitlines(keepends=True)))
        entries = self.entries(manifest=reversed_rows, order=list(reversed(list(self.base))))
        for mode in ("stored", "deflated", "signed_descriptors", "unsigned_descriptors", "dos_regular"):
            with self.subTest(mode=mode):
                options = {}
                for index, (_, raw) in enumerate(entries):
                    if mode == "stored":
                        options[index] = {"method": 0}
                    elif mode == "signed_descriptors":
                        options[index] = {"flags": 8}
                    elif mode == "unsigned_descriptors":
                        compressed = zlib.compressobj(level=6, wbits=-15)
                        payload = compressed.compress(raw) + compressed.flush()
                        options[index] = {"flags": 8, "descriptor": struct.pack(
                            "<III", zlib.crc32(raw) & 0xffffffff, len(payload), len(raw))}
                    elif mode == "dos_regular":
                        options[index] = {"creator": 20, "attributes": 0}
                report, _ = self.accepted(classic_zip(entries, options), manifest=reversed_rows)
                row = next(row for row in report["receipt"]["archive"]["members"]
                           if row["path"] == "ARTIFACT_SHA256SUMS")
                self.assertEqual(row, {"path": "ARTIFACT_SHA256SUMS", "bytes": len(reversed_rows), "sha256": sha(reversed_rows)})

    def test_public_graph_keeps_49_dictionary_ids_55_edges_and_distinct_recorded_axes(self):
        report, _ = self.accepted()
        graph = report["graph"]
        self.assertIs(type(graph["nodes"]), dict)
        self.assertEqual(len(graph["nodes"]), 49)
        self.assertEqual(len(graph["edges"]), 55)
        self.assertEqual(set(graph["dimensions"]), set(graph["nodes"]))
        for identity, axes in graph["dimensions"].items():
            self.assertEqual(set(axes), {"source", "review", "kernel", "computation", "alignment"})
            for axis in ("kernel", "computation", "alignment"):
                self.assertEqual(axes[axis], "not_recorded", identity)
        self.assertTrue(any(axes["source"] == "recorded" for axes in graph["dimensions"].values()))
        self.assertTrue(any(axes["review"] == "recorded" for axes in graph["dimensions"].values()))

    def test_equal_additional_passing_inventory_includes_nonunderscore_unittest_names(self):
        extras = ("test_architecture_graph_artifact_readback.SyntheticControls.test1",
                  "test_architecture_graph_artifact_readback.SyntheticControls.testExtra")
        members = self.fixture(ids=(*REQUIRED_IDS, *extras))
        report, _ = self.accepted(members=members)
        for mode in report["receipt"]["producer_receipt"]["test_modes"]:
            self.assertEqual(mode["test_count"], 91)
            self.assertEqual(mode["test_ids"], sorted([*REQUIRED_IDS, *extras]))

    def test_exact_4096_inventory_is_valid_and_coherent_4097_is_refused(self):
        for count, valid in ((4096, True), (4097, False)):
            with self.subTest(test_count=count):
                extras = tuple("test_architecture_graph_artifact_readback.SyntheticCeiling.test_boundary_"
                               + str(index) for index in range(count - len(REQUIRED_IDS)))
                ids = (*REQUIRED_IDS, *extras)
                self.assertEqual(len(ids), count)
                self.assertEqual(len(set(ids)), count)
                self.assertTrue(set(REQUIRED_IDS).issubset(ids))
                members = self.fixture(ids=ids)
                receipt = document(members["architecture-run-receipt.json"])
                for mode in receipt["test_modes"]:
                    self.assertEqual(mode["test_count"], count)
                    self.assertEqual(mode["test_ids"], sorted(ids))
                    name = "tests-" + mode["mode"] + ".log"
                    self.assertIn(("Ran " + str(count) + " tests in 0.001s\n").encode("ascii"), members[name])
                    self.assertEqual(receipt["evidence_file_sha256"][name], sha(members[name]))
                    self.assertLess(len(members[name]), 16 * MIB)
                self.assertLess(len(members["architecture-run-receipt.json"]), 4 * MIB)
                raw = classic_zip(self.entries(members))
                self.assertLess(len(raw), 64 * MIB)
                expected = self.expected(raw, members)
                self.assertEqual(expected["receipt_sha256"], sha(members["architecture-run-receipt.json"]))
                self.assertEqual(expected["artifact_sha256"], sha(raw))
                if valid:
                    report, _ = self.accepted(raw, members=members, expected=expected)
                    for mode in report["receipt"]["producer_receipt"]["test_modes"]:
                        self.assertEqual(mode["test_count"], count)
                        self.assertEqual(mode["test_ids"], sorted(ids))
                else:
                    self.refused(raw, members=members, expected=expected)

    def test_all_native_flags_are_required_unique_and_opaque_on_argument_refusal(self):
        raw = classic_zip(self.entries())
        path = self.cwd / "TEST-arguments.zip"
        path.write_bytes(raw)
        arguments = self.arguments(path, self.expected(raw))
        for index in range(0, len(arguments), 2):
            with self.subTest(omitted=arguments[index]):
                self.refused(raw, arguments=arguments[:index] + arguments[index + 2:])
            with self.subTest(duplicate=arguments[index]):
                self.refused(raw, arguments=arguments + arguments[index:index + 2])
        for extra in (["--help"], ["--unknown", "PRIVATE_BAD_ARGUMENT"],
                      ["--expected-comm", "a" * 40], ["--output", "PRIVATE_OUTPUT_PATH"]):
            with self.subTest(extra=extra[0]):
                self.refused(raw, arguments=arguments + extra)
        for field, value in (("checked_commit", "A" * 40), ("checked_commit", "a" * 39),
                             ("repository", "owner/repo/extra"), ("repository", "./repo"),
                             ("repository", "owner/.."), ("run_id", "0123"), ("run_id", "0"),
                             ("run_attempt", "1.0"), ("run_attempt", "١"),
                             ("receipt_sha256", "B" * 64), ("artifact_sha256", "c" * 63),
                             ("artifact_id", "0"), ("artifact_id", "01"), ("artifact_id", "1;PRIVATE"),
                             ("artifact_name", "research-architecture-123-2")):
            with self.subTest(field=field, value=value):
                self.refused(raw, expected={**self.expected(raw), field: value})

    def test_native_context_and_both_external_byte_pins_cannot_be_substituted(self):
        raw = classic_zip(self.entries())
        for field, value in (("checked_commit", "d" * 40), ("repository", "TEST-other/TEST-repo"),
                             ("run_id", "124"), ("run_attempt", "2"),
                             ("receipt_sha256", "e" * 64), ("artifact_sha256", "f" * 64)):
            with self.subTest(field=field):
                expected = {**self.expected(raw), field: value}
                expected["artifact_name"] = "research-architecture-" + expected["run_id"] + "-" + expected["run_attempt"]
                self.refused(raw, expected=expected)

    def test_each_closed_member_is_required_and_additional_members_are_refused(self):
        entries = self.entries()
        for index, (name, _) in enumerate(entries):
            with self.subTest(missing=name):
                self.refused(classic_zip(entries[:index] + entries[index + 1:]))
        self.refused(classic_zip(entries + [("TEST-extra.txt", b"PRIVATE_UNSELECTED_EXTRA")]))

    def test_duplicate_case_alias_nul_and_unsafe_archive_member_names_are_refused(self):
        entries = self.entries()
        for name in ("../generated/graph.json", "/generated/graph.json", "C:/generated/graph.json",
                     "generated\\graph.json", "generated//graph.json", "./generated/graph.json",
                     "generated/../graph.json", "generated/graph.json/", "generated/\x00graph.json",
                     "generated/\ngraph.json", "GENERATED/GRAPH.JSON"):
            with self.subTest(name=repr(name)):
                self.refused(classic_zip(entries, {0: {"name": name}}))
        for name in ("generated/graph.json", "GENERATED/GRAPH.JSON"):
            with self.subTest(duplicate=name):
                self.refused(classic_zip(entries + [(name, self.base["generated/graph.json"])]))

    def test_special_attributes_encryption_and_unsupported_flags_or_methods_are_refused(self):
        for option in ({"attributes": (stat.S_IFLNK | 0o777) << 16},
                       {"attributes": (stat.S_IFIFO | 0o600) << 16},
                       {"attributes": (stat.S_IFSOCK | 0o600) << 16},
                       {"attributes": (stat.S_IFCHR | 0o600) << 16},
                       {"attributes": (stat.S_IFDIR | 0o755) << 16},
                       {"creator": 20, "attributes": 0x10},
                       {"flags": 1}, {"flags": 1 << 6}, {"flags": 1 << 13},
                       {"method": 12}, {"method": 99}, {"method": 0, "flags": 2}):
            with self.subTest(option=option):
                self.refused(classic_zip(self.entries(), {0: option}))

    def test_local_central_or_descriptor_contradictions_and_overlap_are_refused(self):
        entries = self.entries()
        for option in ({"local_name": b"TEST-other-name"}, {"local": {"method": 0}},
                       {"local": {"flags": 8}}, {"local": {"crc": 0}},
                       {"local": {"size": 1}}, {"local": {"expanded": 1}},
                       {"offset": 1}, {"flags": 8, "descriptor": struct.pack("<IIII", 0x08074b50, 0, 1, 1)}):
            with self.subTest(option=option):
                self.refused(classic_zip(entries, {0: option}))
        self.refused(classic_zip(entries, {1: {"offset": 0}}))

    def test_truncation_prefix_tail_multidisk_zip64_comments_and_extras_are_refused(self):
        entries = self.entries()
        raw = classic_zip(entries)
        for damaged in (raw[:-1], raw[:-22], raw[:30], raw + b"PRIVATE_TRAILING_BYTES", raw + raw,
                        classic_zip(entries, prefix=b"PRIVATE_PREFIX"),
                        classic_zip(entries, disk=1), classic_zip(entries, zip64=True),
                        classic_zip(entries, comment=b"PRIVATE_ARCHIVE_COMMENT"),
                        classic_zip(entries, {0: {"comment": b"PRIVATE_MEMBER_COMMENT"}}),
                        classic_zip(entries, {0: {"extra": b"\x01\x99\x00\x00"}})):
            with self.subTest(length=len(damaged), tail=damaged[-8:]):
                self.refused(damaged)

    def test_unreferenced_interior_gap_or_hidden_local_record_is_refused_with_coherent_offsets(self):
        entries = self.entries()
        original = classic_zip(entries)
        original_footer = struct.unpack_from("<IHHHHIIH", original, len(original) - 22)
        hidden_archive = classic_zip([("TEST-unreferenced-local-record.txt", b"TEST hidden bytes only")],
                                     {0: {"method": 0}})
        hidden_footer = struct.unpack_from("<IHHHHIIH", hidden_archive, len(hidden_archive) - 22)
        hidden_local = hidden_archive[:hidden_footer[6]]
        self.assertTrue(hidden_local.startswith(b"PK\x03\x04"))
        insertion_index = 6
        for kind, inserted in (("interior_gap", b"TEST unreferenced interior gap bytes"),
                               ("hidden_local_record", hidden_local)):
            with self.subTest(kind=kind):
                raw = classic_zip(entries, {insertion_index: {"before": inserted}})
                self.assertEqual(len(raw), len(original) + len(inserted))
                footer = struct.unpack_from("<IHHHHIIH", raw, len(raw) - 22)
                self.assertEqual(footer[:6], original_footer[:6])
                self.assertEqual(footer[3:5], (13, 13), "TEST changed the referenced inventory")
                self.assertEqual(footer[6], original_footer[6] + len(inserted))
                self.assertEqual(footer[7], 0)
                self.assertEqual(footer[6] + footer[5], len(raw) - 22)
                original_position, position = original_footer[6], footer[6]
                insertion_offset = None
                for index, (name, _) in enumerate(entries):
                    original_record = struct.unpack_from("<IHHHHHHIIIHHHHHII", original, original_position)
                    record = struct.unpack_from("<IHHHHHHIIIHHHHHII", raw, position)
                    self.assertEqual(record[0], 0x02014b50)
                    self.assertEqual(record[:-1], original_record[:-1])
                    self.assertEqual(record[-1], original_record[-1]
                                     + (len(inserted) if index >= insertion_index else 0))
                    record_bytes = 46 + sum(record[10:13])
                    self.assertEqual(raw[position + 46:position + record_bytes],
                                     original[original_position + 46:original_position + record_bytes])
                    self.assertEqual(raw[position + 46:position + 46 + record[10]], name.encode("utf-8"))
                    local_bytes = 30 + record[10] + record[11] + record[8]
                    self.assertEqual(raw[record[-1]:record[-1] + local_bytes],
                                     original[original_record[-1]:original_record[-1] + local_bytes])
                    if index == insertion_index:
                        insertion_offset = original_record[-1]
                    original_position += record_bytes
                    position += record_bytes
                self.assertEqual(position, len(raw) - 22)
                self.assertEqual(original_position, len(original) - 22)
                self.assertIsNotNone(insertion_offset)
                self.assertEqual(raw[insertion_offset:insertion_offset + len(inserted)], inserted)
                expected = self.expected(raw)
                self.assertEqual(expected["artifact_sha256"], sha(raw))
                self.assertEqual(expected["receipt_sha256"], sha(self.base["architecture-run-receipt.json"]))
                self.refused(raw, expected=expected)

    def test_crc_incomplete_deflate_and_unused_compressed_tail_are_refused(self):
        entries = self.entries()
        raw = entries[0][1]
        compressor = zlib.compressobj(level=6, wbits=-15)
        compressed = compressor.compress(raw) + compressor.flush()
        for option in ({"crc": (zlib.crc32(raw) + 1) & 0xffffffff},
                       {"compressed": compressed[:-1]},
                       {"compressed": compressed + b"PRIVATE_UNUSED_COMPRESSED_TAIL"},
                       {"compressed": b"not a deflate stream"}):
            with self.subTest(option=list(option)):
                self.refused(classic_zip(entries, {0: option}))

    def test_raw_zip_at_exact_64_mib_passes_and_one_byte_over_is_refused(self):
        members = copy.deepcopy(self.base)
        console_index = list(members).index("console.log")
        for target, valid in ((64 * MIB, True), (64 * MIB + 1, False)):
            with self.subTest(target=target):
                members["console.log"] = b""
                entries = self.entries(members)
                options = {index: {"method": 0} for index in range(len(entries))}
                options[console_index] = {"method": 8, "compressed": b"\x01\x00\x00\xff\xff"}
                initial = classic_zip(entries, options)
                padding, remainder = divmod(target - len(initial), 5)
                members["console.log"] = b"x" * remainder
                final = b"\x01" + struct.pack("<HH", remainder, 0xffff ^ remainder) + members["console.log"]
                payload = b"\x00\x00\x00\xff\xff" * padding + final
                options[console_index] = {"method": 8, "compressed": payload}
                raw = classic_zip(self.entries(members), options)
                self.assertEqual(len(raw), target, "TEST exact raw-boundary fixture is not exact")
                if valid:
                    self.accepted(raw, members=members)
                else:
                    self.refused(raw, members=members)

    def test_exact_member_limits_accept_bounded_content_and_one_over_refuses(self):
        limits = {"tests-normal.log": 16 * MIB, "tests-optimized.log": 16 * MIB,
                  "runtime-image.json": MIB, "generated/graph.json": 4 * MIB,
                  "architecture-run-receipt.json": 4 * MIB, "console.log": 4 * MIB}
        for name, limit in limits.items():
            for delta in (0, 1):
                with self.subTest(member=name, delta=delta):
                    members = copy.deepcopy(self.base)
                    raw = members[name]
                    if name.startswith("tests-"):
                        members[name] = padded_log(raw, limit + delta)
                    elif name == "console.log":
                        members[name] = b"x" * (limit + delta)
                    else:
                        members[name] = raw + b" " * (limit + delta - len(raw))
                    if name != "architecture-run-receipt.json":
                        self.rebind(members)
                    self.assertEqual(len(members[name]), limit + delta)
                    if delta:
                        self.refused(members=members)
                    else:
                        self.accepted(members=members)

    def test_small_member_budgets_and_exact_128_byte_positive_decimal_are_enforced(self):
        context = {**NATIVE, "run_id": "1" * 127}
        members = self.fixture(context)
        raw = classic_zip(self.entries(members))
        self.assertEqual(len(members["run-id.txt"]), 128)
        self.accepted(raw, members=members, expected=self.expected(raw, members, context))
        for name, limit in (("checked-commit.txt", 128), ("run-id.txt", 128), ("run-attempt.txt", 128),
                            ("container-exit-status.txt", 128), ("exporter.stdout", 65536),
                            ("exporter.stderr", 65536)):
            with self.subTest(member=name):
                members = copy.deepcopy(self.base)
                members[name] = b"x" * (limit + 1)
                self.rebind(members)
                self.refused(members=members)
        self.refused(classic_zip(self.entries(manifest=b"x" * (16384 + 1))))

    def test_actual_deflate_expansion_cannot_exceed_a_small_declared_size(self):
        entries = self.entries()
        console_index = next(index for index, (name, _) in enumerate(entries) if name == "console.log")
        raw = b"x" * (4 * MIB + 1)
        compressor = zlib.compressobj(level=9, wbits=-15)
        payload = compressor.compress(raw) + compressor.flush()
        self.refused(classic_zip(entries, {console_index: {"compressed": payload, "expanded": 1}}))
        members = copy.deepcopy(self.base)
        members["console.log"] = b"x" * (4 * MIB)
        self.accepted(members=members)

    def test_all_twelve_manifest_hashes_and_closed_row_format_are_checked(self):
        manifest = self.entries()[-1][1]
        for name in MEMBERS:
            with self.subTest(member=name):
                bad = manifest.replace(sha(self.base[name]).encode() + b"  " + name.encode(),
                                       b"0" * 64 + b"  " + name.encode())
                self.refused(classic_zip(self.entries(manifest=bad)))
        first = manifest.splitlines(keepends=True)[0]
        for bad in (manifest + first, manifest[:-1], manifest.replace(b"  ", b" *", 1),
                    manifest.replace(b"  ", b" ", 1), b"\n" + manifest, manifest + b"PRIVATE_TRAILING_ROW\n",
                    manifest.replace(first, first.upper(), 1),
                    manifest.replace(first, sha(b"").encode() + b"  ARTIFACT_SHA256SUMS\n", 1)):
            with self.subTest(prefix=bad[:30]):
                self.refused(classic_zip(self.entries(manifest=bad)))

    def test_each_receipt_evidence_hash_is_required_exact_and_independently_recomputed(self):
        for name in EVIDENCE:
            for mutation in ("wrong", "missing"):
                with self.subTest(member=name, mutation=mutation):
                    members = copy.deepcopy(self.base)
                    receipt = document(members["architecture-run-receipt.json"])
                    if mutation == "wrong":
                        receipt["evidence_file_sha256"][name] = "0" * 64
                    else:
                        del receipt["evidence_file_sha256"][name]
                    members["architecture-run-receipt.json"] = encoded(receipt)
                    self.refused(members=members)
        members = copy.deepcopy(self.base)
        self.change_json(members, "architecture-run-receipt.json", lambda receipt:
                         receipt["evidence_file_sha256"].update({"console.log": sha(members["console.log"])}), rebind=False)
        self.refused(members=members)

    def test_original_receipt_external_pin_survives_coherent_manifest_regeneration(self):
        members = copy.deepcopy(self.base)
        original_pin = sha(members["architecture-run-receipt.json"])
        members["architecture-run-receipt.json"] += b" "
        raw = classic_zip(self.entries(members))
        self.accepted(raw, members=members)
        self.refused(raw, members=members, expected={**self.expected(raw, members), "receipt_sha256": original_pin})

    def test_native_text_files_and_original_receipt_context_must_match_expected_native_four(self):
        for name, value in (("checked-commit.txt", b"d" * 40 + b"\n"), ("run-id.txt", b"124\n"),
                            ("run-attempt.txt", b"2\n"), ("checked-commit.txt", b"a" * 40),
                            ("run-id.txt", b"123\n\n"), ("run-attempt.txt", b"01\n"),
                            ("container-exit-status.txt", b"1\n"), ("container-exit-status.txt", b"0.0\n"),
                            ("container-exit-status.txt", b"False\n")):
            with self.subTest(member=name, value=value):
                members = copy.deepcopy(self.base)
                members[name] = value
                self.rebind(members)
                self.refused(members=members)
        for field, value in (("checked_commit", "d" * 40), ("repository", "TEST-other/TEST-repo"),
                             ("run_id", "124"), ("run_attempt", "2")):
            with self.subTest(receipt_field=field):
                members = copy.deepcopy(self.base)
                self.change_json(members, "architecture-run-receipt.json",
                                 lambda receipt: receipt.update({field: value}), rebind=False)
                self.refused(members=members)

    def test_producer_receipt_schema_types_authority_and_closed_native_objects_are_exact(self):
        receipt = document(self.base["architecture-run-receipt.json"])
        for field in receipt:
            with self.subTest(missing=field):
                members = copy.deepcopy(self.base)
                self.change_json(members, "architecture-run-receipt.json", lambda value: value.pop(field), rebind=False)
                self.refused(members=members)
        for field, value in (("schema_version", True), ("schema_version", 1.0),
                             ("scientific_effect", "ACCEPTED"), ("scientific_status_authority", 0),
                             ("scientific_status_authority", True), ("checked_commit", False),
                             ("repository", 0), ("run_id", 123), ("run_attempt", 1.0), ("extra", "PRIVATE_EXTRA")):
            with self.subTest(field=field, value=value):
                members = copy.deepcopy(self.base)
                self.change_json(members, "architecture-run-receipt.json", lambda receipt: receipt.update({field: value}), rebind=False)
                self.refused(members=members)
        members = copy.deepcopy(self.base)
        self.change_json(members, "architecture-run-receipt.json",
                         lambda receipt: receipt["runtime_image"].update({"extra": "PRIVATE_EXTRA"}), rebind=False)
        self.refused(members=members)

    def test_json_duplicates_nonfinite_unicode_and_multiple_documents_are_refused(self):
        for name in ("generated/graph.json", "architecture-run-receipt.json", "runtime-image.json"):
            for mutation in ("duplicate", "duplicate_nested", "nan", "infinity", "overflow", "surrogate", "utf8", "multiple", "truncated"):
                with self.subTest(member=name, mutation=mutation):
                    members = copy.deepcopy(self.base)
                    raw = members[name]
                    if mutation == "duplicate":
                        if name == "runtime-image.json":
                            raw = raw.replace(b'"Id":', b'"Id":"sha256:' + b"f" * 64 + b'","Id":', 1)
                        else:
                            raw = raw.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1', 1)
                    elif mutation == "duplicate_nested":
                        if name == "generated/graph.json":
                            raw = raw.replace(b'"controlling":false', b'"controlling":false,"controlling":false', 1)
                        elif name == "architecture-run-receipt.json":
                            value = b'"id":"sha256:' + b"f" * 64 + b'"'
                            raw = raw.replace(value, value + b"," + value, 1)
                        else:
                            value = b'"TEST_annotation":"synthetic image declaration"'
                            raw = raw.replace(value, value + b"," + value, 1)
                    elif mutation in ("nan", "infinity", "overflow", "surrogate"):
                        token = {"nan": b"NaN", "infinity": b"Infinity", "overflow": b"1e999",
                                 "surrogate": b'"\\ud800"'}[mutation]
                        raw = raw.replace(b'"TEST_annotation":"synthetic image declaration"', b'"TEST_annotation":' + token, 1) if name == "runtime-image.json" else raw.replace(b'"schema_version":1', b'"schema_version":' + token, 1)
                    elif mutation == "utf8":
                        raw += b"\xff"
                    elif mutation == "multiple":
                        raw += b"{}"
                    else:
                        raw = raw[:-2]
                    self.assertNotEqual(raw, members[name], "TEST mutation did not change its target")
                    members[name] = raw
                    if name != "architecture-run-receipt.json":
                        self.rebind(members)
                    self.refused(members=members)

    def test_exact_64_container_depth_is_valid_and_65_is_refused_in_native_annotations(self):
        for levels, valid in ((62, True), (63, False)):
            with self.subTest(container_depth=levels + 2):
                members = copy.deepcopy(self.base)
                nested = "opaque synthetic leaf"
                for _ in range(levels):
                    nested = [nested]
                self.change_json(members, "runtime-image.json", lambda image: image[0].update({"TEST_nested": nested}))
                if valid:
                    self.accepted(members=members)
                else:
                    self.refused(members=members)

    def test_graph_seven_root_fields_and_all_five_source_pins_cannot_be_replaced(self):
        graph = document(self.base["generated/graph.json"])
        for field in graph:
            with self.subTest(missing_root=field):
                members = copy.deepcopy(self.base)
                self.change_json(members, "generated/graph.json", lambda value: value.pop(field))
                self.refused(members=members)
        for field in GRAPH_SOURCE:
            for mutation in ("changed", "missing"):
                with self.subTest(source_field=field, mutation=mutation):
                    members = copy.deepcopy(self.base)
                    changed_source = {**GRAPH_SOURCE}
                    if mutation == "changed":
                        changed_source[field] = "TEST_CHANGED_PIN"
                    else:
                        del changed_source[field]
                    self.change_json(members, "generated/graph.json", lambda value: value.update({"source": changed_source}))
                    self.change_json(members, "architecture-run-receipt.json",
                                     lambda receipt: receipt.update({"graph_source": changed_source}), rebind=False)
                    self.refused(members=members)
        for field, value in (("extra", "PRIVATE_EXTRA"), ("schema_version", 1.0),
                             ("scientific_effect", "ACCEPTED"), ("scientific_status_authority", 0)):
            members = copy.deepcopy(self.base)
            self.change_json(members, "generated/graph.json", lambda graph: graph.update({field: value}))
            self.refused(members=members)

    def test_typed_node_edge_and_dimensions_cannot_change_under_unchanged_source_pins(self):
        graph = document(self.base["generated/graph.json"])
        false_node = next(identity for identity, node in graph["nodes"].items() if node.get("controlling") is False)
        integer_field = next((identity, field) for identity, node in graph["nodes"].items()
                             for field, value in node.items() if type(value) is int)
        for mutation in ("false_to_zero", "integer_to_float", "node_extra", "edge_record", "edge_order",
                         "dimension_missing", "dimension_extra", "dimension_axis_extra", "dimension_kernel",
                         "dimension_source", "dimension_review"):
            with self.subTest(mutation=mutation):
                members = copy.deepcopy(self.base)
                def alter(value):
                    identity = next(iter(value["nodes"]))
                    if mutation == "false_to_zero":
                        value["nodes"][false_node]["controlling"] = 0
                    elif mutation == "integer_to_float":
                        node, field = integer_field
                        value["nodes"][node][field] = float(value["nodes"][node][field])
                    elif mutation == "node_extra":
                        value["nodes"][identity]["TEST_extra"] = "PRIVATE_CHANGED_RECORD"
                    elif mutation == "edge_record":
                        value["edges"][0]["required"] = not value["edges"][0]["required"]
                    elif mutation == "edge_order":
                        value["edges"] = list(reversed(value["edges"]))
                    elif mutation == "dimension_missing":
                        del value["dimensions"][identity]
                    elif mutation == "dimension_extra":
                        value["dimensions"]["TEST.guessed"] = copy.deepcopy(value["dimensions"][identity])
                    elif mutation == "dimension_axis_extra":
                        value["dimensions"][identity]["accepted"] = True
                    else:
                        axis = mutation.removeprefix("dimension_")
                        original = value["dimensions"][identity][axis]
                        value["dimensions"][identity][axis] = "recorded" if original == "not_recorded" else "not_recorded"
                self.change_json(members, "generated/graph.json", alter)
                self.refused(members=members)

    def test_all_89_literal_ids_are_required_even_with_a_coherent_substitute_count(self):
        for missing in REQUIRED_IDS:
            with self.subTest(missing=missing):
                substitute = "test_architecture_graph_artifact_readback.SyntheticControls.test_substitute"
                ids = [identity for identity in REQUIRED_IDS if identity != missing] + [substitute]
                self.refused(members=self.fixture(ids=ids))

    def test_failed_skipped_or_expected_negative_results_never_qualify_as_passing(self):
        extra = "test_architecture_graph_artifact_readback.SyntheticControls.test_extra"
        for mode in ("normal", "optimized"):
            for status in ("FAIL", "ERROR", "skipped 'PRIVATE_SKIP'", "expected failure", "unexpected success"):
                with self.subTest(mode=mode, status=status):
                    members = self.fixture(ids=(*REQUIRED_IDS, extra))
                    name = "tests-" + mode + ".log"
                    line = (extra.rsplit(".", 1)[1] + " (" + extra + ") ... ok").encode()
                    members[name] = members[name].replace(line, line[:-2] + status.encode())
                    self.rebind(members)
                    self.refused(members=members)

    def test_malformed_result_looking_extras_cannot_hide_behind_valid_89_ids_and_footer(self):
        variants = (
            b"test_extra (test_architecture_graph_artifact_readback.Synthetic.test_extra) ...ok",
            b"test_extra (test_architecture_graph_artifact_readback.Synthetic.test_extra)... ok",
            b"test_extra (test_architecture_graph_artifact_readback.Synthetic.test_extra) ok",
            b" test_extra (test_architecture_graph_artifact_readback.Synthetic.test_extra) ... ok",
            b"wrong_short (test_architecture_graph_artifact_readback.Synthetic.test_extra) ...ok",
            b"test_extra (foreign_module.Synthetic.test_extra) ...ok",
            b"test1 (test_architecture_graph_artifact_readback.Synthetic.test1) ...ok",
            b"TEST_extra (test_architecture_graph_artifact_readback.Synthetic.TEST_extra) ... ok",
        )
        for line in variants:
            with self.subTest(line=line):
                members = copy.deepcopy(self.base)
                for mode in ("normal", "optimized"):
                    name = "tests-" + mode + ".log"
                    members[name] = members[name].replace(b"\n----------------------------------------------------------------------", b"\n" + line + b"\n----------------------------------------------------------------------")
                self.rebind(members)
                self.refused(members=members)

    def test_log_duplicates_multiple_runs_footer_and_terminal_success_are_exact(self):
        normal = self.base["tests-normal.log"]
        first = normal.splitlines()[1] + b"\n"
        for raw in (normal.replace(first, first + first, 1), normal + normal,
                    normal.replace(b"Ran 89 tests", b"Ran 88 tests"),
                    normal.replace(b"Ran 89 tests", b"Ran 0 tests"),
                    normal.replace(b"0.001s", b"NaNs"), normal.replace(b"0.001s", b"-1.0s"),
                    normal.replace(b"\nOK\n", b"\nFAILED\n"), normal.replace(b"\nOK\n", b"\n"),
                    normal + b"PRIVATE_NONEMPTY_AFTER_OK\n", normal.replace(b"\n", b"\r\n"),
                    normal + b"\x00", b"Ran 0 tests in 0.001s\n\nOK\n"):
            with self.subTest(length=len(raw)):
                members = copy.deepcopy(self.base)
                members["tests-normal.log"] = raw
                self.rebind(members)
                self.refused(members=members)

    def test_both_complete_suites_and_receipt_mode_counts_and_id_lists_must_agree(self):
        extra_one = "test_architecture_graph_artifact_readback.Synthetic.test_one"
        extra_two = "test_architecture_graph_artifact_readback.Synthetic.test_two"
        members = self.fixture(ids=(*REQUIRED_IDS, extra_one))
        members["tests-optimized.log"] = synthetic_log((*REQUIRED_IDS, extra_two))
        self.change_json(members, "architecture-run-receipt.json", lambda receipt:
                         receipt["test_modes"][1].update({"test_ids": sorted([*REQUIRED_IDS, extra_two])}))
        self.refused(members=members)
        for mutation in ("mode_order", "count_bool", "count_float", "count_changed", "id_missing",
                         "id_duplicate", "id_order", "mode_extra", "mode_missing"):
            with self.subTest(mutation=mutation):
                members = copy.deepcopy(self.base)
                def alter(receipt):
                    modes = receipt["test_modes"]
                    if mutation == "mode_order": modes.reverse()
                    elif mutation == "count_bool": modes[0]["test_count"] = True
                    elif mutation == "count_float": modes[0]["test_count"] = 89.0
                    elif mutation == "count_changed": modes[0]["test_count"] = 88
                    elif mutation == "id_missing": modes[0]["test_ids"].pop()
                    elif mutation == "id_duplicate": modes[0]["test_ids"].append(modes[0]["test_ids"][0])
                    elif mutation == "id_order": modes[0]["test_ids"].reverse()
                    elif mutation == "mode_extra": modes[0]["extra"] = "PRIVATE_MODE_EXTRA"
                    else: modes.pop()
                self.change_json(members, "architecture-run-receipt.json", alter, rebind=False)
                self.refused(members=members)

    def test_runtime_inspect_and_original_receipt_image_identity_are_consistent(self):
        for field, value in (("Id", "sha256:" + "F" * 64), ("Id", False), ("RepoDigests", []),
                             ("RepoDigests", [IMAGE_DIGEST, IMAGE_DIGEST]), ("RepoDigests", ["python@sha256:" + "e" * 64]),
                             ("Os", "darwin"), ("Architecture", "arm64")):
            with self.subTest(field=field):
                members = copy.deepcopy(self.base)
                self.change_json(members, "runtime-image.json", lambda image: image[0].update({field: value}))
                self.refused(members=members)
        for value in ({}, [], [document(self.base["runtime-image.json"])[0]] * 2):
            members = copy.deepcopy(self.base)
            members["runtime-image.json"] = encoded(value)
            self.rebind(members)
            self.refused(members=members)
        members = copy.deepcopy(self.base)
        self.change_json(members, "architecture-run-receipt.json",
                         lambda receipt: receipt["runtime_image"].update({"id": "sha256:" + "e" * 64}), rebind=False)
        self.refused(members=members)
        members = copy.deepcopy(self.base)
        self.change_json(members, "runtime-image.json", lambda image: image[0].update({"Id": "sha256:" + "d" * 64}))
        self.change_json(members, "architecture-run-receipt.json",
                         lambda receipt: receipt["runtime_image"].update({"id": "sha256:" + "d" * 64}), rebind=False)
        self.accepted(members=members)

    def test_fixed_exporter_diagnostics_cannot_claim_a_different_output_or_leak_on_refusal(self):
        for name, raw in (("exporter.stdout", b"/output/PRIVATE_WRONG_GRAPH.json\n"),
                          ("exporter.stdout", b"/output/generated/graph.json"),
                          ("exporter.stdout", b"/output/generated/graph.json\nPRIVATE_EXTRA\n"),
                          ("exporter.stderr", b"PRIVATE_EXPORTER_DIAGNOSTIC")):
            with self.subTest(member=name):
                members = copy.deepcopy(self.base)
                members[name] = raw
                self.refused(members=members)

    def test_input_is_one_regular_nofollow_file_and_fifo_never_waits_for_a_writer(self):
        raw = classic_zip(self.entries())
        original = self.cwd / "TEST-original.zip"
        original.write_bytes(raw)
        leaf = self.cwd / "TEST-leaf-link.zip"
        leaf.symlink_to(original)
        real_parent = self.cwd / "TEST-real-parent"
        real_parent.mkdir()
        (real_parent / "TEST-input.zip").write_bytes(raw)
        linked_parent = self.cwd / "TEST-linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        fifo = self.cwd / "TEST-fifo.zip"
        os.mkfifo(fifo)
        for path in (leaf, linked_parent / "TEST-input.zip", fifo, real_parent,
                     self.cwd / "TEST-missing.zip", real_parent / ".." / "TEST-original.zip"):
            with self.subTest(path=path.name):
                self.refused(raw, arguments=self.arguments(path, self.expected(raw)))
        relative = self.arguments(Path("TEST-original.zip"), self.expected(raw))
        self.accepted(raw, arguments=relative)

    def test_captured_paths_code_urls_and_annotations_are_inert_with_calibrated_observers(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(handler):
                requests.append(handler.path)
                handler.send_response(200)
                handler.end_headers()
                handler.wfile.write(b"TEST observer live")
            def log_message(handler, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = "http://127.0.0.1:" + str(server.server_port) + "/PRIVATE_CAPTURED_URL"
        with urlopen(url + "-calibration", timeout=5) as response:
            self.assertEqual(response.read(), b"TEST observer live")
        self.assertEqual(requests, ["/PRIVATE_CAPTURED_URL-calibration"], "HTTP observer was not live")
        requests.clear()
        secret = self.cwd / "TEST-private-secret"
        secret.write_bytes(b"PRIVATE_SECRET_CANARY_MUST_NOT_BE_READ")
        marker = self.cwd / "TEST-captured-code-executed"
        members = copy.deepcopy(self.base)
        code = ("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('PRIVATE_EXECUTED')\n"
                "from urllib.request import urlopen\nurlopen(" + repr(url) + ").read()\n")
        members["console.log"] = (code + str(secret) + "\n").encode()
        self.change_json(members, "runtime-image.json", lambda image: image[0].update(
            {"TEST_captured_code": code, "TEST_private_path": str(secret), "TEST_url": url}))
        archive = self.cwd / "TEST-input.zip"
        raw = classic_zip(self.entries(members))
        archive.write_bytes(raw)
        before = set(self.cwd.iterdir())
        report, result = self.accepted(raw, members=members, audit=True, observer_port=server.server_port)
        observer_path = self.cwd / "TEST-observer.json"
        observer = document(observer_path.read_bytes())
        self.assertEqual(set(observer), {"calibration", "attempts", "optimize"})
        for label in ("read", "write", "mkdir", "remove", "rename", "rmdir", "list", "network"):
            self.assertGreater(observer["calibration"][label], 0, "inactive observer: " + label)
        self.assertEqual(observer["attempts"], dict.fromkeys(observer["attempts"], 0))
        self.assertEqual(observer["optimize"], sys.flags.optimize)
        self.assertEqual(requests, [], "captured URL was fetched")
        self.assertEqual(secret.read_bytes(), b"PRIVATE_SECRET_CANARY_MUST_NOT_BE_READ")
        self.assertFalse(marker.exists())
        self.assertEqual(set(self.cwd.iterdir()), before | {observer_path})
        for private in (b"PRIVATE_SECRET", b"PRIVATE_CAPTURED_URL", b"PRIVATE_EXECUTED", str(secret).encode()):
            self.assertNotIn(private, result.stdout + result.stderr)
        self.assertEqual(report["custody"], "unknown")


if __name__ == "__main__":
    unittest.main()
