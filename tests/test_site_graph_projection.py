"""Controls for an admitted native graph projected for a future site build.

Fixtures and their result logs are SYNTHETIC_ONLY. They cannot authenticate a
native artifact or establish that the producer tests really executed.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from importlib import import_module
from io import BytesIO
import hashlib
import json
from pathlib import Path
import unittest
from zipfile import ZIP_STORED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
GRAPH_SOURCE = {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "7858329974e28be79f29b22644370084ff43da4f",
    "captured_at": "2026-10-03T15:07:03Z",
    "graph_sha256": "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09",
    "gate_sha256": "a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8",
}
IMAGE = "python@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48"
# Independent literal expectation copied from the pinned reviewed contract.
# Never learn this floor from a receipt or production-module REQUIRED_IDS.
CONTROL_IDS = tuple("""
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
REVIEW_KEYS = ("review_source", "review_issue", "review_basis", "review_provider",
               "review_providers", "review_disposition")
EVIDENCE = ("generated/graph.json", "checked-commit.txt", "run-id.txt",
            "run-attempt.txt", "container-exit-status.txt", "tests-normal.log",
            "tests-optimized.log", "runtime-image.json")
CONSOLE_MARKER = b"SYNTHETIC_PRIVATE_DIAGNOSTIC_36b7_DO_NOT_PROJECT\n"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pretty(value):
    # These original bytes deliberately differ from compact projection JSON.
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n \n").encode("utf-8")


def zip_bytes(members):
    target = BytesIO()
    with ZipFile(target, "w", compression=ZIP_STORED, allowZip64=False) as archive:
        for name in sorted(members):
            info = ZipInfo(name, date_time=(2026, 10, 22, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            archive.writestr(info, members[name])
    return target.getvalue()


def sample(*, graph_edit=None, receipt_edit=None, member_edit=None,
           repository="synthetic-owner/synthetic-producer", run="91001",
           attempt="10", checked_commit="c" * 40):
    original_source = (ROOT / "docs/site/dependency-source/GRAPH.json").read_bytes()
    if (len(original_source) != 38753
            or digest(original_source) != GRAPH_SOURCE["graph_sha256"]):
        raise AssertionError("the independently pinned public source changed")
    source = json.loads(original_source)
    graph = {
        "schema_version": 1, "scientific_effect": "NONE",
        "scientific_status_authority": False, "source": dict(GRAPH_SOURCE),
        "nodes": source["nodes"], "edges": source["edges"],
        "dimensions": {
            name: {
                "source": "recorded" if node.get("source") else "not_recorded",
                "review": "recorded" if any(node.get(key) for key in REVIEW_KEYS) else "not_recorded",
                "kernel": "not_recorded", "computation": "not_recorded",
                "alignment": "not_recorded",
            } for name, node in source["nodes"].items()
        },
    }
    if graph_edit:
        graph_edit(graph)
    ids = sorted(CONTROL_IDS)
    log = ("SYNTHETIC_ONLY fixture input; no runtime success asserted\n"
           + "".join(item.rsplit(".", 1)[1] + " (" + item + ") ... ok\n" for item in ids)
           + "\nRan 89 tests in 0.002s\n\nOK\n").encode()
    members = {
        "checked-commit.txt": (checked_commit + "\n").encode(),
        "run-id.txt": (run + "\n").encode(),
        "run-attempt.txt": (attempt + "\n").encode(),
        "container-exit-status.txt": b"0\n",
        "exporter.stdout": b"/output/generated/graph.json\n",
        "exporter.stderr": b"",
        "generated/graph.json": pretty(graph),
        "runtime-image.json": pretty([{
            "Id": "sha256:" + "6" * 64, "RepoDigests": [IMAGE],
            "Os": "linux", "Architecture": "amd64",
        }]),
        "tests-normal.log": log, "tests-optimized.log": log,
        "console.log": CONSOLE_MARKER,
    }
    if member_edit:
        member_edit(members)
    receipt = {
        "schema_version": 1, "scientific_effect": "NONE",
        "scientific_status_authority": False, "checked_commit": checked_commit,
        "repository": repository, "run_id": run, "run_attempt": attempt,
        "runtime_image": {"id": "sha256:" + "6" * 64, "repo_digest": IMAGE},
        "graph_source": dict(GRAPH_SOURCE),
        "test_modes": [{"mode": mode, "test_count": 89, "test_ids": ids}
                       for mode in ("normal", "optimized")],
        "evidence_file_sha256": {name: digest(members[name]) for name in EVIDENCE},
    }
    if receipt_edit:
        receipt_edit(receipt)
    members["architecture-run-receipt.json"] = pretty(receipt)
    members["ARTIFACT_SHA256SUMS"] = b"".join(
        (digest(members[name]) + "  " + name + "\n").encode()
        for name in sorted(members)
    )
    raw = zip_bytes(members)
    pins = {
        "checked_commit": checked_commit, "repository": repository,
        "run_id": run, "run_attempt": attempt,
        "receipt_sha256": digest(members["architecture-run-receipt.json"]),
        "artifact_id": "92001", "artifact_sha256": digest(raw),
        "artifact_name": "research-architecture-" + run + "-" + attempt,
    }
    return {"archive": raw, "pins": pins, "source": original_source, "members": members}


def boundary_sample():
    """Independent contract serialization arithmetic; never ask the projector for size."""
    from architecture.graph_artifact_readback import read_graph_archive
    value = sample()
    value["pins"]["artifact_id"] = "1"
    admitted = read_graph_archive(value["archive"], value["pins"], value["source"])
    receipt = admitted["receipt"]
    expected_document = {
        "schema_version": 1, "scientific_effect": "NONE",
        "scientific_status_authority": False, "custody": "unknown",
        "graph": {"identity": {"kind": "native-artifact-member",
                                "binding": receipt["binding"], "member": receipt["graph_member"]},
                  "data": admitted["graph"]},
        "admission": receipt,
    }
    base = len((json.dumps(expected_document, sort_keys=True, ensure_ascii=True,
                            separators=(",", ":"), allow_nan=False) + "\n").encode("ascii"))
    limit = 16 * 1024 * 1024
    # artifact_id occurs exactly twice in the independent expected schema.
    # Decimal ASCII digits require no JSON escapes. Nothing in ZIP/source changes.
    # Distinct finite inert arithmetic input pins P(n)=64_304+2*n for this sample.
    # These literals are independent of the projector implementation/cap constant.
    if base != 64_306:
        raise AssertionError("frozen independent fixture size changed")
    digits = 8_356_456
    if digits != 1 + (limit - base) // 2:
        raise AssertionError("independent boundary arithmetic changed")
    if digits <= 1:
        raise AssertionError("fixture has no capacity headroom")
    value["pins"]["artifact_id"] = "1" + "0" * (digits - 1)
    expected_bytes = base + 2 * (digits - 1)
    return value, expected_bytes, limit

class SameArtifactSiteContract(unittest.TestCase):
    def setUp(self):
        # A tests-only hosted cut gets an explicit missing-implementation
        # assertion failure, rather than a discovery error or zero tests.
        self.assertTrue((ROOT / "architecture/site_graph_projection.py").is_file(),
                        "same-artifact projection implementation is missing")
        self.api = import_module("architecture.site_graph_projection")

    def call(self, value):
        return self.api.project_site_graph(value["archive"], value["pins"], value["source"])

    def test_original_graph_and_producer_receipt_remain_exact_byte_strings(self):
        # Catches replacing original member bytes with reserialized JSON.
        value = sample()
        result = self.call(value)
        self.assertEqual(result.original_graph_bytes, value["members"]["generated/graph.json"])
        self.assertEqual(result.original_receipt_bytes, value["members"]["architecture-run-receipt.json"])
        self.assertTrue(result.original_graph_bytes.endswith(b"\n \n"))
        self.assertTrue(result.original_receipt_bytes.endswith(b"\n \n"))
        compact = (json.dumps(json.loads(result.original_graph_bytes), sort_keys=True,
                              separators=(",", ":")) + "\n").encode()
        self.assertNotEqual(result.original_graph_bytes, compact)

    def test_projection_links_the_same_archive_and_complete_member_inventory(self):
        # Catches substituting unrelated source, artifact, member or receipt.
        value = sample()
        result = self.call(value)
        doc = json.loads(result.projection_bytes)
        admission = doc["admission"]
        self.assertEqual(admission["binding"], value["pins"])
        self.assertEqual(admission["archive"], {
            "sha256": digest(value["archive"]), "bytes": len(value["archive"]),
            "members": [{"path": name, "bytes": len(raw), "sha256": digest(raw)}
                        for name, raw in sorted(value["members"].items())],
        })
        graph_member = {"path": "generated/graph.json",
                        "bytes": len(result.original_graph_bytes),
                        "sha256": digest(result.original_graph_bytes)}
        self.assertEqual(doc["graph"]["identity"], {
            "kind": "native-artifact-member", "binding": value["pins"], "member": graph_member,
        })
        self.assertEqual(admission["graph_member"], graph_member)
        self.assertEqual(admission["producer_receipt_sha256"], digest(result.original_receipt_bytes))
        self.assertEqual(admission["producer_receipt"], json.loads(result.original_receipt_bytes))
        self.assertEqual(len(admission["archive"]["members"]), 13)

    def test_all_records_dimensions_and_ordered_edges_are_preserved(self):
        # Catches flattening typed records, losing review qualifiers or reordering edges.
        value = sample()
        doc = json.loads(self.call(value).projection_bytes)
        original = json.loads(value["members"]["generated/graph.json"])
        self.assertEqual(doc["graph"]["data"], original)
        self.assertEqual(len(original["nodes"]), 49)
        self.assertEqual(len(original["edges"]), 55)
        self.assertEqual(doc["graph"]["data"]["edges"], original["edges"])
        self.assertEqual(doc["graph"]["data"]["dimensions"], original["dimensions"])

    def test_declared_native_pins_do_not_authenticate_custody_or_promote_science(self):
        # Catches an accepted badge or scientific claim manufactured from engineering data.
        doc = json.loads(self.call(sample()).projection_bytes)
        for record in (doc, doc["admission"], doc["graph"]["data"]):
            self.assertEqual(record["scientific_effect"], "NONE")
            self.assertIs(record["scientific_status_authority"], False)
        self.assertEqual(doc["custody"], "unknown")
        self.assertEqual(doc["admission"]["custody"], "unknown")
        self.assertEqual(doc["admission"]["verification_scope"],
                         "declared-native-pins-and-archive-content")

    def test_another_producer_and_declared_artifact_label_do_not_rewrite_math_source(self):
        # Catches hardcoded producer identity or conflating producer/build/Math commits.
        value = sample(repository="other-owner/other-producer", run="91002",
                       attempt="12", checked_commit="d" * 40)
        value["pins"]["artifact_id"] = "92002"
        doc = json.loads(self.call(value).projection_bytes)
        self.assertEqual(doc["admission"]["binding"], value["pins"])
        self.assertEqual(doc["graph"]["data"]["source"], GRAPH_SOURCE)
        self.assertNotEqual(value["pins"]["checked_commit"], GRAPH_SOURCE["commit"])
        self.assertEqual(doc["custody"], "unknown")

    def test_each_comparable_native_identity_mismatch_is_refused(self):
        # Catches projecting before the complete reader checks comparable native context.
        value = sample()
        replacements = {
            "checked_commit": "d" * 40, "repository": "other-owner/other-producer",
            "run_id": "91002", "run_attempt": "11", "receipt_sha256": "7" * 64,
            "artifact_sha256": "8" * 64, "artifact_name": "research-architecture-91001-11",
        }
        for key, changed in replacements.items():
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.api.project_site_graph(value["archive"],
                                            {**value["pins"], key: changed}, value["source"])

    def test_missing_extra_or_coerced_native_fields_are_refused(self):
        # Catches partial identity admission or coercing numbers/null into trusted strings.
        value = sample()
        for key in value["pins"]:
            missing = {k: v for k, v in value["pins"].items() if k != key}
            for altered in (missing, {**value["pins"], key: None}):
                with self.subTest(field=key), self.assertRaises(ValueError):
                    self.api.project_site_graph(value["archive"], altered, value["source"])
        with self.assertRaises(ValueError):
            self.api.project_site_graph(value["archive"],
                                        {**value["pins"], "custody": "accepted"}, value["source"])

    def test_self_consistent_changed_graph_is_refused(self):
        # Catches trusting regenerated internal hashes instead of the fixed source records.
        def change_node(graph):
            graph["nodes"][next(iter(graph["nodes"]))]["classification"] = "forged-accepted"
        def change_edge(graph):
            graph["edges"][0]["relation"] = "forged-relation"
        def change_dimension(graph):
            graph["dimensions"][next(iter(graph["nodes"]))]["kernel"] = "verified"
        for edit in (change_node, change_edge, change_dimension):
            with self.subTest(edit=edit.__name__), self.assertRaises(ValueError):
                self.call(sample(graph_edit=edit))

    def test_non_scientific_graph_flags_are_enforced(self):
        # Catches allowing authoritative status to enter through an admitted graph.
        for key, changed in (("schema_version", True), ("scientific_effect", "ACCEPTED"),
                             ("scientific_status_authority", True)):
            value = sample(graph_edit=lambda graph, key=key, changed=changed:
                           graph.__setitem__(key, changed))
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.call(value)

    def test_receipt_semantics_cannot_be_authorized_by_rehashing_it(self):
        # Catches preserving a receipt without verifying its native/scientific meaning.
        for key, changed in (("run_id", "91002"), ("scientific_status_authority", True)):
            value = sample(receipt_edit=lambda receipt, key=key, changed=changed:
                           receipt.__setitem__(key, changed))
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.call(value)

    def test_raw_receipt_pin_cannot_be_substituted_with_canonical_json_hash(self):
        # Catches validating a reconstructed receipt instead of the original member.
        value = sample()
        parsed = json.loads(value["members"]["architecture-run-receipt.json"])
        canonical = (json.dumps(parsed, sort_keys=True, separators=(",", ":")) + "\n").encode()
        wrong_pin = digest(canonical)
        self.assertNotEqual(wrong_pin, value["pins"]["receipt_sha256"])
        with self.assertRaises(ValueError):
            self.api.project_site_graph(value["archive"],
                                        {**value["pins"], "receipt_sha256": wrong_pin}, value["source"])

    def test_failed_or_incomplete_producer_results_cannot_be_projected(self):
        # Catches skipping native status or complete-log validation.
        bad_status = sample(member_edit=lambda members:
                            members.__setitem__("container-exit-status.txt", b"1\n"))
        with self.assertRaises(ValueError):
            self.call(bad_status)
        for mode in ("normal", "optimized"):
            def skipped(members, mode=mode):
                name = "tests-" + mode + ".log"
                members[name] = members[name].replace(b" ... ok\n", b" ... skipped 'synthetic'\n", 1)
            for edit in (skipped, lambda members, mode=mode:
                         members.__setitem__("tests-" + mode + ".log", b"partial\n")):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    self.call(sample(member_edit=edit))

    def test_fixed_independent_source_and_closed_archive_inventory_are_required(self):
        # Catches reading a named member directly before complete archive admission.
        value = sample()
        for changed_source in (b"{}", value["source"] + b"\n"):
            with self.subTest(source_bytes=len(changed_source)), self.assertRaises(ValueError):
                self.api.project_site_graph(value["archive"], value["pins"], changed_source)
        extra = {**value["members"], "unexpected.txt": b"synthetic unexpected member"}
        for changed_archive in (zip_bytes(extra), value["archive"] + b"unreferenced-tail"):
            pins = {**value["pins"], "artifact_sha256": digest(changed_archive)}
            with self.subTest(archive_bytes=len(changed_archive)), self.assertRaises(ValueError):
                self.api.project_site_graph(changed_archive, pins, value["source"])

    def test_public_projection_omits_diagnostics_and_does_not_fabricate_git_identity(self):
        # Catches private console leakage or a generated member mislabelled as a Git blob.
        value = sample()
        result = self.call(value)
        doc = json.loads(result.projection_bytes)
        self.assertNotIn(CONSOLE_MARKER.strip(), result.projection_bytes)
        self.assertNotIn(value["archive"], result.projection_bytes)
        self.assertEqual(set(doc), {"schema_version", "scientific_effect",
                                   "scientific_status_authority", "custody", "graph", "admission"})
        self.assertEqual(set(doc["graph"]["identity"]), {"kind", "binding", "member"})
        self.assertNotIn("blob", doc["graph"]["identity"])
        self.assertEqual(doc["admission"]["archive"]["members"][0]["path"], "ARTIFACT_SHA256SUMS")

    def test_output_is_deterministic_ascii_and_contains_one_complete_document(self):
        # Catches timestamps, unstable ordering, multiple documents or oversized output.
        value = sample()
        one = self.call(value).projection_bytes
        self.assertEqual(one, self.call(value).projection_bytes)
        self.assertTrue(one.endswith(b"\n"))
        self.assertNotIn(b"\n", one[:-1])
        self.assertLessEqual(len(one), 16 * 1024 * 1024)
        self.assertEqual(json.loads(one.decode("ascii"))["schema_version"], 1)

    def test_result_and_native_binding_remain_immutable_after_return(self):
        # Catches retaining mutable caller dictionaries or mutable parsed graph objects.
        value = sample()
        pins_before = dict(value["pins"])
        result = self.call(value)
        self.assertEqual(value["pins"], pins_before)
        for field in ("original_graph_bytes", "original_receipt_bytes", "projection_bytes"):
            self.assertIs(type(getattr(result, field)), bytes)
            with self.subTest(field=field), self.assertRaises(FrozenInstanceError):
                setattr(result, field, b"changed")
        value["pins"]["artifact_id"] = "99999"
        self.assertEqual(json.loads(result.projection_bytes)["admission"]["binding"], pins_before)

    def test_real_admitted_projection_reaches_actual_capacity_or_closest_below(self):
        # Catches reject-all, dropped provenance, wrong multiplicity or an undersized cap.
        from architecture.graph_artifact_readback import read_graph_archive
        value, expected_bytes, limit = boundary_sample()
        admitted = read_graph_archive(value["archive"], value["pins"], value["source"])
        self.assertEqual(len(admitted["receipt"]["binding"]["artifact_id"]),
                         len(value["pins"]["artifact_id"]))
        self.assertIn(limit - expected_bytes, (0, 1))
        result = self.call(value)
        self.assertEqual(len(result.projection_bytes), expected_bytes)
        self.assertLessEqual(len(result.projection_bytes), limit)
        self.assertEqual(result.original_graph_bytes, value["members"]["generated/graph.json"])
        self.assertEqual(result.original_receipt_bytes, value["members"]["architecture-run-receipt.json"])

    def test_real_reader_admits_next_representable_projection_above_actual_capacity(self):
        # Catches removal of the real 16MiB refusal; no patched small cap is used.
        from architecture.graph_artifact_readback import read_graph_archive
        value, below_bytes, limit = boundary_sample()
        value["pins"]["artifact_id"] += "0"
        admitted = read_graph_archive(value["archive"], value["pins"], value["source"])
        self.assertEqual(len(admitted["receipt"]["binding"]["artifact_id"]),
                         len(value["pins"]["artifact_id"]))
        self.assertGreater(below_bytes + 2, limit)
        with self.assertRaisesRegex(ValueError, "projection capacity refused"):
            self.call(value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
