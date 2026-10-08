"""External controls for the host's staged architecture evidence acceptance.

These synthetic TEST logs and identities make no execution or custody claim.
The graph fixture uses the dated, bytes-pinned source as inert JSON data; no gate
or exporter is imported or executed to create it. Run this module only in the
hosted ephemeral runner. The receipt CLI must emit stdout without changing the
staged evidence, leaving separate host redirection responsible for retention.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "tools/architecture_run_receipt.py"
SOURCE_GRAPH = ROOT / "docs/site/dependency-source/GRAPH.json"
FILE_LIMIT = 16 * 1024 * 1024
IMAGE_ID = "sha256:44d8f4434bcb025b8cb5c3151be689a906223e47cc3146971feccda936d2e82b"
IMAGE_DIGEST = "python@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48"
GRAPH_SOURCE = {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "7858329974e28be79f29b22644370084ff43da4f",
    "captured_at": "2026-10-03T15:07:03Z",
    "graph_sha256": "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09",
    "gate_sha256": "a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8",
}
MEMBERS = (
    "generated/graph.json", "checked-commit.txt", "run-id.txt", "run-attempt.txt",
    "container-exit-status.txt", "tests-normal.log", "tests-optimized.log",
    "runtime-image.json",
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
REQUIRED_IDS = tuple(sorted(
    owner + "." + method
    for owner, methods in REQUIRED_METHODS.items()
    for method in methods
))
SANDBOX_IDS = tuple(
    "test_architecture_sandbox.ArchitectureSandbox." + method
    for method in REQUIRED_METHODS["test_architecture_sandbox.ArchitectureSandbox"]
)


def synthetic_log(test_ids=REQUIRED_IDS, *, statuses=None, reported_count=None, summary="OK"):
    statuses = {} if statuses is None else statuses
    lines = [
        identity.rsplit(".", 1)[1] + " (" + identity + ") ... " + statuses.get(identity, "ok")
        for identity in test_ids
    ]
    count = len(test_ids) if reported_count is None else reported_count
    return "\n".join(lines) + "\n\n" + "-" * 70 + "\nRan " + str(count) + " tests in 0.001s\n\n" + summary + "\n"


class ArchitectureReceiptContract(unittest.TestCase):
    def setUp(self):
        # Fail RED before large-file or special-file fixtures are staged. The
        # same assertion is repeated at every actual subprocess launch below.
        self.assertTrue(RECEIPT.is_file(), "required host receipt CLI tools/architecture_run_receipt.py is absent")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.evidence = self.root / "TEST-staged-evidence"
        (self.evidence / "generated").mkdir(parents=True)
        self.expected = {
            "commit": "a" * 40, "repository": "d6g8k5htny-coder/main",
            "run-id": "123", "run-attempt": "1",
        }
        raw_source = SOURCE_GRAPH.read_bytes()
        self.assertEqual(hashlib.sha256(raw_source).hexdigest(), GRAPH_SOURCE["graph_sha256"])
        source = json.loads(raw_source)
        review_fields = (
            "review_source", "review_issue", "review_basis", "review_provider",
            "review_providers", "review_disposition",
        )
        self.graph = {
            "schema_version": 1, "source": copy.deepcopy(GRAPH_SOURCE),
            "nodes": source["nodes"], "edges": source["edges"],
            "dimensions": {
                identity: {
                    "source": "recorded" if node.get("source") else "not_recorded",
                    "review": "recorded" if any(node.get(field) for field in review_fields) else "not_recorded",
                    "kernel": "not_recorded", "computation": "not_recorded", "alignment": "not_recorded",
                }
                for identity, node in source["nodes"].items()
            },
            "scientific_effect": "NONE", "scientific_status_authority": False,
        }
        self.image = [{
            "Id": IMAGE_ID, "RepoDigests": [IMAGE_DIGEST], "Os": "linux", "Architecture": "amd64",
            "Comment": "TEST synthetic inspect record; no Docker invocation",
        }]
        self.write_json("generated/graph.json", self.graph)
        self.write_json("runtime-image.json", self.image)
        for name, value in (
            ("checked-commit.txt", "a" * 40), ("run-id.txt", "123"),
            ("run-attempt.txt", "1"), ("container-exit-status.txt", "0"),
        ):
            self.path(name).write_text(value + "\n", encoding="utf-8")
        for mode in ("normal", "optimized"):
            self.write_log(mode)

    def path(self, name):
        return self.evidence / name

    def write_json(self, name, document):
        self.path(name).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def write_log(self, mode, test_ids=REQUIRED_IDS, **kwargs):
        self.path("tests-" + mode + ".log").write_text(synthetic_log(test_ids, **kwargs), encoding="utf-8")

    def member_bytes(self):
        return {name: self.path(name).read_bytes() for name in MEMBERS}

    def run_receipt(self, *, evidence=None, expected=None, arguments=None):
        self.assertTrue(RECEIPT.is_file(), "required host receipt CLI tools/architecture_run_receipt.py is absent")
        evidence = self.evidence if evidence is None else evidence
        identity = self.expected if expected is None else expected
        if arguments is None:
            arguments = ["--evidence-dir", str(evidence)]
            for name in ("commit", "repository", "run-id", "run-attempt"):
                arguments.extend(["--expected-" + name, identity[name]])
        flags = ["-B", "-S"]
        if sys.flags.optimize:
            flags.append("-" + "O" * sys.flags.optimize)
        env = os.environ.copy()
        env.pop("PYTHONOPTIMIZE", None)
        try:
            return subprocess.run(
                [sys.executable, *flags, str(RECEIPT), *arguments], cwd=self.root,
                env=env, capture_output=True, text=True, encoding="utf-8", timeout=20, check=False,
            )
        except subprocess.TimeoutExpired:
            self.fail("host receipt did not finish within 20 seconds")

    def assert_accepted(self, result, *, test_ids=REQUIRED_IDS, expected=None, image_id=IMAGE_ID):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        identity = self.expected if expected is None else expected
        document = json.loads(result.stdout)
        self.assertEqual(document, {
            "schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
            "checked_commit": identity["commit"], "repository": identity["repository"],
            "run_id": identity["run-id"], "run_attempt": identity["run-attempt"],
            "runtime_image": {"id": image_id, "repo_digest": IMAGE_DIGEST},
            "test_modes": [
                {"mode": mode, "test_count": len(test_ids), "test_ids": sorted(test_ids)}
                for mode in ("normal", "optimized")
            ],
            "graph_source": GRAPH_SOURCE,
            "evidence_file_sha256": {
                name: hashlib.sha256(data).hexdigest() for name, data in self.member_bytes().items()
            },
        })
        self.assertIs(type(document["schema_version"]), int)
        self.assertIs(document["scientific_status_authority"], False)
        for mode in document["test_modes"]:
            self.assertIs(type(mode["test_count"]), int)
        return document

    def assert_refused(self, **kwargs):
        result = self.run_receipt(**kwargs)
        self.assertNotEqual(result.returncode, 0, "invalid staged evidence was accepted")
        decoder = json.JSONDecoder()
        for offset, character in enumerate(result.stdout):
            if character != "{":
                continue
            try:
                document, _ = decoder.raw_decode(result.stdout[offset:])
            except ValueError:
                continue
            if isinstance(document, dict):
                self.assertFalse(
                    document.get("schema_version") == 1
                    and document.get("scientific_effect") == "NONE"
                    and document.get("scientific_status_authority") is False
                    and "evidence_file_sha256" in document,
                    "refusal emitted a successful receipt JSON",
                )
        return result

    def test_valid_53_test_receipt_is_exact_deterministic_and_preserves_staged_bytes(self):
        self.assertEqual(len(REQUIRED_IDS), 53)
        self.assertEqual(len(set(REQUIRED_IDS)), 53)
        before = self.member_bytes()
        first = self.run_receipt()
        self.assert_accepted(first)
        second = self.run_receipt()
        self.assert_accepted(second)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(self.member_bytes(), before)
        self.assertFalse(self.path("architecture-run-receipt.json").exists())
        self.assertEqual({str(path.relative_to(self.evidence)) for path in self.evidence.rglob("*") if path.is_file()}, set(MEMBERS))

    def test_later_all_ok_test_methods_are_counted_and_bound(self):
        test_ids = (*REQUIRED_IDS, "test_architecture_receipt.TESTAdditionalControl.test_additional_control")
        for mode in ("normal", "optimized"):
            self.write_log(mode, test_ids)
        self.assert_accepted(self.run_receipt(), test_ids=test_ids)

    def test_attempt_ten_and_other_native_repository_are_preserved(self):
        self.path("run-attempt.txt").write_text("10\n", encoding="utf-8")
        expected = {**self.expected, "repository": "fixture-owner/research", "run-attempt": "10"}
        self.assert_accepted(self.run_receipt(expected=expected), expected=expected)

    def test_valid_inspected_image_id_is_preserved_instead_of_hardcoded(self):
        image_id = "sha256:" + "d" * 64
        image = copy.deepcopy(self.image)
        image[0]["Id"] = image_id
        image[0]["Config"] = {"Env": ["TEST_FIXTURE=1"]}
        self.write_json("runtime-image.json", image)
        self.assert_accepted(self.run_receipt(), image_id=image_id)

    def test_valid_log_larger_than_64_kib_is_accepted_and_raw_bytes_are_hashed(self):
        path = self.path("tests-normal.log")
        path.write_bytes(b"TEST synthetic diagnostic padding " + b"." * (80 * 1024) + b"\n" + path.read_bytes())
        self.assert_accepted(self.run_receipt())

    def test_valid_log_at_exact_16_mib_limit_is_accepted(self):
        path = self.path("tests-normal.log")
        original = path.read_bytes()
        path.write_bytes(b"." * (FILE_LIMIT - len(original) - 1) + b"\n" + original)
        self.assertEqual(path.stat().st_size, FILE_LIMIT)
        self.assert_accepted(self.run_receipt())

    def test_each_staged_member_is_required(self):
        for name in MEMBERS:
            with self.subTest(name=name):
                path = self.path(name)
                original = path.read_bytes()
                path.unlink()
                self.assert_refused()
                path.write_bytes(original)

    def test_nonzero_or_malformed_container_exit_is_refused(self):
        for value in ("1", "124", "137", "-1", "0\n1", "", "00", "zero"):
            with self.subTest(value=value):
                self.path("container-exit-status.txt").write_text(value + "\n", encoding="utf-8")
                self.assert_refused()

    def test_each_native_identity_file_must_match_current_host_arguments(self):
        for name, stale in (
            ("checked-commit.txt", "d" * 40), ("run-id.txt", "122"), ("run-attempt.txt", "2"),
        ):
            with self.subTest(name=name):
                path = self.path(name)
                original = path.read_bytes()
                path.write_text(stale + "\n", encoding="utf-8")
                self.assert_refused()
                path.write_bytes(original)

    def test_each_expected_host_identity_argument_is_used_and_required(self):
        for name, value in (("commit", "d" * 40), ("run-id", "124"), ("run-attempt", "2")):
            with self.subTest(changed=name):
                self.assert_refused(expected={**self.expected, name: value})
        for omitted in self.expected:
            with self.subTest(omitted=omitted):
                arguments = ["--evidence-dir", str(self.evidence)]
                for name, value in self.expected.items():
                    if name != omitted:
                        arguments.extend(["--expected-" + name, value])
                self.assert_refused(arguments=arguments)

    def test_invalid_matching_host_identity_does_not_authorize_bad_commit_or_decimal(self):
        for argument, name, values in (
            ("commit", "checked-commit.txt", ("A" * 40, "g" * 40, "a" * 39, "a" * 41)),
            ("run-id", "run-id.txt", ("0", "0123", "+123", "123.0", "１２３")),
            ("run-attempt", "run-attempt.txt", ("0", "01", "+1", "1.0", "１")),
        ):
            path = self.path(name)
            original = path.read_bytes()
            for value in values:
                with self.subTest(argument=argument, value=value):
                    path.write_text(value + "\n", encoding="utf-8")
                    self.assert_refused(expected={**self.expected, argument: value})
            path.write_bytes(original)

    def test_native_repository_argument_must_have_owner_and_repository_components(self):
        for value in ("", "owner", "owner/repo/extra", "owner/", "/repo"):
            with self.subTest(value=value):
                self.assert_refused(expected={**self.expected, "repository": value})

    def test_empty_or_zero_discovered_test_logs_are_refused_in_each_mode(self):
        for mode in ("normal", "optimized"):
            path = self.path("tests-" + mode + ".log")
            original = path.read_bytes()
            for contents in ("", "\n", "Ran 0 tests in 0.000s\n\nOK\n"):
                with self.subTest(mode=mode, contents=contents):
                    path.write_text(contents, encoding="utf-8")
                    self.assert_refused()
            path.write_bytes(original)

    def test_every_required_identity_is_needed_even_when_53_other_tests_are_ok(self):
        replacement = "test_architecture_receipt.TESTReplacement.test_unrelated_all_ok"
        for mode in ("normal", "optimized"):
            for missing in REQUIRED_IDS:
                with self.subTest(mode=mode, missing=missing):
                    test_ids = [identity for identity in REQUIRED_IDS if identity != missing] + [replacement]
                    self.write_log(mode, test_ids)
                    self.assert_refused()
            self.write_log(mode)

    def test_each_required_sandbox_control_rejects_skipped_failed_or_error_status(self):
        for mode in ("normal", "optimized"):
            for identity in SANDBOX_IDS:
                for status in ("skipped 'TEST missing sandbox'", "FAIL", "ERROR"):
                    with self.subTest(mode=mode, identity=identity, status=status):
                        self.write_log(mode, statuses={identity: status})
                        self.assert_refused()
            self.write_log(mode)

    def test_skip_or_failure_in_additional_discovered_test_is_also_refused(self):
        extra = "test_architecture_receipt.TESTAdditionalControl.test_additional_control"
        for mode in ("normal", "optimized"):
            for status in ("skipped 'TEST optional'", "FAIL", "ERROR", "expected failure", "unexpected success"):
                with self.subTest(mode=mode, status=status):
                    self.write_log(mode, (*REQUIRED_IDS, extra), statuses={extra: status})
                    self.assert_refused()
            self.write_log(mode)

    def test_summary_must_report_exact_discovered_count_and_terminal_ok(self):
        for mode in ("normal", "optimized"):
            for count in (0, 52, 54):
                with self.subTest(mode=mode, count=count):
                    self.write_log(mode, reported_count=count)
                    self.assert_refused()
            for summary in ("", "FAILED (failures=1)", "OK (skipped=1)", "ok", "OK\nFAILED (errors=1)"):
                with self.subTest(mode=mode, summary=summary):
                    self.write_log(mode, summary=summary)
                    self.assert_refused()
            self.write_log(mode)

    def test_duplicate_test_identity_or_multiple_runs_are_refused(self):
        for mode in ("normal", "optimized"):
            with self.subTest(mode=mode, duplicate="identity"):
                self.write_log(mode, (*REQUIRED_IDS, REQUIRED_IDS[0]))
                self.assert_refused()
            with self.subTest(mode=mode, duplicate="run"):
                path = self.path("tests-" + mode + ".log")
                path.write_text(synthetic_log() + synthetic_log(), encoding="utf-8")
                self.assert_refused()
            self.write_log(mode)

    def test_graph_root_schema_and_non_scientific_flags_are_exact(self):
        for field in self.graph:
            with self.subTest(missing=field):
                graph = copy.deepcopy(self.graph)
                del graph[field]
                self.write_json("generated/graph.json", graph)
                self.assert_refused()
        for field, value in (
            ("extra", "TEST"), ("schema_version", True), ("schema_version", 1.0),
            ("schema_version", 2), ("scientific_effect", "PROMOTE"),
            ("scientific_status_authority", True), ("scientific_status_authority", 0),
            ("scientific_status_authority", "false"),
        ):
            with self.subTest(field=field, value=value):
                graph = copy.deepcopy(self.graph)
                graph[field] = value
                self.write_json("generated/graph.json", graph)
                self.assert_refused()

    def test_every_graph_source_pin_is_required_and_cannot_be_substituted(self):
        for field in GRAPH_SOURCE:
            for replacement in (None, "TEST stale source identity"):
                with self.subTest(field=field, replacement=replacement):
                    graph = copy.deepcopy(self.graph)
                    if replacement is None:
                        del graph["source"][field]
                    else:
                        graph["source"][field] = replacement
                    self.write_json("generated/graph.json", graph)
                    self.assert_refused()
        graph = copy.deepcopy(self.graph)
        graph["source"]["checked_commit"] = "a" * 40
        self.write_json("generated/graph.json", graph)
        self.assert_refused()

    def test_changed_node_or_edge_is_refused_even_when_graph_source_pins_are_unchanged(self):
        graph = copy.deepcopy(self.graph)
        graph["nodes"]["hist.rnu_env.py"]["classification"] = "PROVED_REVIEWED"
        self.write_json("generated/graph.json", graph)
        self.assert_refused()
        graph = copy.deepcopy(self.graph)
        graph["edges"] = graph["edges"][:-1]
        self.write_json("generated/graph.json", graph)
        self.assert_refused()

    def test_missing_extra_or_forged_dimension_cannot_manufacture_scientific_evidence(self):
        node = "hist.rnu_env.py"
        mutations = (
            ("kernel", "recorded"), ("computation", "recorded"), ("alignment", "recorded"),
            ("review", "recorded"), ("source", "recorded"), ("extra", "kernel-checked"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                graph = copy.deepcopy(self.graph)
                graph["dimensions"][node][field] = value
                self.write_json("generated/graph.json", graph)
                self.assert_refused()
        for missing in ("node", "field"):
            with self.subTest(missing=missing):
                graph = copy.deepcopy(self.graph)
                if missing == "node":
                    del graph["dimensions"][node]
                else:
                    del graph["dimensions"][node]["kernel"]
                self.write_json("generated/graph.json", graph)
                self.assert_refused()

    def test_graph_and_runtime_inspect_require_strict_json(self):
        for name in ("generated/graph.json", "runtime-image.json"):
            path = self.path(name)
            original = path.read_bytes()
            for raw in (b"", b"{invalid}", b"null", original + original, b"\xff", b"[" * 2000 + b"0" + b"]" * 2000):
                with self.subTest(name=name, raw=raw[:30]):
                    path.write_bytes(raw)
                    self.assert_refused()
            path.write_bytes(original)

    def test_duplicate_json_keys_are_refused_even_when_last_value_is_valid(self):
        graph = json.dumps(self.graph, separators=(",", ":"))
        self.path("generated/graph.json").write_text('{"scientific_status_authority":true,' + graph[1:], encoding="utf-8")
        self.assert_refused()
        self.write_json("generated/graph.json", self.graph)
        image = json.dumps(self.image[0], separators=(",", ":"))
        self.path("runtime-image.json").write_text('[{"RepoDigests":[], ' + image[1:] + "]", encoding="utf-8")
        self.assert_refused()

    def test_non_finite_numbers_are_refused_even_in_additional_inspect_fields(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(token=token):
                image = json.dumps(self.image[0], separators=(",", ":"))
                self.path("runtime-image.json").write_text('[{"TEST_nonfinite":' + token + "," + image[1:] + "]", encoding="utf-8")
                self.assert_refused()

    def test_runtime_inspect_is_one_linux_amd64_image_with_the_fixed_digest(self):
        for field, value in (
            ("RepoDigests", []), ("RepoDigests", ["python@sha256:" + "d" * 64]),
            ("RepoDigests", [IMAGE_DIGEST, "other@sha256:" + "d" * 64]),
            ("RepoDigests", IMAGE_DIGEST), ("Os", "windows"), ("Architecture", "arm64"),
            ("Id", "sha256:" + "A" * 64), ("Id", "sha256:" + "g" * 64),
            ("Id", "sha256:" + "a" * 63), ("Id", "a" * 64), ("Id", True),
        ):
            with self.subTest(field=field, value=value):
                image = copy.deepcopy(self.image)
                image[0][field] = value
                self.write_json("runtime-image.json", image)
                self.assert_refused()
        for value in ([], self.image[0], [*self.image, *self.image]):
            with self.subTest(shape=value):
                self.write_json("runtime-image.json", value)
                self.assert_refused()
        for field in ("Id", "RepoDigests", "Os", "Architecture"):
            with self.subTest(missing=field):
                image = copy.deepcopy(self.image)
                del image[0][field]
                self.write_json("runtime-image.json", image)
                self.assert_refused()

    def test_each_staged_member_over_16_mib_is_refused(self):
        for name in MEMBERS:
            with self.subTest(name=name):
                path = self.path(name)
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b" " * (FILE_LIMIT + 1 - len(original)))
                    self.assert_refused()
                finally:
                    path.write_bytes(original)

    def test_each_staged_member_must_be_regular_and_cannot_be_a_symlink(self):
        for name in MEMBERS:
            with self.subTest(name=name):
                path = self.path(name)
                original = path.read_bytes()
                target = self.root / "TEST-outside-member"
                target.write_bytes(original)
                path.unlink()
                path.symlink_to(target)
                self.assert_refused()
                path.unlink()
                path.write_bytes(original)

    def test_directory_or_fifo_member_is_refused_without_waiting_for_a_writer(self):
        path = self.path("tests-normal.log")
        original = path.read_bytes()
        path.unlink()
        path.mkdir()
        self.assert_refused()
        path.rmdir()
        os.mkfifo(path)
        self.assert_refused()
        path.unlink()
        path.write_bytes(original)

    def test_evidence_directory_or_generated_parent_symlink_is_refused(self):
        alias = self.root / "TEST-evidence-alias"
        alias.symlink_to(self.evidence, target_is_directory=True)
        self.assert_refused(evidence=alias)
        original = self.evidence / "generated"
        outside = self.root / "TEST-outside-generated"
        original.rename(outside)
        original.symlink_to(outside, target_is_directory=True)
        self.assert_refused()


if __name__ == "__main__":
    unittest.main()
