"""External contract controls for current-run architecture output binding.

Only the stdin CLI is exercised; this module imports no project code. The
producer's strings are synthetic TEST fixtures, not archive-byte custody or
scientific evidence. Run these controls in the hosted ephemeral runner, not on
the owner's laptop.
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "tools/architecture_check_binding.py"
STDIN_LIMIT = 64 * 1024


class ArchitectureBindingContract(unittest.TestCase):
    def setUp(self):
        self.outputs = {
            "checked_commit": "a" * 40,
            "repository": "d6g8k5htny-coder/main",
            "run_id": "123",
            "run_attempt": "1",
            "receipt_sha256": "b" * 64,
            "artifact_id": "12345",
            "artifact_sha256": "c" * 64,
            "artifact_name": "research-architecture-123-1",
        }
        self.payload = {"result": "success", "outputs": copy.deepcopy(self.outputs)}
        self.expected = {
            "commit": "a" * 40,
            "repository": "d6g8k5htny-coder/main",
            "run-id": "123",
            "run-attempt": "1",
        }

    def run_binding(self, payload=None, *, raw=None, expected=None, arguments=None):
        # Every launch must deliberately fail RED if the CLI is absent. A
        # file-not-found subprocess must never count as a successful refusal.
        self.assertTrue(
            EXPORTER.is_file(),
            "required binding CLI tools/architecture_check_binding.py is absent",
        )
        if raw is None:
            raw = json.dumps(self.payload if payload is None else payload)
        if arguments is None:
            identity = self.expected if expected is None else expected
            arguments = []
            for name in ("commit", "repository", "run-id", "run-attempt"):
                arguments.extend(["--expected-" + name, identity[name]])
        python_flags = ["-B", "-S"]
        if sys.flags.optimize:
            python_flags.append("-" + "O" * sys.flags.optimize)
        env = os.environ.copy()
        env.pop("PYTHONOPTIMIZE", None)
        try:
            return subprocess.run(
                [sys.executable, *python_flags, str(EXPORTER), *arguments],
                input=raw, cwd=ROOT, env=env, capture_output=True,
                text=True, encoding="utf-8", timeout=20, check=False,
            )
        except subprocess.TimeoutExpired:
            self.fail("architecture binding did not finish within 20 seconds")

    def assert_accepted(self, result, outputs):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        envelope = json.loads(result.stdout)
        self.assertEqual(envelope, {
            "schema_version": 1,
            "scientific_effect": "NONE",
            "scientific_status_authority": False,
            "binding": outputs,
        })
        self.assertIs(type(envelope["schema_version"]), int)
        self.assertIs(envelope["scientific_status_authority"], False)
        return envelope

    def assert_refused(self, payload=None, **kwargs):
        result = self.run_binding(payload, **kwargs)
        self.assertNotEqual(result.returncode, 0, "invalid binding was accepted")
        # Refusals may explain themselves, but may not emit the success JSON,
        # even alongside diagnostics or as a multiline object before exiting.
        decoder = json.JSONDecoder()
        for offset, character in enumerate(result.stdout):
            if character != "{":
                continue
            try:
                document, _ = decoder.raw_decode(result.stdout[offset:])
            except ValueError:
                continue
            if isinstance(document, dict):
                success = (
                    document.get("schema_version") == 1
                    and document.get("scientific_effect") == "NONE"
                    and document.get("scientific_status_authority") is False
                    and isinstance(document.get("binding"), dict)
                )
                self.assertFalse(success, "refusal emitted a success binding JSON")
        return result

    def changed_output(self, field, value):
        payload = copy.deepcopy(self.payload)
        payload["outputs"][field] = value
        return payload

    def test_valid_current_run_has_exact_stable_non_scientific_output(self):
        first = self.run_binding()
        self.assert_accepted(first, self.outputs)
        second = self.run_binding()
        self.assert_accepted(second, self.outputs)
        self.assertEqual(first.stdout, second.stdout)

    def test_attempt_ten_is_a_positive_decimal_not_a_single_digit(self):
        payload = copy.deepcopy(self.payload)
        payload["outputs"]["run_attempt"] = "10"
        payload["outputs"]["artifact_name"] = "research-architecture-123-10"
        expected = {**self.expected, "run-attempt": "10"}
        self.assert_accepted(self.run_binding(payload, expected=expected), {
            "checked_commit": "a" * 40,
            "repository": "d6g8k5htny-coder/main",
            "run_id": "123",
            "run_attempt": "10",
            "receipt_sha256": "b" * 64,
            "artifact_id": "12345",
            "artifact_sha256": "c" * 64,
            "artifact_name": "research-architecture-123-10",
        })

    def test_other_valid_native_identities_are_accepted_and_preserved(self):
        outputs = {
            "checked_commit": "e" * 40,
            "repository": "fixture-owner/research",
            "run_id": "456",
            "run_attempt": "10",
            "receipt_sha256": "0" * 64,
            "artifact_id": "67890",
            "artifact_sha256": "f" * 64,
            "artifact_name": "research-architecture-456-10",
        }
        payload = {"result": "success", "outputs": outputs}
        expected = {
            "commit": "e" * 40, "repository": "fixture-owner/research",
            "run-id": "456", "run-attempt": "10",
        }
        self.assert_accepted(self.run_binding(payload, expected=expected), outputs)

    def test_valid_json_at_exact_stdin_byte_limit_is_accepted(self):
        raw = json.dumps(self.payload)
        raw += " " * (STDIN_LIMIT - len(raw.encode("utf-8")))
        self.assertEqual(len(raw.encode("utf-8")), STDIN_LIMIT)
        self.assert_accepted(self.run_binding(raw=raw), self.outputs)

    def test_each_non_success_result_is_refused(self):
        for result in (
            "failure", "cancelled", "skipped", "neutral", "timed_out", "pending",
            "", "Success", "SUCCESS", "success ", " success", "success\n",
            None, False, True, 0, 1, 1.0, [], {},
        ):
            with self.subTest(result=result):
                payload = copy.deepcopy(self.payload)
                payload["result"] = result
                self.assert_refused(payload)

    def test_each_top_level_field_is_required(self):
        for field in ("result", "outputs"):
            with self.subTest(field=field):
                payload = copy.deepcopy(self.payload)
                del payload[field]
                self.assert_refused(payload)
        self.assert_refused({})

    def test_extra_top_level_fields_are_refused(self):
        for field in (
            "other", "binding", "schema_version", "scientific_effect",
            "scientific_status_authority", "conclusion", "architecture",
        ):
            with self.subTest(field=field):
                payload = copy.deepcopy(self.payload)
                payload[field] = "TEST extra field"
                self.assert_refused(payload)

    def test_non_object_top_level_values_are_refused(self):
        for value in (None, False, True, 0, 1.0, "", "success", [], [self.payload]):
            with self.subTest(value=value):
                self.assert_refused(raw=json.dumps(value))

    def test_outputs_must_be_an_object(self):
        for value in (None, False, True, 0, 1.0, "", [], [self.outputs]):
            with self.subTest(value=value):
                payload = copy.deepcopy(self.payload)
                payload["outputs"] = value
                self.assert_refused(payload)

    def test_each_of_the_eight_outputs_is_required(self):
        for field in self.outputs:
            with self.subTest(field=field):
                payload = copy.deepcopy(self.payload)
                del payload["outputs"][field]
                self.assert_refused(payload)

    def test_extra_output_fields_are_refused(self):
        for field in (
            "other", "checked_sha", "workflow_run_id", "artifact_url",
            "schema_version", "scientific_effect", "scientific_status_authority",
            "conclusion", "formalization_status",
        ):
            with self.subTest(field=field):
                payload = copy.deepcopy(self.payload)
                payload["outputs"][field] = "TEST extra field"
                self.assert_refused(payload)

    def test_each_output_rejects_non_string_identities_without_coercion(self):
        for field in self.outputs:
            for value in (None, False, True, 0, 123, 1.0, 123.0, [], {}):
                with self.subTest(field=field, value=value):
                    self.assert_refused(self.changed_output(field, value))

    def test_each_output_rejects_an_empty_string(self):
        for field in self.outputs:
            with self.subTest(field=field):
                self.assert_refused(self.changed_output(field, ""))

    def test_stale_checked_commit_is_refused(self):
        self.assert_refused(self.changed_output("checked_commit", "d" * 40))

    def test_repository_substitution_and_case_drift_are_refused(self):
        for repository in (
            "d6g8k5htny-coder/Math-", "other-owner/main", "D6g8k5htny-coder/main",
            "d6g8k5htny-coder/Main", "d6g8k5htny-coder/main ",
        ):
            with self.subTest(repository=repository):
                self.assert_refused(self.changed_output("repository", repository))

    def test_stale_run_is_refused_even_with_a_matching_stale_artifact_name(self):
        for name in ("research-architecture-123-1", "research-architecture-122-1"):
            with self.subTest(name=name):
                payload = self.changed_output("run_id", "122")
                payload["outputs"]["artifact_name"] = name
                self.assert_refused(payload)

    def test_stale_attempt_is_refused_even_with_a_matching_stale_artifact_name(self):
        for name in ("research-architecture-123-1", "research-architecture-123-2"):
            with self.subTest(name=name):
                payload = self.changed_output("run_attempt", "2")
                payload["outputs"]["artifact_name"] = name
                self.assert_refused(payload)

    def test_each_current_native_argument_is_used_for_identity_binding(self):
        substitutions = {
            "commit": "d" * 40,
            "repository": "other-owner/research",
            "run-id": "124",
            "run-attempt": "2",
        }
        for argument, value in substitutions.items():
            with self.subTest(argument=argument):
                self.assert_refused(expected={**self.expected, argument: value})

    def test_commit_format_cannot_be_authorized_by_an_equally_invalid_native_commit(self):
        for value in (
            "A" * 40, "g" * 40, "a" * 39, "a" * 41, "0x" + "a" * 40,
            " " + "a" * 40, "a" * 40 + "\n",
        ):
            with self.subTest(value=value):
                self.assert_refused(
                    self.changed_output("checked_commit", value),
                    expected={**self.expected, "commit": value},
                )

    def test_receipt_and_artifact_sha256_require_lowercase_64_hex_digits(self):
        for field in ("receipt_sha256", "artifact_sha256"):
            for value in (
                "B" * 64, "g" * 64, "b" * 63, "b" * 65, "0x" + "b" * 64,
                " " + "b" * 64, "b" * 64 + "\n", "b" * 32 + "-" + "b" * 31,
            ):
                with self.subTest(field=field, value=value):
                    self.assert_refused(self.changed_output(field, value))

    def test_artifact_id_requires_positive_ascii_decimal_without_leading_zero(self):
        for value in (
            "0", "00", "012345", "-1", "+12345", "12345.0", "1e5",
            " 12345", "12345 ", "12345\n", "１２３４５", "١٢٣٤٥", "12345/../other",
            "12345;echo TEST", "$(echo TEST)", "12345\nartifact_sha256=forged",
        ):
            with self.subTest(value=value):
                self.assert_refused(self.changed_output("artifact_id", value))

    def test_run_and_attempt_decimal_format_cannot_be_authorized_by_invalid_native_args(self):
        for field, argument in (("run_id", "run-id"), ("run_attempt", "run-attempt")):
            for value in (
                "0", "00", "01", "-1", "+1", "1.0", "1e2", " 1", "1 ", "1\n",
                "１", "١", "1;echo TEST", "$(echo TEST)", "1\nresult=success",
            ):
                with self.subTest(field=field, value=value):
                    payload = self.changed_output(field, value)
                    run = payload["outputs"]["run_id"]
                    attempt = payload["outputs"]["run_attempt"]
                    payload["outputs"]["artifact_name"] = "research-architecture-" + run + "-" + attempt
                    self.assert_refused(payload, expected={**self.expected, argument: value})

    def test_artifact_name_is_exactly_bound_to_current_run_and_attempt(self):
        for value in (
            "research-architecture-122-1", "research-architecture-123-2",
            "research-architecture-0123-1", "research-architecture-123-01",
            "Research-architecture-123-1", "research-architecture-123-1.zip",
            " research-architecture-123-1", "research-architecture-123-1 ",
            "research-architecture-123-1\n", "../research-architecture-123-1",
            "research-architecture-123-1;echo TEST", "$(echo TEST)",
        ):
            with self.subTest(value=value):
                self.assert_refused(self.changed_output("artifact_name", value))

    def test_duplicate_top_level_keys_are_refused_even_when_last_value_is_valid(self):
        encoded = json.dumps(self.payload, separators=(",", ":"))
        for prefix in (
            '"result":"failure",', '"result":"success",',
            '"outputs":{},', '"outputs":' + json.dumps(self.outputs) + ",",
        ):
            with self.subTest(prefix=prefix):
                self.assert_refused(raw="{" + prefix + encoded[1:])

    def test_each_duplicate_output_key_is_refused_even_when_last_value_is_valid(self):
        encoded = json.dumps(self.outputs, separators=(",", ":"))
        for field, value in self.outputs.items():
            for earlier in ("TEST stale identity", value):
                with self.subTest(field=field, earlier=earlier):
                    outputs = "{" + json.dumps(field) + ":" + json.dumps(earlier) + "," + encoded[1:]
                    self.assert_refused(raw='{"result":"success","outputs":' + outputs + "}")

    def test_non_finite_json_numbers_are_refused(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
            for field in ("result", "outputs"):
                with self.subTest(token=token, field=field):
                    if field == "result":
                        raw = '{"result":' + token + ',"outputs":' + json.dumps(self.outputs) + "}"
                    else:
                        raw = '{"result":"success","outputs":' + token + "}"
                    self.assert_refused(raw=raw)
            with self.subTest(token=token, field="artifact_id"):
                outputs = json.dumps(self.outputs)
                outputs = outputs.replace('"artifact_id": "12345"', '"artifact_id": ' + token)
                self.assert_refused(raw='{"result":"success","outputs":' + outputs + "}")

    def test_valid_json_over_stdin_byte_limit_is_refused(self):
        raw = json.dumps(self.payload)
        raw += " " * (STDIN_LIMIT + 1 - len(raw.encode("utf-8")))
        self.assertEqual(len(raw.encode("utf-8")), STDIN_LIMIT + 1)
        self.assert_refused(raw=raw)

    def test_deeply_nested_json_is_refused_without_a_success_envelope(self):
        nested = "[" * 2000 + "0" + "]" * 2000
        self.assert_refused(raw='{"result":"success","outputs":' + nested + "}")

    def test_invalid_or_multiple_json_documents_are_refused(self):
        valid = json.dumps(self.payload)
        for raw in (
            "", " \n\t", "{", "{invalid}", valid[:-1],
            valid + "{}", valid + "\n" + valid, valid + " TEST", "// TEST\n" + valid,
            valid.replace('"result": "success"', '"result": "success",', 1),
        ):
            with self.subTest(raw=raw[:80]):
                self.assert_refused(raw=raw)

    def test_each_expected_native_identity_argument_is_required(self):
        for omitted in self.expected:
            with self.subTest(omitted=omitted):
                arguments = []
                for name, value in self.expected.items():
                    if name != omitted:
                        arguments.extend(["--expected-" + name, value])
                self.assert_refused(arguments=arguments)


if __name__ == "__main__":
    unittest.main()
