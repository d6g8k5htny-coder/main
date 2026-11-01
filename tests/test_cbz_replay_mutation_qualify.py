"""Adversarial tests of recorded replay qualification; no Lean or replay execution."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = Path(os.environ.get("CBZ_REPLAY_QUALIFIER", ROOT / "tools/cbz_replay_mutation_qualify.py"))
SOURCE = ROOT / "cbz/evidence/kernel/additional/replay-tool"
MODES = ("normal", "optimized")
CASES = {
    "owned-names": ("test_wrong_owned_name", 60, "owned names differ",
                    'AssertionError: "owned names differ" does not match "module assignment differs from complete compiled census"'),
    "module-assignment": ("test_wrong_module_assignment", 65, "module assignment differs",
                          "AssertionError: ValueError not raised"),
    "axioms": ("test_forbidden_axiom", 69, "forbidden axiom",
               "AssertionError: ValueError not raised"),
}
MUTANT_HASHES = {
    "owned-names": "8550c4d02001d4834461efac04f0a3c623e7fe9db0a5a27a706498fd6659c25d",
    "module-assignment": "e9d132c898ecf5004aba8e7408b7f77113ac5138c0255c5b34b511af06dc1648",
    "axioms": "eb693cb3faf9bcfbe9948caa88f5d4ed7e5e2deed9e0f16bd9eb92a39874385d",
}
NAMES = {
    "GeneratorTests": """cli_validation_round_trip_and_no_overwrite collector_hash_mismatch
duplicate_and_nonfinite_json_are_rejected exclusive_output_preserves_existing_file
existing_second_output_creates_no_first_output missing_or_ambiguous_collector_literal
no_execution_of_collector_python nonempty_unique_simple_modules_and_imports output_alias_is_rejected
outputs_cannot_replace_input positive_generator_and_verbatim_collector symlink_output_alias_is_rejected""".split(),
    "TranscriptTests": """bad_module_records begin_and_completion_counts_agree caller_cannot_expand_axiom_allowlist
duplicate_axiom empty_axiom_list_is_allowed empty_or_inconsistent_expected_census
execution_failure_or_stderr_cannot_qualify forbidden_axiom malformed_allowlist_or_nontext_input
malformed_markers_and_noise missing_extra_or_duplicate_seed missing_or_repeated_begin
missing_or_repeated_pass no_pass_before_begin no_records_after_completion positive_exact_census
raw_stream_digest_preserves_line_endings wrong_module_assignment wrong_owned_name""".split(),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def transcript(case=None):
    target = CASES[case][0] if case else None
    lines = []
    for cls in sorted(NAMES):
        for short in sorted(NAMES[cls]):
            method = "test_" + short
            lines.append("%s (test_replay_tool.%s.%s) ... %s" %
                         (method, cls, method, "FAIL" if method == target else "ok"))
    text = "\n".join(lines) + "\n"
    if case:
        method, line, reason, assertion = CASES[case]
        text += "\n" + "=" * 70 + "\nFAIL: %s (test_replay_tool.TranscriptTests.%s)\n" % (method, method)
        text += "-" * 70 + "\n"
        if case == "owned-names":
            text += "ValueError: module assignment differs from complete compiled census\n\nDuring handling of the above exception, another exception occurred:\n\n"
        text += 'Traceback (most recent call last):\n  File "/fixture/test_replay_tool.py", line %d, in %s\n' % (line, method)
        text += "    with self.assertRaisesRegex(ValueError, '%s'):\n%s\n" % (reason, assertion)
    text += "\n" + "-" * 70 + "\nRan 31 tests in 0.100s\n\n"
    text += ("FAILED (failures=1)" if case else "OK") + "\n"
    return text.encode()


def fixture(directory):
    directory.mkdir()
    report = {
        "schema": "cbz-replay-source-mutations-v1",
        "source_sha256": "b97bee53a590651b83d68056c090ba70e8a055157e65d2bd3fcc5cd0219840b3",
        "test_sha256": "a07eaab8b6fd40a911d313197fae688927ebb3e1d74644628e4888a7e6505fc0",
        "python": "3.11.16 (fixture) [fixture]",
        "python_executable": "/fixture/python",
        "threshold": "100% of the three declared validator-source mutants in each Python mode",
        "baseline": {}, "mutants": {}, "status": "PASS_ALL_DECLARED_MUTANTS_KILLED",
    }
    for label in ("baseline", *CASES):
        for mode in MODES:
            data = transcript(None if label == "baseline" else label)
            (directory / (label + "-" + mode + ".stdout")).write_bytes(b"")
            (directory / (label + "-" + mode + ".stderr")).write_bytes(data)
            row = {"exit_code": 0 if label == "baseline" else 1, "stderr_sha256": digest(data)}
            if label == "baseline":
                report["baseline"][mode] = row
            else:
                row.update(expected_failed_test=CASES[label][0], reason_bound_kill=True)
                report["mutants"].setdefault(label, {"source_sha256": MUTANT_HASHES[label], "modes": {}})["modes"][mode] = row
    save(directory, report)
    return report


def save(directory, report):
    (directory / "REPORT.json").write_text(json.dumps(report) + "\n")


def rebind(directory, report, label, mode, data):
    (directory / (label + "-" + mode + ".stderr")).write_bytes(data)
    row = report["baseline"][mode] if label == "baseline" else report["mutants"][label]["modes"][mode]
    row["stderr_sha256"] = digest(data)
    save(directory, report)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), "the recorded replay post-qualifier is missing")
        spec = importlib.util.spec_from_file_location("qualifier_under_test", TOOL)
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "run"
        self.report = fixture(self.directory)

    def qualify(self):
        return self.tool.qualify(self.directory, SOURCE)

    def reject(self):
        with self.assertRaises(ValueError):
            self.qualify()

    def test_complete_recorded_contract(self):
        result = self.qualify()
        self.assertEqual(result["status"], "PASS_RECORDED_MUTATION_QUALIFICATION")
        self.assertEqual(result["qualified_runs"], 8)
        self.assertEqual(len(result["input_sha256"]), 17)

    def test_same_method_wrong_cause_rejected(self):
        for label in CASES:
            for mode in MODES:
                with self.subTest(label=label, mode=mode):
                    original = transcript(label)
                    bad = original.replace(CASES[label][3].encode(), b"AssertionError: unrelated fixture failure")
                    rebind(self.directory, self.report, label, mode, bad)
                    self.reject()
                    rebind(self.directory, self.report, label, mode, original)

    def test_foreign_error_rejected_even_with_correct_failure(self):
        data = transcript("axioms").replace(b"Ran 31 tests", b"RuntimeError: unrelated failure\nRan 31 tests")
        rebind(self.directory, self.report, "axioms", "normal", data)
        self.reject()

    def test_embedded_expected_cause_does_not_hide_wrong_final_cause(self):
        original = transcript("axioms")
        data = original.replace(b"AssertionError: ValueError not raised",
            b'    print("AssertionError: ValueError not raised")\nAssertionError: unrelated')
        rebind(self.directory, self.report, "axioms", "normal", data)
        self.reject()

    def test_unknown_field_rejected(self):
        self.report["baseline"]["normal"]["unadmitted"] = 1
        save(self.directory, self.report)
        self.reject()

    def test_mixed_failure_rejected(self):
        data = transcript("axioms").replace(b"FAILED (failures=1)", b"FAIL: test_foreign (foreign.Suite.test_foreign)\nAssertionError: foreign\nFAILED (failures=2)")
        rebind(self.directory, self.report, "axioms", "normal", data)
        self.reject()

    def test_count_skip_duplicate_and_unknown_test_rejected(self):
        original = transcript()
        variants = (original.replace(b"Ran 31", b"Ran 30"),
                    original.replace(b"... ok", b"... skipped 'missing'", 1),
                    original.replace(b"test_collector_hash_mismatch", b"test_unknown", 2),
                    original.replace(b"\nOK\n", b"\nOK (skipped=1)\n"),
                    original + original)
        for data in variants:
            with self.subTest(data=data[-60:]):
                rebind(self.directory, self.report, "baseline", "normal", data)
                self.reject()

    def test_raw_digest_and_stdout_rejected(self):
        p = self.directory / "axioms-normal.stderr"
        original = p.read_bytes()
        p.write_bytes(original + b"\n")
        self.reject()
        p.write_bytes(original)
        (self.directory / "axioms-normal.stdout").write_bytes(b"unexpected output")
        self.reject()

    def test_wrong_report_values_rejected(self):
        variants = [
            ("exit", lambda r: r["mutants"]["axioms"]["modes"]["normal"].update(exit_code=2)),
            ("bool-exit", lambda r: r["mutants"]["axioms"]["modes"]["normal"].update(exit_code=True)),
            ("bool-baseline-exit", lambda r: r["baseline"]["normal"].update(exit_code=False)),
            ("source", lambda r: r.update(source_sha256="0"*64)),
            ("test", lambda r: r.update(test_sha256="0"*64)),
            ("mutant-source", lambda r: r["mutants"]["axioms"].update(source_sha256="0"*64)),
            ("false-reason", lambda r: r["mutants"]["axioms"]["modes"]["normal"].update(reason_bound_kill=False)),
            ("different-test", lambda r: r["mutants"]["axioms"]["modes"]["normal"].update(expected_failed_test="test_other")),
            ("missing-mode", lambda r: r["baseline"].pop("optimized")),
            ("extra-mutant", lambda r: r["mutants"].update(extra={})),
            ("unknown-field", lambda r: r["baseline"]["normal"].update(unknown=1)),
            ("runtime", lambda r: r.update(python="3.9.6")),
            ("status", lambda r: r.update(status="FAIL")),
        ]
        for name, edit in variants:
            with self.subTest(name=name):
                r = copy.deepcopy(self.report)
                edit(r)
                save(self.directory, r)
                self.reject()

    def test_duplicate_and_nonfinite_json_rejected(self):
        text = json.dumps(self.report)
        for data in ('{"schema":"duplicate",' + text[1:],
                     text.replace('"exit_code": 0', '"exit_code": NaN', 1),
                     text.replace('"exit_code": 0', '"exit_code": Infinity', 1),
                     text.replace('"exit_code": 0', '"exit_code": 1e999', 1)):
            with self.subTest(data=data[:40]):
                (self.directory / "REPORT.json").write_text(data)
                self.reject()

    def test_missing_extra_and_symlink_inputs_rejected(self):
        p = self.directory / "unexpected"
        p.write_bytes(b"")
        self.reject()
        p.unlink()
        p = self.directory / "axioms-normal.stdout"
        p.unlink()
        self.reject()
        p.symlink_to(self.directory / "baseline-normal.stdout")
        self.reject()

    def test_wrong_source_bytes_rejected(self):
        source = Path(self.temp.name) / "source"
        source.mkdir()
        for name in ("replay_tool.py", "test_replay_tool.py", "run_mutations.py"):
            (source / name).write_bytes((SOURCE / name).read_bytes())
        for name in ("replay_tool.py", "test_replay_tool.py", "run_mutations.py"):
            with self.subTest(name=name):
                original = (source / name).read_bytes()
                (source / name).write_bytes(original + b"\n")
                with self.assertRaises(ValueError):
                    self.tool.qualify(self.directory, source)
                (source / name).write_bytes(original)

    def test_cli_fresh_output_and_failure_receipt(self):
        output = Path(self.temp.name) / "qualified.json"
        command = [sys.executable, *(['-O'] if sys.flags.optimize else []), "-B", str(TOOL),
                   "--input", str(self.directory), "--source-dir", str(SOURCE), "--output", str(output)]
        r = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        before = output.read_bytes()
        r = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(output.read_bytes(), before)
        output.unlink()
        rebind(self.directory, self.report, "axioms", "normal",
               transcript("axioms").replace(b"AssertionError: ValueError not raised", b"AssertionError: unrelated"))
        r = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(json.loads(output.read_text())["status"], "FAIL")

    def test_output_cannot_add_to_input_directory(self):
        output = self.directory / "qualification.json"
        r = subprocess.run([sys.executable, "-B", str(TOOL), "--input", str(self.directory),
                            "--source-dir", str(SOURCE), "--output", str(output)], capture_output=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
