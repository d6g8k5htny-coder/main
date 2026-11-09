"""Adverse execution controls for the node-status mutation grader."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
GRADER = ROOT / "cbz/evidence/node-status-freshness/run_mutations.py"
TEST_ID = "tests.fixture.Fixture.test_selected"


def load_grader():
    spec = importlib.util.spec_from_file_location("node_freshness_grader_under_test", GRADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MutationOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grader = load_grader()

    def check_case(self, method, *, setup="pass", teardown="pass"):
        self.assertTrue(hasattr(self.grader, "run_selected"), "missing provenance-aware runner")
        self.assertTrue(hasattr(self.grader, "qualified_kill"), "missing provenance-aware classifier")
        source = """\
import unittest

def subject():
    raise AssertionError('subject assertion')

class Fixture(unittest.TestCase):
    def setUp(self):
        {setup}

    def tearDown(self):
        {teardown}

    def helper(self):
        self.fail('helper assertion')

    def test_selected(self):
{method}
""".format(setup=setup, teardown=teardown,
           method=textwrap.indent(textwrap.dedent(method).strip(), "        "))
        with tempfile.TemporaryDirectory(prefix="freshness-origin-fixture-") as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests/__init__.py").write_text("", encoding="utf-8")
            (root / "tests/fixture.py").write_text(source, encoding="utf-8")
            return [self.grader.qualified_kill(self.grader.run_selected(root, TEST_ID, optimized), TEST_ID)
                    for optimized in (False, True)]

    def test_selected_body_assertion_is_a_kill(self):
        self.assertEqual(self.check_case("raise AssertionError('selected body')"), [True, True])

    def test_selected_unittest_oracle_is_a_kill(self):
        self.assertEqual(self.check_case("self.assertEqual(1, 2)"), [True, True])

    def test_selected_assert_raises_oracle_is_a_kill(self):
        self.assertEqual(self.check_case("with self.assertRaises(ValueError):\n    pass"), [True, True])

    def test_selected_subtest_oracle_is_a_kill(self):
        self.assertEqual(self.check_case("with self.subTest(number=1):\n    self.assertEqual(1, 2)"), [True, True])

    def test_subject_assertion_is_not_a_kill(self):
        self.assertEqual(self.check_case("self.assertEqual(subject(), 1)"), [False, False])

    def test_real_subject_assertion_is_not_a_kill(self):
        test_id = ("tests.test_node_status_freshness.ContentTests."
                   "test_strict_invalid_records_never_repaired_to_green")
        target_path = "tools/node_status_refresh.py"
        old = "if problems:"
        adverse = "if (_ for _ in ()).throw(AssertionError('subject-origin probe')):"
        paths = [target_path, "tools/validate_node_status.py",
                 "tests/test_node_status_freshness.py", "architecture/node-status.schema.json",
                 "docs/site/current.json"]
        paths += [str(p.relative_to(ROOT)) for p in (ROOT / "records/node_status").glob("*.json")]
        with tempfile.TemporaryDirectory(prefix="freshness-real-origin-") as tmp:
            root = Path(tmp)
            for path in paths:
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / path, root / path)
            for package in ("tools", "tests"):
                (root / package / "__init__.py").write_text("", encoding="utf-8")
            target = root / target_path
            original = target.read_text(encoding="utf-8")
            self.assertEqual(original.count(old), 1)
            for optimized in (False, True):
                target.write_text(original, encoding="utf-8")
                baseline = self.grader.run_selected(root, test_id, optimized)
                self.assertTrue(self.grader.baseline_ok(baseline, test_id), baseline)
                target.write_text(original.replace(old, adverse, 1), encoding="utf-8")
                result = self.grader.run_selected(root, test_id, optimized)
                self.assertFalse(self.grader.qualified_kill(result, test_id), result)
                self.assertEqual(result["failure_count"], 1)
                self.assertEqual(result["error_count"], 0)
                self.assertFalse(result["events"][0]["authorized"])
                self.assertTrue(any(frame["filename"].endswith(target_path)
                                    for frame in result["events"][0]["frames"]))
                print("FRESHNESS_REAL_SUBJECT " + json.dumps({
                    "optimized": optimized, "test_id": test_id,
                    "baseline": baseline, "adverse": result,
                }, sort_keys=True), flush=True)

    def test_helper_assertion_is_not_a_kill(self):
        self.assertEqual(self.check_case("self.helper()"), [False, False])

    def test_nested_callback_assertion_is_not_a_kill(self):
        self.assertEqual(self.check_case("def callback():\n    self.fail('nested assertion')\ncallback()"), [False, False])

    def test_spoofed_unittest_filename_is_not_a_kill(self):
        self.assertEqual(self.check_case("exec(compile(\"raise AssertionError('spoof')\", unittest.case.__file__, 'exec'))"), [False, False])

    def test_equal_but_distinct_oracle_code_is_not_a_kill(self):
        method = ("import types\n"
                  "clone = unittest.TestCase.fail.__code__.replace(co_filename='forged-subject.py')\n"
                  "types.FunctionType(clone, globals())(self, 'forged oracle')")
        self.assertEqual(self.check_case(method), [False, False])

    def test_setup_assertion_is_not_a_kill(self):
        self.assertEqual(self.check_case("pass", setup="raise AssertionError('setup assertion')"), [False, False])

    def test_setup_calling_selected_body_is_not_a_kill(self):
        body = "self.calls += 1\nif self.calls == 1: self.fail('called from setup')"
        self.assertEqual(self.check_case(body, setup="self.calls = 0; self.test_selected()"), [False, False])

    def test_teardown_assertion_is_not_a_kill(self):
        self.assertEqual(self.check_case("pass", teardown="raise AssertionError('teardown assertion')"), [False, False])

    def test_teardown_calling_selected_body_is_not_a_kill(self):
        body = "self.calls += 1\nif self.calls == 2: self.fail('called from teardown')"
        self.assertEqual(self.check_case(body, setup="self.calls = 0", teardown="self.test_selected()"), [False, False])

    def test_selected_failure_with_teardown_failure_is_not_a_kill(self):
        self.assertEqual(self.check_case("self.assertEqual(1, 2)", teardown="raise AssertionError('teardown assertion')"), [False, False])

    def test_ordinary_error_is_not_a_kill(self):
        self.assertEqual(self.check_case("raise ValueError('runtime error')"), [False, False])

    def test_runtime_error_reclassified_as_failure_is_not_a_kill(self):
        body = "self.failureException = ValueError\nraise ValueError('runtime error')"
        self.assertEqual(self.check_case(body), [False, False])

    def test_skipped_test_is_not_a_kill(self):
        self.assertEqual(self.check_case("self.skipTest('not executed')"), [False, False])

    def test_failure_with_later_skip_is_not_a_kill(self):
        self.assertEqual(self.check_case("with self.subTest(number=1):\n    self.fail('body failure')\nself.skipTest('later skip')"), [False, False])

    def test_corrupt_provenance_counts_are_refused(self):
        self.assertTrue(hasattr(self.grader, "qualified_kill"), "missing provenance-aware classifier")
        valid = {"schema": "node-freshness-origin-v2", "protocol_error": False,
                 "test_id": TEST_ID, "selected_id": TEST_ID,
                 "tests_run": 1, "failure_count": 1, "error_count": 0, "skip_count": 0,
                 "expected_failure_count": 0, "unexpected_success_count": 0, "child_exit": 1,
                 "optimize": 0, "requested_optimize": 0, "body_entered": 1,
                 "events": [{"test_id": TEST_ID, "kind": "failure", "authorized": True,
                             "exception": "AssertionError"}]}
        self.assertTrue(self.grader.qualified_kill(valid, TEST_ID))
        for key, value in (("tests_run", True), ("failure_count", True),
                           ("error_count", False), ("child_exit", True), ("body_entered", True),
                           ("body_entered", 0),
                           ("optimize", True)):
            with self.subTest(key=key):
                corrupted = dict(valid, **{key: value})
                self.assertFalse(self.grader.qualified_kill(corrupted, TEST_ID))
        for mode in (-1, 2):
            corrupted = dict(valid, optimize=mode, requested_optimize=mode)
            self.assertFalse(self.grader.qualified_kill(corrupted, TEST_ID))
        self.assertFalse(self.grader.qualified_kill(dict(valid, optimize=1), TEST_ID))
        event = dict(valid["events"][0], exception="ValueError")
        self.assertFalse(self.grader.qualified_kill(dict(valid, events=[event]), TEST_ID))

    def test_non_object_provenance_is_refused(self):
        for malformed in (None, [], "not a report", 1):
            with self.subTest(malformed=malformed):
                self.assertFalse(self.grader.qualified_kill(malformed, TEST_ID))
                self.assertFalse(self.grader.baseline_ok(malformed, TEST_ID))


class MutationGuardSelfTests(unittest.TestCase):
    def test_origin_and_classifier_mutants_fail_for_the_designated_oracle(self):
        grader = load_grader()
        original = GRADER.read_text(encoding="utf-8")
        mutations = (
            ("ignore non-oracle frames", "and all(is_oracle(code) for code, _ in tail))", "and True)",
             "test_subject_assertion_is_not_a_kill"),
            ("compare oracle code by equality", "any(code is allowed for allowed in oracle)",
             "any(code == allowed for allowed in oracle)",
             "test_equal_but_distinct_oracle_code_is_not_a_kill"),
            ("accept selected body from setup",
             (('and state["entered"] == 1 and state["phase"] in ("body", "body-exited")',
               "and True"), ("result.get('body_entered') == 1", "True")),
             None, "test_setup_calling_selected_body_is_not_a_kill"),
            ("accept selected body from teardown",
             'and state["entered"] == 1 and state["phase"] in ("body", "body-exited")',
             "and True", "test_teardown_calling_selected_body_is_not_a_kill"),
            ("ignore entry count", "result.get('body_entered') == 1", "True",
             "test_corrupt_provenance_counts_are_refused"),
            ("accept unknown mode", "result.get('optimize') in (0, 1)", "True",
             "test_corrupt_provenance_counts_are_refused"),
            ("ignore mode mismatch", "result.get('optimize') == result.get('requested_optimize')",
             "True", "test_corrupt_provenance_counts_are_refused"),
            ("accept reclassified runtime error",
             (("and type(err[1]) is AssertionError", "and True"),
              ("and event.get('exception') == 'AssertionError'", "and True")),
             None, "test_runtime_error_reclassified_as_failure_is_not_a_kill"),
            ("ignore authorization", "and event.get('authorized') is True", "and True",
             "test_subject_assertion_is_not_a_kill"),
            ("accept boolean counters", "all(type(result.get(key)) is int for key in count_keys)", "True",
             "test_corrupt_provenance_counts_are_refused"),
            ("ignore skips", "and result.get('skip_count') == 0", "and True",
             "test_failure_with_later_skip_is_not_a_kill"),
        )
        with tempfile.TemporaryDirectory(prefix="freshness-grader-mutants-") as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests/__init__.py").write_text("", encoding="utf-8")
            shutil.copyfile(Path(__file__), root / "tests/test_node_status_mutation_grader.py")
            target = root / "cbz/evidence/node-status-freshness/run_mutations.py"
            target.parent.mkdir(parents=True)
            for name, old, new, method in mutations:
                with self.subTest(mutation=name):
                    edits = old if isinstance(old, tuple) else ((old, new),)
                    for before, _ in edits:
                        self.assertEqual(original.count(before), 1, name)
                    test_id = "tests.test_node_status_mutation_grader.MutationOriginTests." + method
                    for optimized in (False, True):
                        target.write_text(original, encoding="utf-8")
                        baseline = grader.run_selected(root, test_id, optimized)
                        self.assertTrue(grader.baseline_ok(baseline, test_id), (name, optimized, baseline))
                        changed = original
                        for before, after in edits:
                            changed = changed.replace(before, after, 1)
                        target.write_text(changed, encoding="utf-8")
                        mutated = grader.run_selected(root, test_id, optimized)
                        self.assertTrue(grader.qualified_kill(mutated, test_id), (name, optimized, mutated))
                        print("FRESHNESS_GRADER_MUTATION " + json.dumps({
                            "name": name, "optimized": optimized, "test_id": test_id,
                            "old": old, "new": new, "baseline": baseline, "mutated": mutated,
                        }, sort_keys=True), flush=True)


class WorkflowWiringTests(unittest.TestCase):
    @staticmethod
    def accepts(workflow):
        try:
            controls = workflow.split("  controls:\n", 1)[1].split("\n  node-status-refresh:", 1)[0]
            step = controls.split("      - name: Verify freshness mutation provenance\n", 1)[1]
            step = step.split("\n      - name:", 1)[0]
        except IndexError:
            return False
        required_step = """\
        run: |
          set -euo pipefail
          mkdir -p "$RUNNER_TEMP/node-status-grader-evidence"
          python3 -B -S -m unittest tests.test_node_status_mutation_grader -v > "$RUNNER_TEMP/node-status-grader-evidence/controls-normal.stdout" 2> "$RUNNER_TEMP/node-status-grader-evidence/controls-normal.stderr"
          python3 -B -O -S -m unittest tests.test_node_status_mutation_grader -v > "$RUNNER_TEMP/node-status-grader-evidence/controls-optimized.stdout" 2> "$RUNNER_TEMP/node-status-grader-evidence/controls-optimized.stderr"
          python3 -B -S cbz/evidence/node-status-freshness/run_mutations.py > "$RUNNER_TEMP/node-status-grader-evidence/mutations-normal.json" 2> "$RUNNER_TEMP/node-status-grader-evidence/mutations-normal.stderr"
          python3 -B -O -S cbz/evidence/node-status-freshness/run_mutations.py > "$RUNNER_TEMP/node-status-grader-evidence/mutations-optimized.json" 2> "$RUNNER_TEMP/node-status-grader-evidence/mutations-optimized.stderr"
"""
        return ("      - 'tests/test_node_status_mutation_grader.py'\n" in workflow
                and "continue-on-error:" not in controls
                and step + "\n" == required_step
                and "- name: Preserve freshness mutation evidence\n        if: always()" in controls
                and "if-no-files-found: error" in controls)

    def test_freshness_grader_checks_are_required_and_preserve_results(self):
        workflow = (ROOT / ".github/workflows/current-site.yml").read_text(encoding="utf-8")
        self.assertTrue(self.accepts(workflow))

    def test_workflow_contract_refuses_bypasses(self):
        workflow = (ROOT / ".github/workflows/current-site.yml").read_text(encoding="utf-8")
        mutations = (
            ("missing trigger", "      - 'tests/test_node_status_mutation_grader.py'\n", ""),
            ("ignored failure", "mutations-normal.stderr\"\n", "mutations-normal.stderr\" || true\n"),
            ("disabled step", "      - name: Verify freshness mutation provenance\n",
             "      - name: Verify freshness mutation provenance\n        if: false\n"),
            ("missing errexit", "          set -euo pipefail\n", ""),
            ("allowed job failure", "  controls:\n", "  controls:\n    continue-on-error: true\n"),
            ("missing failure artifact",
             "- name: Preserve freshness mutation evidence\n        if: always()",
             "- name: Preserve freshness mutation evidence\n        if: success()"),
        )
        for name, old, new in mutations:
            with self.subTest(mutation=name):
                self.assertEqual(workflow.count(old), 1)
                self.assertFalse(self.accepts(workflow.replace(old, new, 1)))


if __name__ == "__main__":
    unittest.main()
