import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reference_crawler_mutation_check",
    ROOT / "tools" / "cbz" / "reference_crawler_mutation_check.py",
)
MUTATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MUTATION)


class AssertionOrigin(unittest.TestCase):
    def run_fixture(self, tool_source, test_source, expected_names=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool = root / "tool"
            (tool / "tools" / "cbz").mkdir(parents=True)
            (tool / "tools" / "cbz" / "reference_crawler.py").write_text(textwrap.dedent(tool_source))
            test = root / "test_fixture.py"
            test.write_text(textwrap.dedent(test_source))
            original = MUTATION.TEST
            MUTATION.TEST = test
            try:
                return MUTATION.run_suite(tool, expected_names)
            finally:
                MUTATION.TEST = original

    def test_direct_selected_body_assertion_is_authorized(self):
        outcome, _, failed, errored = self.run_fixture(
            "pass\n",
            """
            import unittest

            class Fixture(unittest.TestCase):
                def test_selected(self):
                    self.assertEqual(1, 2)
            """,
        )
        self.assertEqual(outcome, "fail")
        self.assertEqual(failed, {"test_selected"})
        self.assertEqual(errored, set())

    def test_subject_assertion_is_not_an_authorized_kill(self):
        outcome, _, failed, errored = self.run_fixture(
            """
            def subject():
                raise AssertionError("subject-origin")
            """,
            """
            import unittest
            from tools.cbz.reference_crawler import subject

            class Fixture(unittest.TestCase):
                def test_selected(self):
                    subject()
            """,
        )
        self.assertEqual((outcome, failed, errored), ("invalid", set(), set()))

    def test_spoofed_unittest_filename_is_not_an_authorized_kill(self):
        outcome, _, failed, errored = self.run_fixture(
            "pass\n",
            '''
            import unittest
            class Fixture(unittest.TestCase):
                def test_selected(self):
                    exec(compile("raise AssertionError('subject')", unittest.case.__file__, 'exec'))
            ''',
        )
        self.assertEqual((outcome, failed, errored), ("invalid", set(), set()))

    def test_child_receives_exact_parent_optimization(self):
        source = '''
            import sys, unittest
            class Fixture(unittest.TestCase):
                def test_selected(self):
                    self.assertEqual(sys.flags.optimize, %d)
        ''' % sys.flags.optimize
        outcome, detail, failed, errored = self.run_fixture("pass\n", source)
        self.assertEqual((outcome, detail, failed, errored), ("pass", "ran=1", set(), set()))

    def test_optimized_nested_parents_pass_mode_to_children(self):
        selector = ("tests.test_reference_crawler_mutation_check.AssertionOrigin."
                    "test_child_receives_exact_parent_optimization")
        env = dict(os.environ)
        env.pop("PYTHONOPTIMIZE", None)
        for flag in ("-O", "-OO"):
            with self.subTest(flag=flag):
                child = subprocess.run([sys.executable, "-B", flag, "-S", "-m", "unittest", "-v", selector],
                                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(child.returncode, 0, child.stdout + child.stderr)

    def test_declared_population_mismatch_is_invalid(self):
        outcome, _, failed, errored = self.run_fixture(
            "pass\n",
            '''
            import unittest
            class Fixture(unittest.TestCase):
                def test_selected(self):
                    pass
            ''',
            expected_names=["test_other"],
        )
        self.assertEqual((outcome, failed, errored), ("invalid", set(), set()))

    def test_declared_population_matches_by_unique_membership(self):
        source = '''
            import unittest
            class Fixture(unittest.TestCase):
                def test_a(self):
                    pass
                def test_b(self):
                    pass
        '''
        result = self.run_fixture("pass\n", source, expected_names=["test_b", "test_a"])
        self.assertEqual(result, ("pass", "ran=2", set(), set()))
        result = self.run_fixture("pass\n", source, expected_names=["test_a", "test_a"])
        self.assertEqual((result[0], result[2], result[3]), ("invalid", set(), set()))

    def test_equal_but_distinct_unittest_code_is_not_authorized(self):
        outcome, _, failed, errored = self.run_fixture(
            "pass\n",
            '''
            import types, unittest
            class Fixture(unittest.TestCase):
                def test_selected(self):
                    original = unittest.TestCase.fail
                    cloned = original.__code__.replace(co_filename=unittest.case.__file__)
                    types.FunctionType(cloned, original.__globals__)(self, 'subject')
            ''',
        )
        self.assertEqual((outcome, failed, errored), ("invalid", set(), set()))

    def test_skip_and_expected_failure_are_not_a_pass(self):
        for body, decorator in (("self.skipTest('unexecuted')", ""),
                                ("self.fail('expected')", "@unittest.expectedFailure"),
                                ("pass", "@unittest.expectedFailure")):
            source = '''
                import unittest
                class Fixture(unittest.TestCase):
                    %s
                    def test_selected(self):
                        %s
            ''' % (decorator, body)
            with self.subTest(body=body):
                outcome, _, failed, errored = self.run_fixture("pass\n", source)
                self.assertEqual((outcome, failed, errored), ("invalid", set(), set()))

    def test_skip_cannot_coexist_with_an_authorized_failure(self):
        outcome, _, failed, errored = self.run_fixture(
            "pass\n",
            '''
            import unittest
            class Fixture(unittest.TestCase):
                def test_a_failure(self):
                    self.fail('oracle')
                def test_b_skip(self):
                    self.skipTest('unexecuted')
            ''',
        )
        self.assertEqual((outcome, failed, errored), ("invalid", set(), set()))

    def origin_failmap(self):
        mutants = {}
        for name, tests in MUTATION.DESIGNATED.items():
            mutants[name] = {
                "fail": list(tests),
                "error": [],
                "provenance": [
                    {
                        "test": test,
                        "kind": "failure",
                        "authorized": True,
                        "exception": "AssertionError",
                        "selected": {"filename": "/tmp/test.py", "firstlineno": 1, "name": test},
                        "frames": [
                            {
                                "filename": "/tmp/test.py",
                                "firstlineno": 1,
                                "name": test,
                                "qualname": "Fixture." + test,
                                "lineno": 2,
                                "selected": True,
                                "oracle": False,
                            }
                        ],
                    }
                    for test in tests
                ],
            }
        return {
            "schema": "reference-crawler-mutant-failmap/v3",
            "execution": MUTATION.execution_binding(),
            "population": {
                "test_file_sha256": MUTATION.TEST_SHA256,
                "tests": list(MUTATION.EXPECTED_TESTS),
            },
            "mutants": mutants,
        }

    def test_failmap_accepts_origin_bound_entries(self):
        self.assertEqual(MUTATION.failmap_problems(self.origin_failmap()), [])

    def test_old_failmap_schema_is_not_rebound(self):
        doc = self.origin_failmap()
        doc["schema"] = "reference-crawler-mutant-failmap/v2"
        self.assertTrue(any("schema" in problem for problem in MUTATION.failmap_problems(doc)))

    def test_failmap_requires_exact_execution_binding(self):
        self.assertEqual(MUTATION.execution_binding().get("python_version"), sys.version)
        for bad in ({}, {"optimize": True}, {"optimize": 99},
                    {"crawler_sha256": "0" * 64}, {"grader_sha256": "0" * 64}):
            with self.subTest(bad=bad):
                doc = self.origin_failmap()
                doc["execution"] = bad
                self.assertTrue(any("execution" in problem for problem in MUTATION.failmap_problems(doc)))

    def test_failmap_requires_origin_schema_and_provenance(self):
        doc = self.origin_failmap()
        doc.pop("schema")
        doc["mutants"][next(iter(MUTATION.DESIGNATED))].pop("provenance")
        problems = MUTATION.failmap_problems(doc)
        self.assertTrue(any("schema" in problem for problem in problems), problems)
        self.assertTrue(any("provenance" in problem for problem in problems), problems)

    def test_failmap_rejects_unauthorized_provenance(self):
        doc = self.origin_failmap()
        name = next(iter(MUTATION.DESIGNATED))
        test = MUTATION.DESIGNATED[name][0]
        doc["mutants"][name]["provenance"].append(
            {"test": test, "kind": "failure", "authorized": False, "selected": {}, "frames": []}
        )
        self.assertTrue(
            any("unauthorized assertion failure" in problem for problem in MUTATION.failmap_problems(doc))
        )

    def test_failmap_rejects_errors_duplicates_and_bad_origin_shape(self):
        name = next(iter(MUTATION.DESIGNATED))
        test = MUTATION.DESIGNATED[name][0]
        for defect in ("errors", "extra_error_event", "duplicate", "bool_line", "wrong_frame", "wrong_exception"):
            with self.subTest(defect=defect):
                doc = self.origin_failmap()
                entry = doc["mutants"][name]
                event = entry["provenance"][0]
                if defect == "errors":
                    entry["error"] = [test]
                elif defect == "extra_error_event":
                    entry["provenance"].append({"test": test, "kind": "error", "authorized": False})
                elif defect == "duplicate":
                    entry["fail"].append(entry["fail"][0])
                elif defect == "bool_line":
                    event["selected"]["firstlineno"] = True
                elif defect == "wrong_frame":
                    event["frames"][0]["firstlineno"] = 2
                else:
                    event["exception"] = "RuntimeError"
                self.assertTrue(MUTATION.failmap_problems(doc), defect)

    def test_shard_partition_covers_each_mutant_once(self):
        names = [name for name, _, _ in MUTATION.MUTANTS]
        shards = [MUTATION.mutants_for_shard(MUTATION.MUTANTS, index, 3) for index in range(3)]
        flattened = [name for shard in shards for name, _, _ in shard]
        self.assertCountEqual(flattened, names)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual([len(shard) for shard in shards], [37, 36, 36])

    def test_merge_shard_failmaps_requires_and_combines_complete_partition(self):
        full = self.origin_failmap()
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(3):
                selected = MUTATION.mutants_for_shard(MUTATION.MUTANTS, index, 3)
                names = [name for name, _, _ in selected]
                shard = {
                    "schema": "reference-crawler-mutant-failmap-shard/v3",
                    "execution": full["execution"],
                    "population": full["population"],
                    "shard": {"index": index, "count": 3},
                    "mutants": {name: full["mutants"][name] for name in names},
                }
                path = Path(tmp) / f"shard-{index}.json"
                path.write_text(json.dumps(shard))
                paths.append(path)
            merged, problems = MUTATION.merge_shard_failmaps(paths)
            self.assertEqual(problems, [])
            self.assertEqual(MUTATION.failmap_problems(merged), [])
            _, incomplete = MUTATION.merge_shard_failmaps(paths[:-1])
            self.assertTrue(any("missing shard" in problem for problem in incomplete), incomplete)

    def test_grader_source_mutants_are_killed(self):
        self.assertTrue(hasattr(MUTATION, "grader_mutation_selftest"))
        self.assertEqual(MUTATION.grader_mutation_selftest(), (16, 16))

    def test_grader_selftest_refuses_forged_failure_text(self):
        method = "test_subject_assertion_is_not_an_authorized_kill"
        anchor = "def run_suite(tool_dir, expected_names=None, *, selector=None, copy_test=True):"
        mutant = ("probe", anchor, anchor.replace("copy_test=True", "copy_test=False"), method)
        fake = types.SimpleNamespace(returncode=1, stdout="FAIL: " + method + " fake\n", stderr="")
        with mock.patch.object(MUTATION, "GRADER_MUTANTS", (mutant,)):
            with mock.patch.object(MUTATION.subprocess, "run", return_value=fake):
                self.assertEqual(MUTATION.grader_mutation_selftest(), (0, 1))

    def test_mutant_fixture_copies_the_registered_acknowledgements(self):
        source = MUTATION.TOOL.read_text()
        with MUTATION.with_tool(source) as tmp:
            copied = Path(tmp) / "tools" / "cbz" / "reference_crawler_acknowledged.json"
            original = MUTATION.TOOL.with_name("reference_crawler_acknowledged.json")
            self.assertTrue(copied.is_file())
            self.assertEqual(copied.read_bytes(), original.read_bytes())

    def test_helper_assertion_is_not_an_authorized_kill(self):
        outcome, _, failed, errored = self.run_fixture(
            "pass\n",
            """
            import unittest

            def helper(case):
                case.assertEqual(1, 2)

            class Fixture(unittest.TestCase):
                def test_selected(self):
                    helper(self)
            """,
        )
        self.assertEqual((outcome, failed, errored), ("invalid", set(), set()))

    def test_nested_callback_assertion_is_not_an_authorized_kill(self):
        outcome, _, failed, errored = self.run_fixture(
            """
            def subject(callback):
                callback()
            """,
            """
            import unittest
            from tools.cbz.reference_crawler import subject

            class Fixture(unittest.TestCase):
                def callback(self):
                    self.assertEqual(1, 2)

                def test_selected(self):
                    subject(self.callback)
            """,
        )
        self.assertEqual((outcome, failed, errored), ("invalid", set(), set()))


if __name__ == "__main__":
    unittest.main()
