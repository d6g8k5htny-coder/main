import importlib.util
import json
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reference_crawler_mutation_check",
    ROOT / "tools" / "cbz" / "reference_crawler_mutation_check.py",
)
MUTATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MUTATION)


class AssertionOrigin(unittest.TestCase):
    def run_fixture(self, tool_source, test_source):
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
                return MUTATION.run_suite(tool)
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
                        "selected": {"filename": "/tmp/test.py", "firstlineno": 1, "name": test},
                        "frames": [
                            {
                                "filename": "/tmp/test.py",
                                "firstlineno": 1,
                                "name": test,
                                "qualname": "Fixture." + test,
                                "lineno": 2,
                                "selected": True,
                            }
                        ],
                    }
                    for test in tests
                ],
            }
        return {
            "schema": "reference-crawler-mutant-failmap/v2",
            "population": {
                "test_file_sha256": MUTATION.TEST_SHA256,
                "tests": list(MUTATION.EXPECTED_TESTS),
            },
            "mutants": mutants,
        }

    def test_failmap_accepts_origin_bound_entries(self):
        self.assertEqual(MUTATION.failmap_problems(self.origin_failmap()), [])

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
                    "schema": "reference-crawler-mutant-failmap-shard/v2",
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
        self.assertEqual(MUTATION.grader_mutation_selftest(), (3, 3))

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
