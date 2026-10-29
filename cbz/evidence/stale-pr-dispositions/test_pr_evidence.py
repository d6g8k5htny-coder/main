"""Tests for pr_evidence.py: signal detection, git analysis and the rule order."""

import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_evidence as pe  # noqa: E402


def run(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write(repo, path, text):
    p = Path(repo) / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


class LeanSignals(unittest.TestCase):
    def test_each_token_is_found(self):
        lines = ["theorem t : 1 = 1 := by sorry", "  admit", "axiom bad : False",
                 "example : 2 + 2 = 4 := by native_decide", "exact sorryAx _"]
        self.assertEqual(pe.lean_signals(lines),
                         {"sorry": 1, "admit": 1, "axiom": 1, "native_decide": 1, "sorryAx": 1})

    def test_line_comments_and_longer_identifiers_are_not_hits(self):
        lines = ["-- sorry, this is prose", "theorem admits_x : True := trivial",
                 "#print axioms foo", "def sorry_free := 1", "x.sorry"]
        self.assertEqual(pe.lean_signals(lines), {})

    def test_block_comments_are_reported_conservatively(self):
        # Block comments are not tracked, so a token inside one is still a signal.
        self.assertEqual(pe.lean_signals(["/- sorry -/"]), {"sorry": 1})


class SkipPatterns(unittest.TestCase):
    def test_every_skip_form_is_matched(self):
        forms = ["@unittest.skip('x')", "@unittest.skipIf(True, 'x')", "@unittest.skipUnless(False, 'x')",
                 "self.skipTest('x')", "@pytest.mark.skip(reason='x')", "pytest.skip('x')",
                 "@pytest.mark.xfail", "@unittest.expectedFailure"]
        for form in forms:
            with self.subTest(form=form):
                self.assertIsNotNone(pe.SKIP_RE.search(form))

    def test_each_pattern_is_needed(self):
        # Every pattern must be the only match for at least one form, so none is redundant.
        forms = ["@unittest.skip('x')", "self.skipTest('x')", "@pytest.mark.skip",
                 "pytest.skip('x')", "@pytest.mark.xfail", "@unittest.expectedFailure"]
        for pattern in pe.SKIP_PATTERNS:
            with self.subTest(pattern=pattern):
                self.assertTrue(any(pe.re.search(pattern, f) for f in forms))

    def test_ordinary_code_is_not_a_skip(self):
        self.assertIsNone(pe.SKIP_RE.search("skipped = compute()  # not a skip"))


class FailingChecks(unittest.TestCase):
    def test_only_unsuccessful_conclusions_count(self):
        runs = ["verify=completed/success", "a=completed/failure", "b=completed/skipped",
                "c=completed/timed_out", "d=completed/cancelled", "e=in_progress/-", "f=completed/neutral"]
        self.assertEqual(pe.failing_checks(runs), ["a=completed/failure", "c=completed/timed_out",
                                                   "d=completed/cancelled"])


class NotForMerge(unittest.TestCase):
    def test_explicit_statements_match(self):
        self.assertIsNotNone(pe.not_for_merge_evidence("G1-F1: fix (draft, do not merge)", ""))
        self.assertIsNotNone(pe.not_for_merge_evidence("x", "An author-side draft. Not for merge yet."))
        self.assertIsNotNone(pe.not_for_merge_evidence("x", "This branch must not be merged."))

    def test_descriptions_of_actions_do_not_match(self):
        for body in ("Integration is pending; no merge or deployment occurred.",
                     "The original author and this custodian do not merge.",
                     "No merge or scientific acceptance is authorized by the venue approval."):
            with self.subTest(body=body):
                self.assertIsNone(pe.not_for_merge_evidence("x", body))


class FloatSignals(unittest.TestCase):
    def test_float_constructs_are_counted(self):
        src = "import math\nx = 0.5\ny = float('1')\nz = math.sqrt(2)\nw = math.gcd(4, 6)\n"
        self.assertEqual(pe.float_signals(src), {"float_literals": 1, "float_calls": 1,
                                                 "inexact_math_calls": 1, "float_libraries": 0})

    def test_exact_code_has_no_signal(self):
        src = "from fractions import Fraction\nq = Fraction(1, 3) + 2\nn = 10 ** 6\n"
        self.assertFalse(any(pe.float_signals(src).values()))

    def test_numpy_import_is_counted(self):
        self.assertEqual(pe.float_signals("import numpy as np\n")["float_libraries"], 1)
        self.assertEqual(pe.float_signals("from scipy import special\n")["float_libraries"], 1)

    def test_line_filter_counts_only_given_lines(self):
        src = "a = 0.5\nb = 1\nc = 2.5\n"
        self.assertEqual(pe.float_signals(src, lines={3})["float_literals"], 1)
        self.assertEqual(pe.float_signals(src, lines={2})["float_literals"], 0)

    def test_unparseable_source_is_reported(self):
        self.assertIsNone(pe.float_signals("def (:\n"))


class GitAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        run(self.repo, "init", "-q", "-b", "main")
        run(self.repo, "config", "user.email", "t@example.invalid")
        run(self.repo, "config", "user.name", "t")
        write(self.repo, "tests/test_a.py", "def test_one():\n    pass\n\ndef test_two():\n    pass\n")
        write(self.repo, "README.md", "base\n")
        run(self.repo, "add", "-A")
        run(self.repo, "commit", "-q", "-m", "base")
        self.base = pe.rev(self.repo, "main")

    def tearDown(self):
        self.tmp.cleanup()

    def branch(self, name, edits, deletes=()):
        run(self.repo, "checkout", "-q", "-b", name, self.base)
        for path, text in edits.items():
            write(self.repo, path, text)
        for path in deletes:
            run(self.repo, "rm", "-q", path)
        run(self.repo, "add", "-A")
        run(self.repo, "commit", "-q", "-m", name)
        run(self.repo, "checkout", "-q", "main")
        return pe.rev(self.repo, name)

    def test_removed_and_skipped_tests_are_signals(self):
        head = self.branch("weaken", {"tests/test_a.py":
                                      "import unittest\n\n@unittest.skip('x')\ndef test_one():\n    pass\n"})
        paths = pe.changed_paths(self.repo, self.base, head)
        sig = pe.test_signals(self.repo, self.base, head, paths)
        self.assertEqual(sig["removed_test_functions"], ["tests/test_a.py::test_two"])
        self.assertEqual(len(sig["added_skips"]), 1)

    def test_api_local_head_mismatch_is_refused_before_analysis(self):
        head = self.branch("fresh", {"NEW.md": "fresh\n"})
        run(self.repo, "update-ref", "refs/cbz/pr/7", head)
        raw = {"pull": {"number": 7, "head": {"sha": self.base, "ref": "fresh"},
                         "base": {"sha": self.base, "ref": "main"}}}
        with self.assertRaisesRegex(ValueError, "head mismatch"):
            pe.analyze_one(self.repo, 7, raw, "main", {})

    def test_api_wrong_pr_number_is_refused(self):
        run(self.repo, "update-ref", "refs/cbz/pr/7", self.base)
        raw = {"pull": {"number": 8, "head": {"sha": self.base, "ref": "fresh"},
                         "base": {"sha": self.base, "ref": "main"}}}
        with self.assertRaisesRegex(ValueError, "number mismatch"):
            pe.analyze_one(self.repo, 7, raw, "main", {})

    def test_a_moved_test_function_is_not_a_removal(self):
        head = self.branch("reorder", {"tests/test_a.py":
                                       "def test_two():\n    pass\n\ndef test_one():\n    pass\n"})
        paths = pe.changed_paths(self.repo, self.base, head)
        self.assertEqual(pe.test_signals(self.repo, self.base, head, paths)["removed_test_functions"], [])

    def test_deleted_test_file_is_a_signal(self):
        head = self.branch("delete", {}, deletes=["tests/test_a.py"])
        paths = pe.changed_paths(self.repo, self.base, head)
        self.assertEqual(pe.test_signals(self.repo, self.base, head, paths)["deleted_test_files"],
                         ["tests/test_a.py"])

    def test_ancestry_positive_case(self):
        head = self.branch("landed", {"README.md": "landed\n"})
        run(self.repo, "merge", "-q", "--no-edit", "landed")
        self.assertTrue(pe.is_ancestor(self.repo, head, "main"))
        self.assertTrue(pe.is_ancestor(self.repo, self.base, "main"))

    def test_deleted_path_matches_only_when_target_also_lacks_it(self):
        head = self.branch("drop", {}, deletes=["README.md"])
        paths = pe.changed_paths(self.repo, self.base, head)
        self.assertEqual(pe.content_on_target(self.repo, head, "main", paths), {"README.md": False})
        run(self.repo, "rm", "-q", "README.md")
        run(self.repo, "commit", "-q", "-m", "drop on main")
        self.assertEqual(pe.content_on_target(self.repo, head, "main", paths), {"README.md": True})

    def test_content_already_on_target_and_ancestry(self):
        head = self.branch("feature", {"README.md": "changed\n"})
        self.assertFalse(pe.is_ancestor(self.repo, head, "main"))
        paths = pe.changed_paths(self.repo, self.base, head)
        self.assertEqual(pe.content_on_target(self.repo, head, "main", paths), {"README.md": False})
        # Land the same content on main in a separate commit: content matches, ancestry does not.
        write(self.repo, "README.md", "changed\n")
        run(self.repo, "commit", "-q", "-am", "same content")
        self.assertEqual(pe.content_on_target(self.repo, head, "main", paths), {"README.md": True})
        self.assertFalse(pe.is_ancestor(self.repo, head, "main"))

    def test_only_added_lines_count_for_float_signals(self):
        write(self.repo, "certificates/check.py", "x = 0.5\n")
        run(self.repo, "add", "-A")
        run(self.repo, "commit", "-q", "-m", "pre-existing float")
        base = pe.rev(self.repo, "main")
        run(self.repo, "checkout", "-q", "-b", "addfloat")
        write(self.repo, "certificates/check.py", "x = 0.5\ny = 1\nz = 2.5\n")
        write(self.repo, "certificates/clean.py", "from fractions import Fraction\nq = Fraction(1, 2)\n")
        run(self.repo, "add", "-A")
        run(self.repo, "commit", "-q", "-m", "add")
        run(self.repo, "checkout", "-q", "main")
        head = pe.rev(self.repo, "addfloat")
        self.assertEqual(pe.added_line_numbers(self.repo, base, head, "certificates/check.py"), {2, 3})
        sig = pe.python_float_signals(self.repo, base, head, pe.changed_paths(self.repo, base, head))
        self.assertEqual(set(sig), {"certificates/check.py"})
        self.assertEqual(sig["certificates/check.py"]["float_literals"], 1)
        self.assertTrue(sig["certificates/check.py"]["certificate_like"])
        self.assertEqual(pe.certificate_float(sig), sig)

    def test_merge_tree_reports_conflicts(self):
        head = self.branch("left", {"README.md": "left\n"})
        write(self.repo, "README.md", "right\n")
        run(self.repo, "commit", "-q", "-am", "right")
        clean, tree, conflicts = pe.merge_tree(self.repo, "main", head)
        self.assertFalse(clean)
        self.assertEqual(conflicts, ["README.md"])
        other = self.branch("other", {"NEW.md": "new\n"})
        clean, tree, conflicts = pe.merge_tree(self.repo, "main", other)
        self.assertTrue(clean)
        self.assertRegex(tree, r"^[0-9a-f]{40}$")
        self.assertEqual(conflicts, [])


def evidence(**overrides):
    ev = {"head_matches_api": True, "head_is_ancestor_of_target": False, "changed_paths": ["M\tx"],
          "content_on_target": {"x": False}, "base_ref": "main", "target_ref": "main",
          "parent_pr": None, "not_for_merge": None, "base_on_target": False, "merge_clean": True,
          "conflicts": [], "failing_checks": [], "layer_a_signals": {"lean": {}, "tests": {
              "removed_test_functions": [], "deleted_test_files": [], "added_skips": []},
              "python_float": {}}, "reviews": []}
    ev.update(overrides)
    return ev


class Publishable(unittest.TestCase):
    def test_free_text_is_dropped_and_verdict_class_kept(self):
        ev = evidence(title="t", reviews=[{"id": 1, "state": "COMMENTED", "commit": "c", "at_head": True,
                                           "headline": "PASS from some named model"}],
                      last_comments=[{"id": 2, "created_at": "x", "headline": "quoted text"}])
        out = pe.publishable(ev)
        self.assertEqual(out["reviews"], [{"id": 1, "state": "COMMENTED", "commit": "c",
                                           "at_head": True, "positive_verdict": True,
                                           "reviewer": {"id": None, "login": None}}])
        self.assertEqual(out["last_comments"], [{"id": 2, "created_at": "x"}])
        self.assertNotIn("headline", repr(out))

    def test_review_identity_preserved_in_public_projection(self):
        raw = {"pull": {"head": {"sha": "abc"}}, "reviews": [
            {"id": 1, "state": "COMMENTED", "commit_id": "abc", "body": "PASS_SCOPED",
             "user": {"id": 42, "login": "reviewer"}}]}
        reviews = pe.review_summary(raw)
        self.assertEqual(reviews[0]["reviewer"], {"id": 42, "login": "reviewer"})
        projected = pe.publishable(evidence(title="t", reviews=reviews, last_comments=[]))
        self.assertEqual(projected["reviews"][0]["reviewer"], {"id": 42, "login": "reviewer"})


class FetchBinding(unittest.TestCase):
    def test_existing_raw_is_refreshed_and_api_reread(self):
        pull = {"number": 7, "state": "open", "draft": False, "title": "x", "body": "",
                "created_at": "a", "updated_at": "b", "mergeable_state": "clean",
                "head": {"ref": "x", "sha": "a" * 40},
                "base": {"ref": "main", "sha": "b" * 40}}
        calls = []
        def fake_run(cmd, **kwargs):
            path = cmd[-1]
            calls.append(path)
            value = pull if path.endswith("/pulls/7") else (
                {"total_count": 0, "check_runs": []} if "/check-runs" in path else [])
            return subprocess.CompletedProcess(cmd, 0, stdout=pe.json.dumps(value))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "7.json").write_text('{"stale":true}')
            with patch.object(pe.subprocess, "run", side_effect=fake_run):
                pe.fetch("owner/repo", [7], out)
            data = pe.json.loads((out / "7.json").read_text())
            self.assertIn("pull", data)
            self.assertEqual(data["pull"]["head"]["sha"], "a" * 40)
            self.assertEqual(calls.count("repos/owner/repo/pulls/7"), 2)

    def test_head_change_during_fetch_preserves_previous_raw(self):
        pull = {"number": 7, "state": "open", "draft": False, "title": "x", "body": "",
                "created_at": "a", "updated_at": "b", "mergeable_state": "clean",
                "head": {"ref": "x", "sha": "a" * 40},
                "base": {"ref": "main", "sha": "b" * 40}}
        reads = 0
        def fake_run(cmd, **kwargs):
            nonlocal reads
            path = cmd[-1]
            if path.endswith("/pulls/7"):
                reads += 1
                value = dict(pull)
                if reads == 2:
                    value["head"] = {"ref": "x", "sha": "c" * 40}
            else:
                value = {"total_count": 0, "check_runs": []} if "/check-runs" in path else []
            return subprocess.CompletedProcess(cmd, 0, stdout=pe.json.dumps(value))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "7.json").write_text("original\n")
            with patch.object(pe.subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(ValueError, "changed during fetch"):
                    pe.fetch("owner/repo", [7], out)
            self.assertEqual((out / "7.json").read_text(), "original\n")


class RuleOrder(unittest.TestCase):
    def test_unbound_metadata_never_recommends_disposition(self):
        for fields in ({"head_matches_api": False}, {"head_matches_api": False,
                        "head_is_ancestor_of_target": True}):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "unbound"):
                pe.recommend(evidence(**fields))

    def test_block_and_amend_are_not_positive_verdicts(self):
        self.assertFalse(pe.pass_verdict("BLOCK after a partial PASS"))
        self.assertFalse(pe.pass_verdict("AMEND after PASS on prior head"))
    def test_each_rule_in_order(self):
        self.assertEqual(pe.recommend(evidence(head_is_ancestor_of_target=True))[0], "SUPERSEDED")
        self.assertEqual(pe.recommend(evidence(content_on_target={"x": True}))[0], "SUPERSEDED")
        self.assertEqual(pe.recommend(evidence(base_ref="feature", parent_pr=7))[0], "STACK-DEPENDENT")
        self.assertEqual(pe.recommend(evidence(base_ref="feature", base_on_target=True))[0],
                         "RETARGET-CANDIDATE")
        self.assertEqual(pe.recommend(evidence(base_ref="feature"))[0], "STACK-DEPENDENT")
        self.assertEqual(pe.recommend(evidence(not_for_merge="title: x (do not merge)"))[0], "QUARANTINE-ARCHIVE")
        self.assertEqual(pe.recommend(evidence(merge_clean=False, conflicts=["x"]))[0],
                         "QUARANTINE-ARCHIVE")
        lean = evidence()
        lean["layer_a_signals"]["lean"] = {"a.lean:sorry": 1}
        self.assertEqual(pe.recommend(lean)[0], "LAYER-A-REVIEW-FIRST")
        cert = evidence()
        cert["layer_a_signals"]["python_float"] = {"certificates/c.py": {"float_literals": 1,
                                                                          "certificate_like": True}}
        self.assertEqual(pe.recommend(cert)[0], "LAYER-A-REVIEW-FIRST")
        other = evidence()
        other["layer_a_signals"]["python_float"] = {"tools/plot.py": {"float_literals": 1,
                                                                        "certificate_like": False}}
        self.assertEqual(pe.recommend(other)[0], "INTEGRATE-CANDIDATE-UNREVIEWED")
        reviewed = evidence(reviews=[{"at_head": True, "headline": "PASS_SCOPED"}])
        self.assertEqual(pe.recommend(reviewed)[0], "INTEGRATE-CANDIDATE")
        stale = evidence(reviews=[{"at_head": False, "headline": "PASS_SCOPED"}])
        self.assertEqual(pe.recommend(stale)[0], "INTEGRATE-CANDIDATE-UNREVIEWED")
        self.assertEqual(pe.recommend(evidence(failing_checks=["verify=completed/failure"]))[0],
                         "REPAIR-OR-ARCHIVE")
        accepted = evidence(reviews=[{"at_head": True, "headline": "ACCEPT_SCOPED_SOURCE for x"}])
        self.assertEqual(pe.recommend(accepted)[0], "INTEGRATE-CANDIDATE")
        held = evidence(reviews=[{"at_head": True, "headline": "HOLD after PASS of part"}])
        self.assertEqual(pe.recommend(held)[0], "INTEGRATE-CANDIDATE-UNREVIEWED")
        failed = evidence(reviews=[{"at_head": True, "headline": "FAIL after PASS attempt"}])
        self.assertEqual(pe.recommend(failed)[0], "INTEGRATE-CANDIDATE-UNREVIEWED")

    def test_empty_change_set_is_not_superseded(self):
        # A PR with no changed paths must not count as already landed.
        self.assertNotEqual(pe.recommend(evidence(changed_paths=[], content_on_target={}))[0],
                            "SUPERSEDED")


if __name__ == "__main__":
    unittest.main()
