"""Tests for the CBZ-0 G11 census (tools/cbz/g11_census.py).

Every finding class is planted once in a synthetic tree, beside look-alikes that
must not be flagged; the mutation self-test (tools/cbz/g11_mutation_selftest.py)
requires these tests to fail when any detector is weakened.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "cbz"))
import g11_census as g11  # noqa: E402
import g11_mutation_selftest as g11m  # noqa: E402

EXPECTED_MUTATION_INVENTORY_SHA256 = "d1f564e77ab8446b9410e84f77b4d966a932a647335bcdb63ff4101b000d40a6"

DOCUMENTED_TESTED = {
    "tools/README.md": b"Run `tools/good.py` to check things.\n",
    "tests/test_good.py": b"import good\n",
}


def run(tree, all_paths=None):
    return g11.census({k: (v.encode() if isinstance(v, str) else v) for k, v in tree.items()}, all_paths)


def classes(report, path=None, area=None):
    return sorted((f["class"], f["line"]) for f in report["findings"]
                  if (path is None or f["path"] == path) and (area is None or f["area"] == area))


def stubs(text, path="tools/good.py"):
    return [(f["line"], f["detail"]) for f in run({path: text})["findings"]
            if f["class"] in ("stub", "stub-context")]


class Areas(unittest.TestCase):
    def test_area_of_each_path_family(self):
        cases = {"tools/a.py": "tooling", ".github/workflows/x.yml": "tooling",
                 "docs/site/app.mjs": "site", "docs/site/vendor/three.js": "vendor",
                 "tests/test_a.py": "tests", "tools/x/test_y.py": "tests", "conftest.py": "tests",
                 "docs/site/app.test.mjs": "tests", "experiments/e.py": "research",
                 "reviews/r/s.py": "evidence", "docs/reports/h.py": "evidence"}
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(g11.area(path), expected)


class PythonStubs(unittest.TestCase):
    def test_each_stub_body(self):
        self.assertEqual(stubs("def a():\n    pass\n"), [(1, "a: pass")])
        self.assertEqual(stubs("def a():\n    ...\n"), [(1, "a: ellipsis")])
        self.assertEqual(stubs("def a():\n    raise NotImplementedError\n"), [(1, "a: raise NotImplementedError")])
        self.assertEqual(stubs("def a():\n    raise NotImplementedError('later')\n"),
                         [(1, "a: raise NotImplementedError")])
        self.assertEqual(stubs('def a():\n    """Only a docstring."""\n'), [(1, "a: docstring only")])

    def test_docstring_then_stub_body(self):
        self.assertEqual(stubs('def a():\n    """Doc."""\n    pass\n'), [(1, "a: pass")])

    def test_methods_async_and_nested(self):
        text = ("class K:\n    def m(self):\n        pass\n"
                "async def b():\n    ...\n"
                "def outer():\n    def inner():\n        pass\n    return inner\n")
        self.assertEqual(stubs(text), [(2, "m: pass"), (4, "b: ellipsis"), (7, "inner: pass")])

    def test_real_bodies_are_not_stubs(self):
        for text in ("def a():\n    return 1\n",
                     'def a():\n    """Doc."""\n    return 1\n',
                     "def a(x):\n    if x:\n        pass\n    return x\n",
                     "def a():\n    pass\n    return 2\n",
                     "def a():\n    raise ValueError('bad')\n",
                     "class E(Exception):\n    pass\n"):
            with self.subTest(text=text):
                self.assertEqual(stubs(text), [])

    def test_legitimate_contexts_are_reported_but_do_not_fail(self):
        text = ("from abc import ABC, abstractmethod\nfrom typing import Protocol, overload\n"
                "class Base(ABC):\n    def m(self):\n        ...\n"
                "class Shape(Protocol):\n    def area(self) -> float:\n        ...\n"
                "class Bad(ValueError):\n    def __init__(self):\n        pass\n"
                "class C(ABC):\n    @abstractmethod\n    def a(self):\n        pass\n"
                "    @overload\n    def f(self, x: int) -> int:\n        ...\n"
                "    @overload\n    def f(self, x: str) -> str:\n        ...\n"
                "    def f(self, x):\n        return x\n")
        report = run({"tools/good.py": text, **DOCUMENTED_TESTED})
        contexts = sorted(f["detail"] for f in report["findings"] if f["class"] == "stub-context")
        self.assertEqual(contexts, ["a: pass; @abstractmethod",
                                    "area: ellipsis; method of a Protocol class",
                                    "f: ellipsis; @overload", "f: ellipsis; @overload"])
        self.assertEqual(sorted(f["detail"] for f in report["failing"]), ["__init__: pass", "m: ellipsis"])
        self.assertEqual(report["status"], "FAIL")

    def test_request_handler_log_message_override_is_legitimate(self):
        text = ("from http.server import SimpleHTTPRequestHandler\n"
                "class Handler(SimpleHTTPRequestHandler):\n    def log_message(self, *args):\n        pass\n"
                "    def do_GET(self):\n        pass\n"
                "class Other:\n    def log_message(self, *args):\n        pass\n")
        found = sorted((f["class"], f["line"]) for f in run({"tools/good.py": text})["findings"]
                       if f["class"].startswith("stub"))
        self.assertEqual(found, [("stub", 5), ("stub", 8), ("stub-context", 3)])

    def test_production_stub_fails_but_research_stub_does_not(self):
        bad = run({"tools/good.py": "def a():\n    pass\n", **DOCUMENTED_TESTED})
        self.assertEqual([(f["path"], f["class"]) for f in bad["failing"]], [("tools/good.py", "stub")])
        self.assertEqual(bad["status"], "FAIL")
        research = run({"experiments/e.py": "def a():\n    pass\n"})
        self.assertEqual(classes(research), [("stub", 1)])
        self.assertEqual(research["status"], "PASS")


class Markers(unittest.TestCase):
    def test_python_comment_is_marker_and_string_is_mention(self):
        text = ('def a():\n    # TODO: finish\n    return "TODO in data"\n'
                '# FIXME later\nx = 1  # XXX\n# TBD\n# STUB\n# not implemented yet\n')
        self.assertEqual(classes(run({"tools/good.py": text}), "tools/good.py"),
                         [("marker", 2), ("marker", 4), ("marker", 5), ("marker", 6), ("marker", 7),
                          ("marker", 8), ("mention", 3)])

    def test_lowercase_words_and_lookalikes_are_not_markers(self):
        text = "# todo list, xxx-large, stubs, tbd-free\nTODOS = 1  # TODOS are fine\n"
        self.assertEqual(classes(run({"tools/good.py": text}), "tools/good.py"), [])

    def test_docstring_marker_is_a_mention(self):
        self.assertEqual(classes(run({"tools/good.py": 'def a():\n    """TODO: doc."""\n    return 1\n'})),
                         [("mention", 2)])

    def test_javascript_comments_and_strings(self):
        text = ('// TODO top\nconst s = "TODO not a comment";\nconst u = "http://x"; // FIXME tail\n'
                "/* XXX\n   still TBD */ const t = 'STUB';\nconst r = `// TODO in template`;\n")
        self.assertEqual(classes(run({"docs/site/app.mjs": text})),
                         [("marker", 1), ("marker", 3), ("marker", 4), ("marker", 5),
                          ("mention", 2), ("mention", 5), ("mention", 6)])

    def test_escaped_quote_inside_string(self):
        text = 'const s = "a \\" // TODO still data";\n'
        self.assertEqual(classes(run({"docs/site/app.mjs": text})), [("mention", 1)])

    def test_shell_comments_and_strings(self):
        text = '#!/bin/sh\n# TODO setup\necho "TODO in data"  # FIXME\nn=${#TODO}\n'
        self.assertEqual(classes(run({"tools/good.sh": text})),
                         [("marker", 2), ("marker", 3), ("mention", 3), ("mention", 4)])

    def test_production_marker_fails(self):
        report = run({"tools/good.py": "# TODO\n", **DOCUMENTED_TESTED})
        self.assertEqual(report["status"], "FAIL")

    def test_site_marker_fails_and_vendor_marker_does_not(self):
        self.assertEqual(run({"docs/site/app.mjs": "// FIXME\n"})["status"], "FAIL")
        self.assertEqual(run({"docs/site/vendor/lib.js": "// FIXME\n"})["status"], "PASS")


class Workflows(unittest.TestCase):
    def test_noop_steps(self):
        text = ("jobs:\n  a:\n    steps:\n"
                "      - run: true\n"
                "      - name: colon\n        run: ':'\n"
                "      - run: exit 0  # placeholder\n"
                "      - name: block\n        run: |\n          # nothing\n          true\n        shell: bash\n"
                "      - run: |\n          true\n        shell: bash\n"
                "      - run: echo hi\n"
                "      - name: real block\n        run: |\n          true\n          make test\n")
        found = [f["line"] for f in run({".github/workflows/w.yml": text})["findings"] if f["class"] == "noop"]
        self.assertEqual(found, [4, 6, 7, 9, 13])

    def test_production_noop_fails(self):
        self.assertEqual(run({".github/workflows/w.yml": "      - run: true\n"})["status"], "FAIL")
        self.assertEqual(run({".github/workflows/w.yml": "      - run: make\n"})["status"], "PASS")


class ToolChecks(unittest.TestCase):
    def tools(self, tree, all_paths=None):
        return run(tree, all_paths)["tools"]

    def test_documented_and_exercised_by_path_import_and_basename(self):
        tree = {"tools/a.py": "x = 1\n", "tools/b.py": "x = 1\n", "tools/c.mjs": "export const x = 1;\n", "tools/d.py": "x = 1\n",
                "docs/USE.md": "See tools/a.py, b.py, c.mjs and d.py.\n",
                "tests/test_a.py": "from tools import helper_x, a\n",
                "tests/test_d.py": "import d\n",
                ".github/workflows/ci.yml": "      - run: python tools/b.py\n",
                "tests/c.test.mjs": "import { x } from '../tools/c.mjs';\n"}
        report = self.tools(tree)
        for path in ("tools/a.py", "tools/b.py", "tools/c.mjs", "tools/d.py"):
            with self.subTest(tool=path):
                self.assertTrue(report[path]["documented"])
                self.assertTrue(report[path]["referenced"])

    def test_undocumented_and_unexercised_tools_fail(self):
        result = run({"tools/lonely.py": "x = 1\n", "tools/README.md": "nothing here\n"})
        self.assertEqual(result["failing_tools"], ["tools/lonely.py"])
        self.assertFalse(result["tools"]["tools/lonely.py"]["documented"])
        self.assertFalse(result["tools"]["tools/lonely.py"]["referenced"])
        self.assertEqual(result["status"], "FAIL")

    def test_exercised_through_an_exercised_tool(self):
        tree = {"tools/top.py": "import helper\n", "tools/helper.py": "x = 1\n", "tools/orphan.py": "x = 1\n",
                "tools/README.md": "tools/top.py tools/helper.py tools/orphan.py\n",
                "tests/test_top.py": "import top\n"}
        report = self.tools(tree)
        self.assertEqual(report["tools/helper.py"]["referenced_by"], ["via tools/top.py"])
        self.assertFalse(report["tools/orphan.py"]["referenced"])

    def test_ambiguous_basename_needs_the_full_path(self):
        tree = {"tools/run.py": "x = 1\n", "docs/USE.md": "call run.py\n", "tests/test_x.py": "import run\n"}
        ambiguous = self.tools(tree, sorted(tree) + ["experiments/run.py"])
        self.assertFalse(ambiguous["tools/run.py"]["documented"])
        self.assertFalse(ambiguous["tools/run.py"]["referenced"])
        unique = self.tools(tree)
        self.assertTrue(unique["tools/run.py"]["documented"])
        self.assertTrue(unique["tools/run.py"]["referenced"])

    def test_full_path_counts_even_when_basename_is_ambiguous(self):
        tree = {"tools/run.py": "x = 1\n", "docs/USE.md": "call tools/run.py\n",
                ".github/workflows/ci.yml": "      - run: python tools/run.py\n"}
        report = self.tools(tree, sorted(tree) + ["experiments/run.py"])
        self.assertTrue(report["tools/run.py"]["documented"])
        self.assertTrue(report["tools/run.py"]["referenced"])

    def test_module_docstring_is_recorded(self):
        report = self.tools({"tools/a.py": '"""Does a."""\n', "tools/b.py": "x = 1\n"})
        self.assertEqual((report["tools/a.py"]["module_docstring"], report["tools/b.py"]["module_docstring"]),
                         (True, False))

    def test_tests_and_init_files_are_not_tools(self):
        report = self.tools({"tools/__init__.py": "", "tools/x/test_y.py": "x = 1\n", "tools/a.py": "x = 1\n"})
        self.assertEqual(sorted(report), ["tools/a.py"])


class FailClosed(unittest.TestCase):
    def test_undecodable_and_unparseable_production_files_fail(self):
        for raw, expected in ((b"x = '\xff'\n", "undecodable"), (b"def broken(:\n", "unparseable")):
            with self.subTest(raw=raw):
                report = run({"tools/good.py": raw, **DOCUMENTED_TESTED})
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual([f["class"] for f in report["failing"]], [expected])

    def test_unparseable_research_file_is_reported_not_failed(self):
        report = run({"experiments/old.py": b"print 'python 2'\n"})
        self.assertEqual(classes(report), [("unparseable", 0)])
        self.assertEqual(report["status"], "PASS")


class CommandLine(unittest.TestCase):
    def repo(self, files):
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", d]))
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@invalid",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@invalid")
        subprocess.run(["git", "init", "-q", d], check=True)
        for path, text in files.items():
            Path(d, path).parent.mkdir(parents=True, exist_ok=True)
            Path(d, path).write_bytes(text)
        subprocess.run(["git", "-C", d, "add", "-A"], check=True)
        subprocess.run(["git", "-C", d, "commit", "-q", "-m", "fixture"], check=True, env=env)
        return d

    def cli(self, root, *args):
        script = str(ROOT / "tools" / "cbz" / "g11_census.py")
        return subprocess.run([sys.executable, "-B", "-S", script, "--root", root, *args],
                              capture_output=True, text=True)

    def test_exit_codes_and_exact_committed_bytes(self):
        good = {"tools/good.py": b"x = 1\n", "tools/README.md": b"tools/good.py\n", "tests/test_good.py": b"import good\n"}
        root = self.repo(good)
        self.assertEqual(self.cli(root).returncode, 0)
        Path(root, "tools/good.py").write_bytes(b"def a():\n    pass\n")  # uncommitted: not read
        self.assertEqual(self.cli(root).returncode, 0)
        bad = self.repo({**good, "tools/good.py": b"def a():\n    pass\n"})
        result = self.cli(bad)
        self.assertEqual(result.returncode, 1)
        self.assertIn("tools/good.py:1: tooling stub (a: pass)", result.stdout)
        unexercised = self.repo({
            "tools/lonely.py": b"x = 1\n",
            "tools/README.md": b"tools/lonely.py\n",
        })
        unexercised_result = self.cli(unexercised)
        self.assertEqual(unexercised_result.returncode, 1)
        self.assertIn("tools/lonely.py: tool unexercised", unexercised_result.stdout)
        self.assertEqual(self.cli(root, "--rev", "no-such-revision").returncode, 2)




class AdversarialControls(unittest.TestCase):
    def test_decoy_names_do_not_exempt_stubs(self):
        text = (
            "from abc import abstractmethod as am\nimport typing as t\n"
            "def abstractmethod(fn): return fn\n"
            "class Fake:\n    @staticmethod\n    def abstractmethod(fn): return fn\n"
            "class Protocol: pass\nclass ABC: pass\nclass ValueError: pass\n"
            "@abstractmethod\ndef decoy_decorator():\n    pass\n"
            "@Fake.abstractmethod\ndef decoy_qualified():\n    pass\n"
            "class NotReallyError:\n    def fail(self):\n        pass\n"
            "class NotReallyProtocol:\n    def fail(self):\n        ...\n"
            "class NotReallyABC:\n    def fail(self):\n        pass\n"
            "class NotReallyRequestHandler:\n    def log_message(self, *args):\n        pass\n"
            "class DecoyProtocol(Protocol):\n    def proto(self):\n        pass\n"
            "class DecoyABC(ABC):\n    def abc(self):\n        pass\n"
            "class ShadowedBuiltin(ValueError):\n    def builtin(self):\n        pass\n"
            "class Genuine:\n    @am\n    def abstract(self):\n        pass\n"
            "class Shape(t.Protocol):\n    def area(self):\n        ...\n"
        )
        report = run({"tools/good.py": text, **DOCUMENTED_TESTED})
        self.assertEqual(sorted(f["detail"].split(":")[0] for f in report["failing"]),
                         ["abc", "abstract", "builtin", "decoy_decorator", "decoy_qualified",
                          "fail", "fail", "fail", "log_message", "proto"])
        self.assertEqual(report["status"], "FAIL")
        local_shadow = (
            "from abc import abstractmethod\n"
            "def factory():\n"
            "    abstractmethod = lambda fn: fn\n"
            "    @abstractmethod\n"
            "    def local_decoy():\n"
            "        pass\n"
        )
        local_report = run({"tools/good.py": local_shadow, **DOCUMENTED_TESTED})
        self.assertEqual([(f["class"], f["detail"]) for f in local_report["findings"]
                          if f["class"].startswith("stub")], [("stub", "local_decoy: pass")])

    def test_comment_and_string_mentions_do_not_exercise_tools(self):
        tree = {
            "tools/runner.py": "def main():\n    return 0\n",
            "tools/README.md": "Use tools/runner.py.\n",
            "tests/test_runner.py": "# tools/runner.py\nDOC = 'tools/runner.py'\nprint('tools/runner.py')\n",
        }
        report = run(tree)
        self.assertFalse(report["tools"]["tools/runner.py"]["referenced"])
        self.assertEqual(report["failing_tools"], ["tools/runner.py"])

        python_cmd = run({
            "tools/runner.py": "def main(): return 0\n",
            "tools/README.md": "Use tools/runner.py.\n",
            "tests/test_runner.py": "import subprocess\nsubprocess.run(\"python -c 'tools/runner.py'\", shell=True)\n",
        })
        self.assertFalse(python_cmd["tools"]["tools/runner.py"]["referenced"])

        workflow_text = "jobs:\n  test:\n    steps:\n      - run: \"sh -c 'echo tools/runner.py'\"\n"
        shell_cmd = run({
            "tools/runner.py": "def main(): return 0\n",
            "tools/README.md": "Use tools/runner.py.\n",
            ".github/workflows/ci.yml": workflow_text,
        })
        self.assertEqual(g11._workflow_commands(workflow_text)[0][1], ["sh -c 'echo tools/runner.py'"])
        self.assertFalse(shell_cmd["tools"]["tools/runner.py"]["referenced"])

        js_cmd = run({
            "tools/runner.mjs": "export const run = () => 1;\n",
            "tools/README.md": "Use tools/runner.mjs.\n",
            "tests/runner.test.mjs": "const sample = `\nimport '../tools/runner.mjs';\n`;\n",
        })
        self.assertFalse(js_cmd["tools"]["tools/runner.mjs"]["referenced"])

    def test_empty_and_comment_only_run_blocks_fail(self):
        text = ("jobs:\n  a:\n    steps:\n"
                "      - run: |\n          # only comment\n"
                "      - run: ''\n"
                "      - run: # empty\n"
                "      - run: true\n")
        found = [f["line"] for f in run({".github/workflows/w.yml": text})["findings"]
                 if f["class"] == "noop"]
        self.assertEqual(found, [4, 6, 7, 8])

    def test_yaml_comments_are_scanned_for_markers(self):
        text = ("name: ci\n# TODO workflow comment\njobs:\n  a:\n    steps:\n"
                "      - run: echo '# FIXME in data'\n")
        report = run({".github/workflows/w.yml": text})
        self.assertEqual(classes(report), [("marker", 2), ("mention", 6)])


class MutationAttribution(unittest.TestCase):
    def test_mutation_kill_requires_exact_target_assertion(self):
        expected = "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion"
        good = {"parsed": True, "status": "FAILED", "exit": 1, "ran": 1, "failures": 1,
                "errors": 0, "skipped": 0, "test_id": expected,
                "traceback_method": "test_mutation_kill_requires_exact_target_assertion",
                "traceback_line": 17, "exception_type": "AssertionError"}
        self.assertTrue(g11m.qualifies_kill(good, expected, 17))
        wrong_line = dict(good, traceback_line=18)
        self.assertFalse(g11m.qualifies_kill(wrong_line, expected, 17))
        wrong_test = dict(good, test_id=expected + ".other")
        self.assertFalse(g11m.qualifies_kill(wrong_test, expected, 17))
        not_failed = dict(good, status="OK")
        self.assertFalse(g11m.qualifies_kill(not_failed, expected, 17))
        has_errors = dict(good, errors=1)
        self.assertFalse(g11m.qualifies_kill(has_errors, expected, 17))
        has_skips = dict(good, skipped=1)
        self.assertFalse(g11m.qualifies_kill(has_skips, expected, 17))

        expected_parser = "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion"
        output = (
            "Ran 1 test in 0.001s\n\n"
            "FAIL: test_mutation_kill_requires_exact_target_assertion (tests.test_g11_census.MutationAttribution)\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_g11_census.py", line 123, in test_mutation_kill_requires_exact_target_assertion\n'
            "AssertionError: expected\n\n"
            "FAILED (failures=1)\n"
        )
        proc = subprocess.CompletedProcess(["python"], 1, output, "")
        parsed = g11m._parse_run(proc)
        self.assertEqual(parsed["test_id"], expected_parser)
        self.assertTrue(g11m.qualifies_kill(parsed, expected_parser, 123))





class ReviewResidualControls(unittest.TestCase):
    def test_stub_exemptions_are_immediate_and_scope_safe(self):
        text = (
            "from abc import ABC, abstractmethod\n"
            "from typing import Protocol, overload\n"
            "from http.server import BaseHTTPRequestHandler\n"
            "class Abstract(ABC):\n"
            "    @abstractmethod\n"
            "    def required(self): ...\n"
            "    def concrete(self): ...\n"
            "    class Nested:\n"
            "        def inner_abc(self): ...\n"
            "class Interface(Protocol):\n"
            "    def declared(self): ...\n"
            "    class Nested:\n"
            "        def inner_protocol(self): ...\n"
            "class Failure(Exception):\n"
            "    def exception_method(self): ...\n"
            "def build(ABC, Exception, BaseHTTPRequestHandler):\n"
            "    class ShadowedABC(ABC):\n"
            "        def shadowed_abc(self): ...\n"
            "    class ShadowedError(Exception):\n"
            "        def shadowed_exception(self): ...\n"
            "    class ShadowedHandler(BaseHTTPRequestHandler):\n"
            "        def shadowed_handler_log_message(self, *args): ...\n"
            "class Unpaired:\n"
            "    @abstractmethod\n"
            "    def abstract_without_abc(self): ...\n"
            "    @overload\n"
            "    def overload_without_implementation(self, value: int): ...\n"
        )
        report = run({"tools/good.py": text, **DOCUMENTED_TESTED})
        stub_names = {f["detail"].split(":")[0] for f in report["findings"] if f["class"] == "stub"}
        context_names = {f["detail"].split(":")[0] for f in report["findings"] if f["class"] == "stub-context"}
        self.assertIn("required", context_names)
        self.assertIn("declared", context_names)
        self.assertIn("inner_abc", stub_names)
        self.assertIn("shadowed_abc", stub_names)
        for name in ("concrete", "inner_abc", "inner_protocol", "exception_method",
                     "shadowed_abc", "shadowed_exception", "shadowed_handler_log_message",
                     "abstract_without_abc", "overload_without_implementation"):
            self.assertIn(name, stub_names)
            self.assertNotIn(name, context_names)

    def test_tool_reachability_and_heredoc_boundaries(self):
        uncalled = run({
            "tools/runner.py": "def main(): return 0\n",
            "tools/README.md": "Use tools/runner.py.\n",
            "tests/test_runner.py": "def unused_helper():\n    import runner\n",
        })
        self.assertFalse(uncalled["tools"]["tools/runner.py"]["referenced"])
        dead = run({
            "tools/runner.py": "def main(): return 0\n",
            "tools/README.md": "Use tools/runner.py.\n",
            "tests/test_runner.py": "if False:\n    import runner\n",
        })
        self.assertFalse(dead["tools"]["tools/runner.py"]["referenced"])
        reachable = run({
            "tools/runner.py": "def main(): return 0\n",
            "tools/README.md": "Use tools/runner.py.\n",
            "tests/test_runner.py": (
                "import unittest\n"
                "class RunnerTests(unittest.TestCase):\n"
                "    def test_reaches_helper(self): self.helper()\n"
                "    def helper(self):\n"
                "        import runner\n"
            ),
        })
        self.assertTrue(reachable["tools"]["tools/runner.py"]["referenced"])
        decoy = run({
            "tools/runner.py": "def main(): return 0\n",
            "tools/README.md": "Use tools/runner.py.\n",
            "tests/test_runner.py": (
                "class TestCase:\n    pass\n"
                "class Fake(TestCase):\n"
                "    def test_decoy(self):\n"
                "        import runner\n"
            ),
        })
        self.assertFalse(decoy["tools"]["tools/runner.py"]["referenced"])
        imported_alias = run({
            "tools/runner.py": "def main(): return 0\n",
            "tools/README.md": "Use tools/runner.py.\n",
            "tests/test_runner.py": (
                "from unittest import TestCase as BaseCase\n"
                "class RunnerTests(BaseCase):\n"
                "    def test_reaches_helper(self):\n"
                "        import runner\n"
            ),
        })
        self.assertTrue(imported_alias["tools"]["tools/runner.py"]["referenced"])
        workflow = ("jobs:\n  test:\n    steps:\n      - run: |\n"
                    "          cat <<'EOF'\n"
                    "          python tools/runner.py\n"
                    "          EOF\n")
        heredoc = run({
            "tools/runner.py": "def main(): return 0\n",
            "tools/README.md": "Use tools/runner.py.\n",
            ".github/workflows/ci.yml": workflow,
        })
        self.assertFalse(heredoc["tools"]["tools/runner.py"]["referenced"])
        real_command = run({
            "tools/runner.py": "def main(): return 0\n",
            "tools/README.md": "Use tools/runner.py.\n",
            ".github/workflows/ci.yml": workflow + "          python tools/runner.py\n",
        })
        self.assertTrue(real_command["tools"]["tools/runner.py"]["referenced"])

    def test_mutation_inventory_is_sealed(self):
        source = Path(g11m.__file__).read_bytes()
        start = source.index(b"MUTATIONS = [")
        end = source.index(b"\n\n\ndef _build_cases", start)
        self.assertEqual(hashlib.sha256(source[start:end]).hexdigest(),
                         EXPECTED_MUTATION_INVENTORY_SHA256)
        cases = g11m._build_cases()
        self.assertEqual(len(cases), 75)
        g11m._validate_mutation_inventory(cases)
        altered_cases = (cases[:-1], cases + [cases[0]], cases[1:] + cases[:1])
        for altered in altered_cases:
            with self.assertRaisesRegex(g11m.HarnessError, "mutation inventory"):
                g11m._validate_mutation_inventory(altered)
        weakened = list(cases)
        original = weakened[0]
        weakened[0] = (original[0], original[1], original[2],
                       original[3] + " # same-name weakening", original[4], original[5])
        with self.assertRaisesRegex(g11m.HarnessError, "mutation inventory mismatch"):
            g11m._validate_mutation_inventory(tuple(weakened))

    def test_mutation_child_environment_is_sanitized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "__init__.py").write_text("", encoding="utf-8")
            (tests_dir / "test_probe.py").write_text(
                "import os, unittest\n"
                "class Probe(unittest.TestCase):\n"
                "    def test_environment(self):\n"
                "        for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONPYCACHEPREFIX'):\n"
                "            self.assertNotIn(key, os.environ)\n"
                "        self.assertNotIn('poison', unittest.__file__)\n",
                encoding="utf-8")
            poison = root / "poison"
            poison.mkdir()
            (poison / "unittest.py").write_text("raise RuntimeError('poisoned unittest')\n", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(poison)
            env["PYTHONHOME"] = str(poison)
            env["PYTHONPYCACHEPREFIX"] = str(root / "pycache")
            outcome = g11m._run(str(root), "tests.test_probe", env=env)
            self.assertEqual(outcome.get("status"), "OK")
            self.assertEqual(outcome.get("ran"), 1)

    def test_in_scope_tracked_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "G11 Test"],
                           check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "g11@example.invalid"],
                           check=True, capture_output=True)
            target = root / "target.txt"
            target.write_text("not a scanned source\n", encoding="utf-8")
            link = root / "tools" / "cbz" / "linked.py"
            link.parent.mkdir(parents=True)
            link.symlink_to(target)
            subprocess.run(["git", "-C", str(root), "add", "tools/cbz/linked.py"],
                           check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "symlink fixture"],
                           check=True, capture_output=True)
            with self.assertRaisesRegex(ValueError, "in-scope non-regular tracked path"):
                g11.read_tree(str(root), "HEAD")


if __name__ == "__main__":
    unittest.main()