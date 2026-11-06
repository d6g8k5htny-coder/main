"""Tests for the CBZ-0 G11 census (tools/cbz/g11_census.py).

Every finding class is planted once in a synthetic tree, beside look-alikes that
must not be flagged; the mutation self-test (tools/cbz/g11_mutation_selftest.py)
requires these tests to fail when any detector is weakened.
"""
import ast
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

EXPECTED_MUTATION_INVENTORY_SHA256 = "f6de27b4f13e27b4187d6c2ef9bf1d7ebbd3f96f905f5b4f266b1a71fabc011e"

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
    def test_job_defaults_run_mapping_is_not_a_step(self):
        text = ("jobs:\n  test:\n    defaults:\n      run:\n        shell: bash\n"
                "    steps:\n      - run: make test\n")
        self.assertEqual(g11.noop_steps(text), [])

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
                "docs/USE.md": "See tools/a.py, tools/b.py, tools/c.mjs and d.py.\n",
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

    def test_reachable_python_path_values_are_execution_references(self):
        tree = {
            "tools/subprocess_tool.py": "x = 1\n",
            "tools/spec_tool.py": "x = 1\n",
            "tools/runpy_tool.py": "x = 1\n",
            "tools/env_tool.py": "x = 1\n",
            "tools/local_tool.py": "x = 1\n",
            "tools/legacy_tool.py": "x = 1\n",
            "tools/uncertain_tool.py": "x = 1\n",
            "tools/lazy_tool.py": "x = 1\n",
            "tools/wrapped_tool.py": "x = 1\n",
            "tools/README.md": (
                "tools/subprocess_tool.py tools/spec_tool.py tools/runpy_tool.py "
                "tools/env_tool.py tools/local_tool.py tools/legacy_tool.py "
                "tools/uncertain_tool.py tools/lazy_tool.py tools/wrapped_tool.py\n"
            ),
            "tests/test_paths.py": (
                "import importlib.util, os, runpy, subprocess, sys, unittest\n"
                "from pathlib import Path\n"
                "ROOT = Path(__file__).resolve().parents[1]\n"
                "CLI = ROOT / 'tools/subprocess_tool.py'\n"
                "SPEC = ROOT / 'tools/spec_tool.py'\n"
                "RUNPY = ROOT / 'tools/runpy_tool.py'\n"
                "ENV = Path(os.environ.get('G11_ENV_TOOL', ROOT / 'tools/env_tool.py'))\n"
                "HERE = os.path.dirname(os.path.abspath(__file__))\n"
                "WORKSPACE = os.path.dirname(HERE)\n"
                "LEGACY = os.path.join(WORKSPACE, 'tools', 'legacy_tool.py')\n"
                "UNCERTAIN_REAL = os.path.join(WORKSPACE, 'tools', 'uncertain_tool.py')\n"
                "UNCERTAIN = os.environ.get('G11_UNCERTAIN_TOOL') or UNCERTAIN_REAL\n"
                "LAZY = ROOT / 'tools/lazy_tool.py'\n"
                "WRAPPED = ROOT / 'tools/wrapped_tool.py'\n"
                "class Paths(unittest.TestCase):\n"
                "    def test_executes(self):\n"
                "        command = [sys.executable] + (['-O'] if sys.flags.optimize else []) + ['-B', str(CLI)]\n"
                "        subprocess.run(command)\n"
                "        importlib.util.spec_from_file_location('spec_tool', SPEC)\n"
                "        runpy.run_path(str(RUNPY), run_name='__main__')\n"
                "        subprocess.check_call([sys.executable, str(ENV)])\n"
                "        subprocess.run([sys.executable, LEGACY])\n"
                "        subprocess.run([sys.executable, UNCERTAIN])\n"
                "        local = ROOT / 'tools/local_tool.py'\n"
                "        importlib.util.spec_from_file_location('local_tool', local)\n"
                "        module = None\n"
                "        if module is None:\n"
                "            importlib.util.spec_from_file_location('lazy_tool', LAZY)\n"
                "        def launch(arguments):\n"
                "            return subprocess.run(arguments)\n"
                "        launch([sys.executable, str(WRAPPED)])\n"
            ),
        }
        report = self.tools(tree)
        referenced = {path for path, item in report.items() if item["referenced"]}
        self.assertEqual(referenced, {
            "tools/env_tool.py", "tools/lazy_tool.py", "tools/legacy_tool.py", "tools/local_tool.py",
            "tools/runpy_tool.py", "tools/wrapped_tool.py",
            "tools/spec_tool.py", "tools/subprocess_tool.py",
        })

    def test_none_guard_branches_do_not_combine_impossible_bindings(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_cache.py": (
                "import runpy, unittest\n"
                "prefix = 'other/'\n"
                "suffix = 'other.py'\n"
                "cache = None\n"
                "if cache is None:\n"
                "    prefix = 'tools/'\n"
                "else:\n"
                "    suffix = 'unused.py'\n"
                "class Cache(unittest.TestCase):\n"
                "    def test_load(self):\n"
                "        runpy.run_path(prefix + suffix)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_none_guard_branch_calls_use_isolated_bindings(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_cache.py": (
                "import runpy\n"
                "prefix = 'other/'\n"
                "cache = None\n"
                "if cache is None:\n"
                "    prefix = 'tools/'\n"
                "else:\n"
                "    runpy.run_path(prefix + 'unused.py')\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_none_guard_join_invalidates_alternative_values(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_cache.py": (
                "import runpy\n"
                "cache = None\n"
                "if cache is None:\n"
                "    script = 'other.py'\n"
                "else:\n"
                "    script = 'tools/unused.py'\n"
                "runpy.run_path(script)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_path_values_need_reachable_execution_not_mentions(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/decoy.py": "x = 1\n",
            "tools/dead.py": "x = 1\n",
            "tools/fake_loader.py": "x = 1\n",
            "tools/fake_wrapper.py": "x = 1\n",
            "tools/conditional.py": "x = 1\n",
            "tools/README.md": (
                "tools/live.py tools/decoy.py tools/dead.py tools/fake_loader.py "
                "tools/fake_wrapper.py tools/conditional.py\n"
            ),
            "tests/test_paths.py": (
                "import subprocess, sys, unittest\n"
                "from pathlib import Path\n"
                "ROOT = Path(__file__).resolve().parents[1]\n"
                "LIVE = ROOT / 'tools/live.py'\n"
                "DECOY = ROOT / 'tools/decoy.py'\n"
                "DEAD = ROOT / 'tools/dead.py'\n"
                "def unused():\n"
                "    subprocess.run([sys.executable, str(DEAD)])\n"
                "class Paths(unittest.TestCase):\n"
                "    def test_executes(self):\n"
                "        self.assertTrue(DECOY.name)\n"
                "        subprocess.run([sys.executable, str(LIVE)])\n"
            ),
            "tests/test_fake_loader.py": (
                "import importlib.util, sys, unittest\n"
                "from pathlib import Path\n"
                "ROOT = Path(__file__).resolve().parents[1]\n"
                "FAKE = ROOT / 'tools/fake_loader.py'\n"
                "FAKE_WRAPPER = ROOT / 'tools/fake_wrapper.py'\n"
                "CONDITIONAL = ROOT / 'tools/conditional.py'\n"
                "class Loader:\n"
                "    def spec_from_file_location(self, *args): return None\n"
                "    def run_path(self, *args): return None\n"
                "loader = Loader()\n"
                "class Paths(unittest.TestCase):\n"
                "    def test_decoys(self):\n"
                "        loader.spec_from_file_location('fake', FAKE)\n"
                "        loader.run_path(str(FAKE))\n"
                "        def label(arguments): return list(arguments)\n"
                "        label([sys.executable, str(FAKE_WRAPPER)])\n"
                "        if __import__('os').environ.get('G11_UNKNOWN_BRANCH'):\n"
                "            importlib.util.spec_from_file_location('conditional', CONDITIONAL)\n"
            ),
        }
        report = self.tools(tree)
        self.assertEqual({path: item["referenced"] for path, item in report.items()}, {
            "tools/conditional.py": False, "tools/dead.py": False, "tools/decoy.py": False,
            "tools/fake_loader.py": False, "tools/fake_wrapper.py": False, "tools/live.py": True,
        })

    def test_absolute_parent_paths_stop_at_the_root(self):
        imports = g11._import_bindings(ast.parse("from pathlib import Path\n"))
        within = ast.parse("Path('/tmp/project/tests/test_case.py').parents[1]", mode="eval").body
        beyond = ast.parse("Path('/tmp/project/tests/test_case.py').parents[99]", mode="eval").body
        self.assertEqual(g11._python_static_value(within, {}, {}, "tests/test_case.py", imports),
                         "/tmp/project")
        self.assertIsNone(g11._python_static_value(beyond, {}, {}, "tests/test_case.py", imports))

    def test_resolved_command_lists_reject_command_mode_arguments(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/argument.py": "x = 1\n",
            "tools/module_argument.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/argument.py tools/module_argument.py\n",
            "tests/test_commands.py": (
                "import subprocess, unittest\n"
                "class Commands(unittest.TestCase):\n"
                "    def test_commands(self):\n"
                "        subprocess.run(['python3', 'tools/live.py'])\n"
                "        subprocess.run(['python3', '-c', 'print(1)', 'tools/argument.py'])\n"
                "        subprocess.run(['python3', '-m', 'unittest', 'tools/module_argument.py'])\n"
            ),
        }
        report = self.tools(tree)
        self.assertEqual({path: item["referenced"] for path, item in report.items()}, {
            "tools/argument.py": False, "tools/live.py": True, "tools/module_argument.py": False,
        })

    def test_interpreter_mode_switch_after_script_is_program_argument(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/README.md": "tools/live.py\n",
            "tests/test_commands.py": (
                "import subprocess, unittest\n"
                "class Commands(unittest.TestCase):\n"
                "    def test_command(self):\n"
                "        subprocess.run(['python3', 'tools/live.py', '-m'])\n"
            ),
        }
        self.assertTrue(self.tools(tree)["tools/live.py"]["referenced"])

    def test_interpreter_option_operands_do_not_hide_the_script(self):
        tree = {
            "tools/direct.py": "x = 1\n",
            "tools/forwarded.py": "x = 1\n",
            "tools/README.md": "tools/direct.py tools/forwarded.py\n",
            "tests/test_commands.py": (
                "import subprocess, unittest\n"
                "def launch(command):\n"
                "    return subprocess.run(command)\n"
                "class Commands(unittest.TestCase):\n"
                "    def test_commands(self):\n"
                "        subprocess.run(['python3', '-W', 'ignore', 'tools/direct.py'])\n"
                "        launch(['python3', '-X', 'dev', 'tools/forwarded.py'])\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/direct.py": True, "tools/forwarded.py": True,
        })

    def test_unresolved_command_elements_do_not_shift_the_executable(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/starred.py": "x = 1\n",
            "tools/bound.py": "x = 1\n",
            "tools/inline.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/starred.py tools/bound.py tools/inline.py\n",
            "tests/test_commands.py": (
                "import shutil, subprocess, sys, unittest\n"
                "command = [shutil.which('echo'), 'tools/bound.py']\n"
                "flags = ['-B']\n"
                "class Commands(unittest.TestCase):\n"
                "    def test_commands(self):\n"
                "        subprocess.run(['python3', 'tools/live.py'])\n"
                "        subprocess.run([sys.executable, *flags, 'tools/starred.py'])\n"
                "        subprocess.run(command)\n"
                "        subprocess.run([shutil.which('echo'), 'tools/inline.py'])\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/bound.py": False, "tools/inline.py": False,
            "tools/live.py": True, "tools/starred.py": True,
        })

    def test_dynamic_command_suffix_does_not_hide_a_proven_tool_prefix(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/shifted.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/shifted.py\n",
            "tests/test_commands.py": (
                "import os, subprocess, sys, unittest\n"
                "LIVE = 'tools/live.py'\n"
                "SHIFTED = 'tools/shifted.py'\n"
                "class Commands(unittest.TestCase):\n"
                "    def test_commands(self):\n"
                "        suffix = os.environ.get('G11_DYNAMIC_SUFFIX')\n"
                "        subprocess.run([sys.executable, '-B', LIVE, suffix])\n"
                "        subprocess.run([sys.executable, suffix, SHIFTED])\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/live.py": True, "tools/shifted.py": False,
        })

    def test_unittest_lifecycle_methods_are_reachable(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/dead.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/dead.py\n",
            "tests/test_lifecycle.py": (
                "import importlib.util, unittest\n"
                "LIVE = 'tools/live.py'\n"
                "DEAD = 'tools/dead.py'\n"
                "def setUp():\n"
                "    importlib.util.spec_from_file_location('dead', DEAD)\n"
                "class Lifecycle(unittest.TestCase):\n"
                "    def setUp(self):\n"
                "        importlib.util.spec_from_file_location('live', LIVE)\n"
                "    def helper(self):\n"
                "        importlib.util.spec_from_file_location('dead', DEAD)\n"
                "    def test_body(self):\n"
                "        self.assertTrue(True)\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/dead.py": False, "tools/live.py": True,
        })

    def test_boolean_or_static_values_follow_short_circuit_order(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unknown.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unknown.py tools/unused.py\n",
            "tests/test_paths.py": (
                "import os, runpy, unittest\n"
                "UNUSED = 'other.py' or 'tools/unused.py'\n"
                "LIVE = '' or 'tools/live.py'\n"
                "UNKNOWN = os.environ.get('G11_UNKNOWN') or 'tools/unknown.py'\n"
                "class Paths(unittest.TestCase):\n"
                "    def test_paths(self):\n"
                "        runpy.run_path(UNUSED)\n"
                "        runpy.run_path(LIVE)\n"
                "        runpy.run_path(UNKNOWN)\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/live.py": True, "tools/unknown.py": False, "tools/unused.py": False,
        })

    def test_destructured_paths_bind_corresponding_values(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_bindings.py": (
                "import runpy, unittest\n"
                "UNUSED, LIVE = 'tools/unused.py', 'tools/live.py'\n"
                "class Bindings(unittest.TestCase):\n"
                "    def test_live(self):\n"
                "        runpy.run_path(LIVE)\n"
            ),
        }
        report = self.tools(tree)
        self.assertEqual({path: item["referenced"] for path, item in report.items()}, {
            "tools/live.py": True, "tools/unused.py": False,
        })

    def test_wrapper_process_call_must_be_reachable(self):
        tree = {
            "tools/dead.py": "x = 1\n",
            "tools/README.md": "tools/dead.py\n",
            "tests/test_wrapper.py": (
                "import subprocess, unittest\n"
                "class Wrapper(unittest.TestCase):\n"
                "    def test_wrapper(self):\n"
                "        def launch(arguments):\n"
                "            if False:\n"
                "                return subprocess.run(arguments)\n"
                "        launch(['python3', 'tools/dead.py'])\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/dead.py"]["referenced"])

    def test_wrapper_process_call_after_return_is_unreachable(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_wrapper_return.py": (
                "import subprocess\n"
                "def launch(command):\n"
                "    return print(command)\n"
                "    subprocess.run(command)\n"
                "launch(['python3', 'tools/unused.py'])\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_deferred_caller_sees_wrapper_defined_before_invocation(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/README.md": "tools/live.py\n",
            "tests/test_deferred_wrapper.py": (
                "import subprocess\n"
                "def exercise():\n"
                "    launch(['python3', 'tools/live.py'])\n"
                "def launch(command):\n"
                "    return subprocess.run(command)\n"
                "exercise()\n"
            ),
        }
        self.assertTrue(self.tools(tree)["tools/live.py"]["referenced"])

    def test_deferred_caller_uses_module_value_at_invocation(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_deferred_value.py": (
                "import runpy\n"
                "script = 'other.py'\n"
                "def exercise():\n"
                "    runpy.run_path(script)\n"
                "exercise()\n"
                "script = 'tools/unused.py'\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_nested_helper_effects_precede_caller_continuation(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_nested_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "def change():\n"
                "    command.clear()\n"
                "    command.extend(['echo'])\n"
                "def exercise():\n"
                "    change()\n"
                "    subprocess.run(command)\n"
                "exercise()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_nested_helper_global_assignment_updates_module_binding(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_global_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "def change():\n"
                "    global command\n"
                "    command = ['echo']\n"
                "change()\n"
                "subprocess.run(command)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_nested_helper_global_assignment_is_read_in_same_scope(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/README.md": "tools/live.py\n",
            "tests/test_global_read.py": (
                "import runpy\n"
                "script = 'other.py'\n"
                "def exercise():\n"
                "    global script\n"
                "    script = 'tools/live.py'\n"
                "    runpy.run_path(script)\n"
                "exercise()\n"
            ),
        }
        self.assertTrue(self.tools(tree)["tools/live.py"]["referenced"])

    def test_wrapper_rebinding_invalidates_forwarding(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_wrapper.py": (
                "import subprocess, unittest\n"
                "class Wrapper(unittest.TestCase):\n"
                "    def test_wrapper(self):\n"
                "        def launch(arguments):\n"
                "            arguments = ['echo', 'hello']\n"
                "            return subprocess.run(arguments)\n"
                "        launch(['python3', 'tools/unused.py'])\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_wrapper_element_write_invalidates_forwarding(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_wrapper.py": (
                "import subprocess, unittest\n"
                "def launch(command):\n"
                "    command[1] = 'other.py'\n"
                "    return subprocess.run(command)\n"
                "class Wrapper(unittest.TestCase):\n"
                "    def test_wrapper(self):\n"
                "        launch(['python3', 'tools/unused.py'])\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_command_element_delete_invalidates_aggregate_and_aliases(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_delete.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py', 'other.py']\n"
                "alias = command\n"
                "del command[1]\n"
                "subprocess.run(alias)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_wrapper_element_delete_invalidates_forwarding(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_wrapper_delete.py": (
                "import subprocess\n"
                "def launch(command):\n"
                "    del command[1]\n"
                "    subprocess.run(command)\n"
                "launch(['python3', 'tools/unused.py', 'other.py'])\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_wrapper_parameter_alias_mutation_invalidates_forwarding(self):
        tree = {
            "tools/write.py": "x = 1\n",
            "tools/mutator.py": "x = 1\n",
            "tools/README.md": "tools/write.py tools/mutator.py\n",
            "tests/test_wrapper_aliases.py": (
                "import subprocess, unittest\n"
                "def launch_write(command):\n"
                "    alias = command\n"
                "    alias[1] = 'other.py'\n"
                "    return subprocess.run(command)\n"
                "def launch_mutator(command):\n"
                "    alias = command\n"
                "    alias.clear()\n"
                "    alias.extend(['echo'])\n"
                "    return subprocess.run(command)\n"
                "class WrapperAliases(unittest.TestCase):\n"
                "    def test_aliases(self):\n"
                "        launch_write(['python3', 'tools/write.py'])\n"
                "        launch_mutator(['python3', 'tools/mutator.py'])\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/mutator.py": False, "tools/write.py": False,
        })

    def test_reassigned_wrapper_binding_is_not_execution_evidence(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_wrapper_reassigned.py": (
                "import subprocess\n"
                "def launch(command):\n"
                "    return subprocess.run(command)\n"
                "launch = print\n"
                "launch(['python3', 'tools/unused.py'])\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_later_nonforwarding_wrapper_definition_replaces_registration(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_wrapper_redefined.py": (
                "import subprocess\n"
                "def launch(command):\n"
                "    return subprocess.run(command)\n"
                "def launch(command):\n"
                "    return print(command)\n"
                "launch(['python3', 'tools/unused.py'])\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_helper_element_write_invalidates_module_binding(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_helper.py": (
                "import subprocess, unittest\n"
                "command = ['python3', 'tools/unused.py']\n"
                "def launch():\n"
                "    command[1] = 'other.py'\n"
                "    return subprocess.run(command)\n"
                "class Helper(unittest.TestCase):\n"
                "    def test_helper(self):\n"
                "        launch()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_command_alias_element_write_invalidates_all_aliases(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_alias.py": (
                "import subprocess, unittest\n"
                "command = ['python3', 'tools/unused.py']\n"
                "alias = command\n"
                "alias[1] = 'other.py'\n"
                "class Alias(unittest.TestCase):\n"
                "    def test_alias(self):\n"
                "        subprocess.run(command)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_command_mutator_calls_invalidate_aliases_and_forwarding(self):
        tree = {
            "tools/direct.py": "x = 1\n",
            "tools/forwarded.py": "x = 1\n",
            "tools/README.md": "tools/direct.py tools/forwarded.py\n",
            "tests/test_mutators.py": (
                "import subprocess, unittest\n"
                "command = ['python3', 'tools/direct.py']\n"
                "alias = command\n"
                "def launch(arguments):\n"
                "    arguments.clear()\n"
                "    arguments.extend(['echo'])\n"
                "    return subprocess.run(arguments)\n"
                "class Mutators(unittest.TestCase):\n"
                "    def test_mutators(self):\n"
                "        alias.clear()\n"
                "        alias.extend(['echo'])\n"
                "        subprocess.run(command)\n"
                "        launch(['python3', 'tools/forwarded.py'])\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/direct.py": False, "tools/forwarded.py": False,
        })

    def test_wrapper_forwarding_rejects_command_mode_arguments(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/argument.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/argument.py\n",
            "tests/test_wrapper.py": (
                "import subprocess, unittest\n"
                "def launch(arguments):\n"
                "    return subprocess.run(arguments)\n"
                "class Wrapper(unittest.TestCase):\n"
                "    def test_wrapper(self):\n"
                "        launch(['python3', 'tools/live.py'])\n"
                "        launch(['python3', '-c', 'print(1)', 'tools/argument.py'])\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/argument.py": False, "tools/live.py": True,
        })

    def test_import_loader_wrapper_forwards_a_proven_tool_path(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/dead.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/dead.py\n",
            "tests/test_wrapper.py": (
                "import importlib.util, unittest\n"
                "LIVE = 'tools/live.py'\n"
                "DEAD = 'tools/dead.py'\n"
                "def load(path):\n"
                "    return importlib.util.spec_from_file_location('loaded', path)\n"
                "def exercise(load):\n"
                "    load(DEAD)\n"
                "class Wrapper(unittest.TestCase):\n"
                "    def test_wrapper(self):\n"
                "        load(LIVE)\n"
                "        exercise(print)\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/dead.py": False, "tools/live.py": True,
        })

    def test_class_methods_do_not_register_as_module_wrappers(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_wrapper.py": (
                "import subprocess, unittest\n"
                "def launch(arguments):\n"
                "    return print(arguments)\n"
                "class Dormant:\n"
                "    @staticmethod\n"
                "    def launch(arguments):\n"
                "        return subprocess.run(arguments)\n"
                "class Wrapper(unittest.TestCase):\n"
                "    def test_wrapper(self):\n"
                "        subprocess.run(['python3', 'tools/live.py'])\n"
                "        launch(['python3', 'tools/unused.py'])\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/live.py": True, "tools/unused.py": False,
        })

    def test_conditional_command_lists_select_only_the_taken_branch(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_commands.py": (
                "import subprocess, unittest\n"
                "LIVE = ['tools/live.py'] if True else ['echo']\n"
                "UNUSED = ['tools/unused.py'] if False else ['echo']\n"
                "class Commands(unittest.TestCase):\n"
                "    def test_commands(self):\n"
                "        subprocess.run(LIVE)\n"
                "        subprocess.run(UNUSED)\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/live.py": True, "tools/unused.py": False,
        })

    def test_local_callee_shadows_do_not_inherit_module_wrappers(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_wrapper.py": (
                "import subprocess, unittest\n"
                "def launch(arguments):\n"
                "    return subprocess.run(arguments)\n"
                "def exercise(launch):\n"
                "    launch(['python3', 'tools/unused.py'])\n"
                "class Wrapper(unittest.TestCase):\n"
                "    def test_wrapper(self):\n"
                "        launch(['python3', 'tools/live.py'])\n"
                "        exercise(print)\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/live.py": True, "tools/unused.py": False,
        })

    def test_local_path_shadows_do_not_inherit_module_values(self):
        tree = {
            "tools/parameter.py": "x = 1\n",
            "tools/assignment.py": "x = 1\n",
            "tools/README.md": "tools/parameter.py tools/assignment.py\n",
            "tests/test_shadows.py": (
                "import runpy, unittest\n"
                "PARAMETER = 'tools/parameter.py'\n"
                "ASSIGNMENT = 'tools/assignment.py'\n"
                "def load_parameter(PARAMETER):\n"
                "    runpy.run_path(PARAMETER)\n"
                "def load_assignment():\n"
                "    ASSIGNMENT = input()\n"
                "    runpy.run_path(ASSIGNMENT)\n"
                "class Shadows(unittest.TestCase):\n"
                "    def test_shadows(self):\n"
                "        load_parameter('other.py')\n"
                "        load_assignment()\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/assignment.py": False, "tools/parameter.py": False,
        })

    def test_augmented_assignment_replaces_static_path_value(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_augmented.py": (
                "import runpy, unittest\n"
                "SCRIPT = 'tools/unused.py'\n"
                "SCRIPT += '.backup'\n"
                "class Augmented(unittest.TestCase):\n"
                "    def test_augmented(self):\n"
                "        runpy.run_path(SCRIPT)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_list_augmented_assignment_preserves_alias_identity(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_augmented_alias.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "alias = command\n"
                "command += ['argument']\n"
                "alias[1] = 'other.py'\n"
                "subprocess.run(command)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_command_element_assignment_invalidates_static_sequence(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_command.py": (
                "import subprocess, unittest\n"
                "command = ['python3', 'tools/unused.py']\n"
                "command[1] = 'other.py'\n"
                "class Command(unittest.TestCase):\n"
                "    def test_command(self):\n"
                "        subprocess.run(command)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_path_constructor_resolves_every_component(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_path.py": (
                "import runpy, unittest\n"
                "from pathlib import Path\n"
                "SCRIPT = Path('tools/unused.py', '/tmp/other.py')\n"
                "class PathUse(unittest.TestCase):\n"
                "    def test_path(self):\n"
                "        runpy.run_path(str(SCRIPT))\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_function_local_process_import_is_trusted(self):
        tree = {
            "tools/worker.py": "x = 1\n",
            "tools/README.md": "tools/worker.py\n",
            "tests/test_local.py": (
                "import unittest\n"
                "class Local(unittest.TestCase):\n"
                "    def test_local(self):\n"
                "        import subprocess\n"
                "        subprocess.run(['python3', 'tools/worker.py'])\n"
            ),
        }
        self.assertTrue(self.tools(tree)["tools/worker.py"]["referenced"])

    def test_sibling_tool_path_execution_is_a_reference(self):
        tree = {
            "tools/family/worker.py": "x = 1\n",
            "tools/README.md": "tools/family/worker.py\n",
            "tools/family/test_worker.py": (
                "import importlib.util, unittest\n"
                "from pathlib import Path\n"
                "class Worker(unittest.TestCase):\n"
                "    def test_worker(self):\n"
                "        path = Path(__file__).with_name('worker.py')\n"
                "        importlib.util.spec_from_file_location('worker', path)\n"
            ),
        }
        report = self.tools(tree)
        self.assertTrue(report["tools/family/worker.py"]["referenced"])

    def test_js_url_child_process_execution_is_a_reference(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/decoy.mjs": "export const decoy = true;\n",
            "tools/fake.mjs": "export const fake = true;\n",
            "tools/README.md": "tools/live.mjs tools/decoy.mjs tools/fake.mjs\n",
            "tests/live.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "import {fileURLToPath} from 'node:url';\n"
                "const script=fileURLToPath(new URL('../tools/live.mjs',import.meta.url));\n"
                "const decoy=fileURLToPath(new URL('../tools/decoy.mjs',import.meta.url));\n"
                "spawnSync(process.execPath,[script],{encoding:'utf8'});\n"
                "spawnSync('echo', ['hello']); String(decoy);\n"
            ),
            "tests/fake.test.mjs": (
                "const script=new URL('../tools/fake.mjs',import.meta.url);\n"
                "function spawnSync() {}\n"
                "spawnSync(process.execPath,[script]);\n"
            ),
        }
        report = self.tools(tree)
        self.assertEqual({path: item["referenced"] for path, item in report.items()}, {
            "tools/decoy.mjs": False, "tools/fake.mjs": False, "tools/live.mjs": True,
        })

    def test_js_quoted_and_property_names_do_not_credit_bindings(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/quoted.mjs": "export const quoted = true;\n",
            "tools/property.mjs": "export const property = true;\n",
            "tools/key.mjs": "export const key = true;\n",
            "tools/README.md": (
                "tools/live.mjs tools/quoted.mjs tools/property.mjs tools/key.mjs\n"
            ),
            "tests/expressions.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "const quoted=new URL('../tools/quoted.mjs',import.meta.url);\n"
                "const property=new URL('../tools/property.mjs',import.meta.url);\n"
                "const key=new URL('../tools/key.mjs',import.meta.url);\n"
                "spawnSync(process.execPath,[live, 'quoted', holder.property, {key: 'value'}]);\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/key.mjs": False, "tools/live.mjs": True,
            "tools/property.mjs": False, "tools/quoted.mjs": False,
        })

    def test_js_property_calls_do_not_use_named_child_process_imports(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/property.mjs": "export const property = true;\n",
            "tools/README.md": "tools/live.mjs tools/property.mjs\n",
            "tests/property-call.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "const property=new URL('../tools/property.mjs',import.meta.url);\n"
                "const fake={spawnSync(value){return value;}};\n"
                "spawnSync(process.execPath,[live]);\n"
                "fake.spawnSync(property);\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/live.mjs": True, "tools/property.mjs": False,
        })

    def test_js_quoted_call_text_does_not_execute_bindings(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/live.mjs tools/unused.mjs\n",
            "tests/quoted-call.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "const unused=new URL('../tools/unused.mjs',import.meta.url);\n"
                "spawnSync(process.execPath,[live]);\n"
                "console.log('spawnSync(unused)');\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/live.mjs": True, "tools/unused.mjs": False,
        })

    def test_js_reassigned_url_binding_does_not_keep_its_old_path(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/live.mjs tools/unused.mjs\n",
            "tests/reassigned.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "let script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "script='other.mjs';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "spawnSync(process.execPath,[script, live]);\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/live.mjs": True, "tools/unused.mjs": False,
        })

    def test_js_parameter_shadows_named_child_process_import(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/live.mjs tools/unused.mjs\n",
            "tests/shadow.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "const unused=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function exercise(spawnSync) { spawnSync(process.execPath,[unused]); }\n"
                "exercise(() => {});\n"
                "spawnSync(process.execPath,[live]);\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/live.mjs": True, "tools/unused.mjs": False,
        })

    def test_js_local_declaration_shadows_named_child_process_import(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/local-shadow.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function exercise() {\n"
                "  spawnSync('echo',[script]);\n"
                "  const spawnSync=(...args) => console.log(args);\n"
                "}\n"
                "exercise();\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_enclosing_declaration_shadows_named_child_process_import(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/enclosing-shadow.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function outer() {\n"
                "  const spawnSync=(...args) => console.log(args);\n"
                "  function inner() { spawnSync('echo',[script]); }\n"
                "  inner();\n"
                "}\n"
                "outer();\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_default_initializer_reference_is_not_a_parameter_binding(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/README.md": "tools/live.mjs\n",
            "tests/default.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/live.mjs',import.meta.url);\n"
                "function exercise(callback = spawnSync) {\n"
                "  spawnSync(process.execPath,[script]);\n"
                "}\n"
                "exercise();\n"
            ),
        }
        self.assertTrue(self.tools(tree)["tools/live.mjs"]["referenced"])

    def test_js_parameter_shadows_url_binding(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/shadow-url.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function exercise(script) {\n"
                "  spawnSync('echo',[script]);\n"
                "}\n"
                "exercise('other.mjs');\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_url_binding_does_not_escape_function_scope(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/url-scope.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script='other.mjs';\n"
                "function unused() {\n"
                "  const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "}\n"
                "spawnSync('echo',[script]);\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_unreachable_child_process_calls_are_not_references(self):
        tree = {
            "tools/branch.mjs": "export const branch = true;\n",
            "tools/helper.mjs": "export const helper = true;\n",
            "tools/README.md": "tools/branch.mjs tools/helper.mjs\n",
            "tests/unreachable.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const branch=new URL('../tools/branch.mjs',import.meta.url);\n"
                "const helper=new URL('../tools/helper.mjs',import.meta.url);\n"
                "if (false) { spawnSync(process.execPath,[branch]); }\n"
                "function unused() { spawnSync(process.execPath,[helper]); }\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/branch.mjs": False, "tools/helper.mjs": False,
        })

    def test_js_uncalled_helper_chain_is_not_reachable(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/helper-chain.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function launch() { spawnSync(process.execPath,[script]); }\n"
                "function unused() { launch(); }\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_rebound_helper_is_not_reachable(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/helper-rebind.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function launch() { spawnSync('echo',[script]); }\n"
                "launch = () => {};\n"
                "launch();\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_lexically_shadowed_helper_is_not_reachable(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/helper-shadow.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function launch() { spawnSync('echo',[script]); }\n"
                "function exercise() { const launch = () => {}; launch(); }\n"
                "exercise();\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_node_test_callback_is_a_reachable_root(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/live.mjs tools/unused.mjs\n",
            "tests/node-callback.test.mjs": (
                "import test from 'node:test';\n"
                "import {spawnSync} from 'node:child_process';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "const unused=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function launch() { spawnSync(process.execPath,[live]); }\n"
                "function fake() { spawnSync(process.execPath,[unused]); }\n"
                "test('reachable',()=>{ launch(); });\n"
                "notTest('untrusted',()=>{ fake(); });\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/live.mjs": True, "tools/unused.mjs": False,
        })

    def test_js_node_test_registration_must_be_reachable(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/node-callback.test.mjs": (
                "import test from 'node:test';\n"
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function unused() {\n"
                "  test('unreachable',()=>{ spawnSync('echo',[script]); });\n"
                "}\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_node_test_registrar_must_resolve_to_import(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/node-callback.test.mjs": (
                "import test from 'node:test';\n"
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function exercise(test) {\n"
                "  test('shadowed',()=>{ spawnSync('echo',[script]); });\n"
                "}\n"
                "exercise(()=>{});\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

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
            "from abc import ABC as RealABC, abstractmethod as am\nimport typing as t\n"
            "def abstractmethod(fn): return fn\n"
            "class Fake:\n    @staticmethod\n    def abstractmethod(fn): return fn\n"
            "class Protocol: pass\nclass ABC: pass\nclass ValueError: pass\n"
            "@abstractmethod\ndef decoy_decorator():\n    pass\n"
            "@Fake.abstractmethod\ndef decoy_qualified():\n    pass\n"
            "class NotReallyError:\n    def fail(self):\n        pass\n"
            "class NotReallyProtocol:\n    def fail(self):\n        ...\n"
            "class NotReallyABC:\n    def fail(self):\n        pass\n"
            "class StreamRequestHandler: pass\n"
            "class NotReallyRequestHandler(StreamRequestHandler):\n    def log_message(self, *args):\n        pass\n"
            "class DecoyProtocol(Protocol):\n    def proto(self):\n        pass\n"
            "class DecoyABC(ABC):\n    def abc(self):\n        pass\n"
            "class ShadowedBuiltin(ValueError):\n    def builtin(self):\n        pass\n"
            "class Genuine:\n    @am\n    def abstract(self):\n        pass\n"
            "class Shape(t.Protocol):\n    def area(self):\n        ...\n"
            "class RealBase(RealABC):\n    @abstractmethod\n    def decoy_method(self):\n        pass\n"
        )
        report = run({"tools/good.py": text, **DOCUMENTED_TESTED})
        self.assertEqual(sorted(f["detail"].split(":")[0] for f in report["failing"]),
                         ["abc", "abstract", "builtin", "decoy_decorator", "decoy_method", "decoy_qualified",
                          "fail", "fail", "fail", "log_message", "proto"])
        self.assertEqual(report["status"], "FAIL")
        local_shadow = (
            "from abc import abstractmethod\n"
            "from typing import overload\n"
            "def factory():\n"
            "    abstractmethod = lambda fn: fn\n"
            "    @abstractmethod\n"
            "    def local_decoy():\n"
            "        pass\n"
            "def overloads():\n"
            "    overload = lambda fn: fn\n"
            "    @overload\n"
            "    def local_overload(x: int): ...\n"
            "    def local_overload(x):\n"
            "        return x\n"
        )
        local_report = run({"tools/good.py": local_shadow, **DOCUMENTED_TESTED})
        self.assertEqual([(f["class"], f["detail"]) for f in local_report["findings"]
                          if f["class"].startswith("stub")],
                         [("stub", "local_decoy: pass"), ("stub", "local_overload: ellipsis")])

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

        workflow_text = ("jobs:\n  test:\n    steps:\n      - run: \"sh -c 'echo tools/runner.py'\"\n"
                         "      - run: python3 -c tools/runner.py\n")
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
        current = output.replace("(tests.test_g11_census.MutationAttribution)", "(%s)" % expected_parser)
        parsed_current = g11m._parse_run(subprocess.CompletedProcess(["python"], 1, current, ""))
        self.assertEqual(parsed_current["test_id"], expected_parser)
        subtest = current.replace("(%s)\n" % expected_parser, "(%s) (tool='tools/d.py')\n" % expected_parser)
        parsed_subtest = g11m._parse_run(subprocess.CompletedProcess(["python"], 1, subtest, ""))
        self.assertEqual(parsed_subtest["test_id"], expected_parser)





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
            "        @abstractmethod\n"
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
        self.assertEqual(len(cases), 151)
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
