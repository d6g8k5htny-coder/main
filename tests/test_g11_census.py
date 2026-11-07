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

EXPECTED_MUTATION_INVENTORY_SHA256 = "0bdb4d9232d4d14567c03ab185e0454654357d4ff364a5057fc5ae0d08c9e776"

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

    def test_interpreter_attached_and_exit_only_options_do_not_select_scripts(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/attached-unused.py": "x = 1\n",
            "tools/version-unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/attached-unused.py tools/version-unused.py\n",
            "tests/test_commands.py": (
                "import subprocess\n"
                "subprocess.run(['python3', '-B', 'tools/live.py'])\n"
                "subprocess.run(['python3', '-cpass', 'tools/attached-unused.py'])\n"
                "subprocess.run(['python3', '--version', 'tools/version-unused.py'])\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/attached-unused.py": False, "tools/live.py": True,
             "tools/version-unused.py": False},
        )

    def test_subprocess_executable_override_blocks_command_credit(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/direct-unused.py": "x = 1\n",
            "tools/dynamic-unused.py": "x = 1\n",
            "tools/kwargs-unused.py": "x = 1\n",
            "tools/wrapper-unused.py": "x = 1\n",
            "tools/README.md": (
                "tools/live.py tools/direct-unused.py tools/dynamic-unused.py "
                "tools/kwargs-unused.py tools/wrapper-unused.py\n"
            ),
            "tests/test_executable_override.py": (
                "import subprocess\n"
                "def launch(command):\n"
                "    subprocess.run(command, executable='/bin/echo')\n"
                "subprocess.run(['python3', 'tools/direct-unused.py'], "
                "executable='/bin/echo')\n"
                "override = '/bin/echo'\n"
                "subprocess.run(['python3', 'tools/dynamic-unused.py'], "
                "executable=override)\n"
                "options = {'executable': '/bin/echo'}\n"
                "subprocess.run(['python3', 'tools/kwargs-unused.py'], **options)\n"
                "launch(['python3', 'tools/wrapper-unused.py'])\n"
                "subprocess.run(['python3', 'tools/live.py'], executable=None)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/direct-unused.py": False, "tools/dynamic-unused.py": False,
             "tools/kwargs-unused.py": False, "tools/live.py": True,
             "tools/wrapper-unused.py": False},
        )

    def test_subprocess_shell_mode_blocks_command_credit(self):
        tree = {
            "tools/dynamic-unused.py": "x = 1\n",
            "tools/kwargs-unused.py": "x = 1\n",
            "tools/live.py": "x = 1\n",
            "tools/positional-unused.py": "x = 1\n",
            "tools/shell-unused.py": "x = 1\n",
            "tools/wrapper-unused.py": "x = 1\n",
            "tools/README.md": (
                "tools/dynamic-unused.py tools/kwargs-unused.py tools/live.py "
                "tools/positional-unused.py tools/shell-unused.py "
                "tools/wrapper-unused.py\n"
            ),
            "tests/test_shell_mode.py": (
                "import subprocess\n"
                "def launch(command):\n"
                "    subprocess.run(command, shell=True)\n"
                "subprocess.run(['python3', 'tools/shell-unused.py'], shell=True)\n"
                "mode = True\n"
                "subprocess.run(['python3', 'tools/dynamic-unused.py'], shell=mode)\n"
                "options = {'shell': True}\n"
                "subprocess.run(['python3', 'tools/kwargs-unused.py'], **options)\n"
                "subprocess.Popen(['python3', 'tools/positional-unused.py'], -1)\n"
                "launch(['python3', 'tools/wrapper-unused.py'])\n"
                "subprocess.run(['python3', 'tools/live.py'], shell=False)\n"
                "subprocess.run(['python3', 'tools/live.py'], shell=None)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/dynamic-unused.py": False, "tools/kwargs-unused.py": False,
             "tools/live.py": True, "tools/positional-unused.py": False,
             "tools/shell-unused.py": False, "tools/wrapper-unused.py": False},
        )

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

    def test_unknown_boolean_operand_effects_invalidate_downstream_bindings(self):
        tree = {
            "tools/direct-unused.py": "x = 1\n",
            "tools/live.py": "x = 1\n",
            "tools/wrapper-unused.py": "x = 1\n",
            "tools/README.md": (
                "tools/direct-unused.py tools/live.py tools/wrapper-unused.py\n"
            ),
            "tests/test_boolean_effects.py": (
                "import subprocess\n"
                "flag = True\n"
                "direct = ['python3', 'tools/direct-unused.py']\n"
                "flag and direct.__setitem__(slice(None), ['echo'])\n"
                "subprocess.run(direct)\n"
                "live = ['python3', 'tools/live.py']\n"
                "False and live.clear()\n"
                "subprocess.run(live)\n"
                "def launch(command):\n"
                "    flag and command.__setitem__(slice(None), ['echo'])\n"
                "    subprocess.run(command)\n"
                "launch(['python3', 'tools/wrapper-unused.py'])\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/direct-unused.py": False, "tools/live.py": True,
             "tools/wrapper-unused.py": False},
        )

    def test_nested_helper_reads_nearest_enclosing_static_binding(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_closure.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "def exercise():\n"
                "    script = 'tools/live.py'\n"
                "    def launch():\n"
                "        runpy.run_path(script)\n"
                "    launch()\n"
                "exercise()\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.py": True, "tools/unused.py": False},
        )

    def test_python_helper_requires_bindable_arguments(self):
        tree = {
            "tools/direct-unused.py": "x = 1\n",
            "tools/live.py": "x = 1\n",
            "tools/method-unused.py": "x = 1\n",
            "tools/README.md": (
                "tools/direct-unused.py tools/live.py tools/method-unused.py\n"
            ),
            "tests/test_helper_binding.py": (
                "import runpy, unittest\n"
                "def launch(required):\n"
                "    runpy.run_path('tools/direct-unused.py')\n"
                "def live(optional=None):\n"
                "    runpy.run_path('tools/live.py')\n"
                "try:\n"
                "    launch()\n"
                "except TypeError:\n"
                "    pass\n"
                "live()\n"
                "class Commands(unittest.TestCase):\n"
                "    def launch(self, required):\n"
                "        runpy.run_path('tools/method-unused.py')\n"
                "    def test_binding(self):\n"
                "        try:\n"
                "            self.launch()\n"
                "        except TypeError:\n"
                "            pass\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/direct-unused.py": False, "tools/live.py": True,
             "tools/method-unused.py": False},
        )

    def test_str_conversion_does_not_preserve_aggregate_command_semantics(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_commands.py": (
                "import pathlib, subprocess, unittest\n"
                "class Commands(unittest.TestCase):\n"
                "    def test_commands(self):\n"
                "        command = ['python3', 'tools/unused.py']\n"
                "        try:\n"
                "            subprocess.run(str(command), check=True)\n"
                "        except FileNotFoundError:\n"
                "            pass\n"
                "        subprocess.run(['python3', str(pathlib.Path('tools/live.py'))])\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/live.py": True, "tools/unused.py": False,
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

    def test_call_argument_helper_effects_precede_process_attribution(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_argument_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "def change():\n"
                "    command.clear()\n"
                "    command.extend(['echo'])\n"
                "    return False\n"
                "subprocess.run(command, check=change())\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_helper_identity_is_captured_before_call_argument_effects(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/README.md": "tools/live.py\n",
            "tests/test_helper_capture.py": (
                "import subprocess\n"
                "def launch(_value):\n"
                "    subprocess.run(['python3', 'tools/live.py'])\n"
                "def replace():\n"
                "    global launch\n"
                "    launch = None\n"
                "    return None\n"
                "launch(replace())\n"
            ),
        }
        self.assertTrue(self.tools(tree)["tools/live.py"]["referenced"])

    def test_call_preserves_earlier_argument_value_before_later_rebinding(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_argument_capture.py": (
                "import runpy\n"
                "script = 'other.py'\n"
                "def change():\n"
                "    global script\n"
                "    script = 'tools/unused.py'\n"
                "    return {}\n"
                "runpy.run_path(script, init_globals=change())\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_sequence_preserves_earlier_element_value_before_later_effects(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_sequence_capture.py": (
                "import subprocess\n"
                "script = 'other.py'\n"
                "def change():\n"
                "    global script\n"
                "    script = 'tools/unused.py'\n"
                "    return 'argument'\n"
                "command = ['python3', script, change()]\n"
                "subprocess.run(command)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_sequence_capture_survives_list_concatenation(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_sequence_concat.py": (
                "import subprocess\n"
                "script = 'other.py'\n"
                "def change():\n"
                "    global script\n"
                "    script = 'tools/unused.py'\n"
                "    return 'argument'\n"
                "command = ['python3', script] + [change()]\n"
                "subprocess.run(command)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_concatenation_preserves_named_operand_before_later_effects(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_named_concat_capture.py": (
                "import subprocess\n"
                "command = ['python3', 'other.py']\n"
                "def change():\n"
                "    global command\n"
                "    command = ['python3', 'tools/unused.py']\n"
                "    return 'argument'\n"
                "result = command + [change()]\n"
                "subprocess.run(result)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_python_helper_stops_after_unconditional_termination(self):
        tree = {
            "tools/direct.py": "x = 1\n",
            "tools/branch.py": "x = 1\n",
            "tools/README.md": "tools/direct.py tools/branch.py\n",
            "tests/test_helper_termination.py": (
                "import runpy\n"
                "def direct():\n"
                "    return\n"
                "    runpy.run_path('tools/direct.py')\n"
                "def selected_branch():\n"
                "    if True:\n"
                "        raise RuntimeError('stop')\n"
                "    runpy.run_path('tools/branch.py')\n"
                "direct()\n"
                "try:\n"
                "    selected_branch()\n"
                "except RuntimeError:\n"
                "    pass\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/branch.py": False, "tools/direct.py": False},
        )

    def test_selected_python_branch_stops_after_termination(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_branch_termination.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "def launch():\n"
                "    if True:\n"
                "        return\n"
                "        runpy.run_path(script)\n"
                "launch()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_reassigned_python_helper_is_not_executed(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_helper_reassignment.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "def launch():\n"
                "    runpy.run_path(script)\n"
                "launch = print\n"
                "launch()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_repeated_helper_invocations_apply_each_mutation(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_repeated_helper.py": (
                "import subprocess\n"
                "command = ['echo']\n"
                "def clear():\n"
                "    global command\n"
                "    command.clear()\n"
                "    command.extend(['echo'])\n"
                "def exercise():\n"
                "    global command\n"
                "    clear()\n"
                "    command = ['python3', 'tools/unused.py']\n"
                "    clear()\n"
                "    subprocess.run(command)\n"
                "exercise()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_helper_invocations_use_fresh_local_binding_environments(self):
        tree = {
            "tools/helper-unused.py": "x = 1\n",
            "tools/import-unused.py": "x = 1\n",
            "tools/value-unused.py": "x = 1\n",
            "tools/README.md": (
                "tools/helper-unused.py tools/import-unused.py tools/value-unused.py\n"
            ),
            "tests/test_fresh_helper_locals.py": (
                "import runpy, subprocess\n"
                "def exercise():\n"
                "    try:\n"
                "        subprocess.run(command)\n"
                "    except UnboundLocalError:\n"
                "        pass\n"
                "    command = ['python3', 'tools/value-unused.py']\n"
                "exercise()\n"
                "exercise()\n"
                "def helper_exercise():\n"
                "    try:\n"
                "        launch()\n"
                "    except UnboundLocalError:\n"
                "        pass\n"
                "    def launch(): runpy.run_path('tools/helper-unused.py')\n"
                "helper_exercise()\n"
                "helper_exercise()\n"
                "def import_exercise():\n"
                "    try:\n"
                "        local_subprocess.run(['python3', 'tools/import-unused.py'])\n"
                "    except UnboundLocalError:\n"
                "        pass\n"
                "    import subprocess as local_subprocess\n"
                "import_exercise()\n"
                "import_exercise()\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/helper-unused.py": False, "tools/import-unused.py": False,
             "tools/value-unused.py": False},
        )

    def test_global_assignment_invalidates_module_helper_identity(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_global_helper_replacement.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "def launch():\n"
                "    runpy.run_path(script)\n"
                "def replace():\n"
                "    global launch\n"
                "    launch = print\n"
                "replace()\n"
                "launch()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_boolean_short_circuit_skips_unreachable_loader(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_short_circuit.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "False and runpy.run_path(script)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_conditional_expression_skips_unreachable_loader(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_conditional_expression.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "runpy.run_path(script) if False else None\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unawaited_async_helper_body_is_not_executed(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_async_helper.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "async def launch():\n"
                "    runpy.run_path(script)\n"
                "launch().close()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_async_forwarding_wrapper_is_not_executed(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_async_wrapper.py": (
                "import runpy\n"
                "async def launch(script):\n"
                "    runpy.run_path(script)\n"
                "launch('tools/unused.py').close()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_uniterated_generator_helper_body_is_not_executed(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_generator_helper.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "def launch():\n"
                "    runpy.run_path(script)\n"
                "    yield None\n"
                "launch().close()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_uniterated_generator_method_body_is_not_executed(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_generator_method.py": (
                "import runpy, unittest\n"
                "script = 'tools/unused.py'\n"
                "class GeneratorMethod(unittest.TestCase):\n"
                "    def launch(self):\n"
                "        runpy.run_path(script)\n"
                "        yield None\n"
                "    def test_generator(self):\n"
                "        self.launch().close()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_comprehension_bodies_need_established_iteration(self):
        tree = {
            "tools/list-unused.py": "x = 1\n",
            "tools/generator-unused.py": "x = 1\n",
            "tools/README.md": "tools/list-unused.py tools/generator-unused.py\n",
            "tests/test_comprehensions.py": (
                "import runpy\n"
                "list_script = 'tools/list-unused.py'\n"
                "generator_script = 'tools/generator-unused.py'\n"
                "[runpy.run_path(list_script) for item in []]\n"
                "(runpy.run_path(generator_script) for item in [1])\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/generator-unused.py": False, "tools/list-unused.py": False},
        )

    def test_comprehension_bodies_require_supported_iterables(self):
        tree = {
            "tools/direct-unused.py": "x = 1\n",
            "tools/wrapper-unused.py": "x = 1\n",
            "tools/README.md": "tools/direct-unused.py tools/wrapper-unused.py\n",
            "tests/test_non_iterable_comprehensions.py": (
                "import runpy\n"
                "direct = 'tools/direct-unused.py'\n"
                "try:\n"
                "    [runpy.run_path(direct) for _ in 1]\n"
                "except TypeError:\n"
                "    pass\n"
                "def launch(script):\n"
                "    [runpy.run_path(script) for _ in 1]\n"
                "try:\n"
                "    launch('tools/wrapper-unused.py')\n"
                "except TypeError:\n"
                "    pass\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/direct-unused.py": False, "tools/wrapper-unused.py": False},
        )

    def test_unresolved_eager_comprehension_effects_invalidate_provenance(self):
        tree = {
            "tools/direct-unused.py": "x = 1\n",
            "tools/wrapper-unused.py": "x = 1\n",
            "tools/README.md": "tools/direct-unused.py tools/wrapper-unused.py\n",
            "tests/test_comprehension_effects.py": (
                "import subprocess\n"
                "values = [1]\n"
                "direct = ['python3', 'tools/direct-unused.py']\n"
                "[direct.__setitem__(slice(None), ['echo']) for _ in values]\n"
                "subprocess.run(direct, check=True)\n"
                "def launch(command, items):\n"
                "    [command.__setitem__(slice(None), ['echo']) for _ in items]\n"
                "    return subprocess.run(command, check=True)\n"
                "launch(['python3', 'tools/wrapper-unused.py'], values)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/direct-unused.py": False, "tools/wrapper-unused.py": False},
        )

    def test_comprehension_target_shadows_outer_static_binding(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_comprehension_target.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "[runpy.run_path(script) for script in ['other.py']]\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_forwarding_wrapper_respects_boolean_short_circuiting(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_wrapper_short_circuit.py": (
                "import runpy\n"
                "def launch(script):\n"
                "    False and runpy.run_path(script)\n"
                "launch('tools/unused.py')\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unknown_wrapper_branch_effects_invalidate_forwarding(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_wrapper_branch.py": (
                "import subprocess\n"
                "def launch(command):\n"
                "    flag = True\n"
                "    if flag:\n"
                "        command[:] = ['echo']\n"
                "    subprocess.run(command, check=True)\n"
                "launch(['python3', 'tools/unused.py'])\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_forwarding_wrapper_respects_conditional_expression_reachability(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_wrapper_conditional_expression.py": (
                "import runpy\n"
                "def launch(script):\n"
                "    runpy.run_path(script) if False else None\n"
                "launch('tools/unused.py')\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unknown_conditional_expression_effects_invalidate_provenance(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/wrapper-unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py tools/wrapper-unused.py\n",
            "tests/test_conditional_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "flag = True\n"
                "command.__setitem__(slice(None), ['echo']) if flag else None\n"
                "subprocess.run(command, check=True)\n"
                "def launch(arguments):\n"
                "    other = True\n"
                "    arguments.clear() if other else None\n"
                "    subprocess.run(arguments, check=True)\n"
                "launch(['python3', 'tools/wrapper-unused.py'])\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/unused.py": False, "tools/wrapper-unused.py": False})

    def test_forwarding_wrapper_requires_valid_call_binding(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_wrapper_binding.py": (
                "import subprocess\n"
                "def launch(command, required):\n"
                "    subprocess.run(command, check=True)\n"
                "try:\n"
                "    launch(['python3', 'tools/unused.py'])\n"
                "except TypeError:\n"
                "    pass\n"
                "launch(['python3', 'tools/live.py'], required=True)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.py": True, "tools/unused.py": False})

    def test_named_expression_invalidates_local_import_identity(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_walrus_import.py": (
                "class Fake:\n"
                "    @staticmethod\n"
                "    def run(*args, **kwargs):\n"
                "        return None\n"
                "def exercise():\n"
                "    import subprocess\n"
                "    (subprocess := Fake)\n"
                "    subprocess.run(['python3', 'tools/unused.py'])\n"
                "def restored():\n"
                "    (subprocess := Fake)\n"
                "    import subprocess\n"
                "    subprocess.run(['python3', 'tools/live.py'])\n"
                "exercise()\n"
                "restored()\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.py": True, "tools/unused.py": False})

    def test_unknown_branch_rebinding_invalidates_local_import_identity(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_branch_import.py": (
                "class Fake:\n"
                "    @staticmethod\n"
                "    def run(*args, **kwargs):\n"
                "        return None\n"
                "def exercise(flag):\n"
                "    import subprocess\n"
                "    command = ['python3', 'tools/unused.py']\n"
                "    if flag:\n"
                "        subprocess = Fake\n"
                "    subprocess.run(command)\n"
                "exercise(True)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_annotation_only_statement_preserves_local_import_identity(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_annotation_import.py": (
                "class Fake:\n"
                "    @staticmethod\n"
                "    def run(*args, **kwargs):\n"
                "        return None\n"
                "def live():\n"
                "    import subprocess\n"
                "    subprocess: object\n"
                "    subprocess.run(['python3', 'tools/live.py'], check=True)\n"
                "def stale():\n"
                "    import subprocess\n"
                "    subprocess: object = Fake\n"
                "    subprocess.run(['python3', 'tools/unused.py'], check=True)\n"
                "live()\n"
                "stale()\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.py": True, "tools/unused.py": False})

    def test_helper_argument_alias_mutation_updates_caller_aggregate(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/rebound-live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/keyword-unused.py": "x = 1\n",
            "tools/default-unused.py": "x = 1\n",
            "tools/vararg-unused.py": "x = 1\n",
            "tools/kwarg-unused.py": "x = 1\n",
            "tools/README.md": (
                "tools/live.py tools/rebound-live.py tools/unused.py "
                "tools/keyword-unused.py tools/default-unused.py "
                "tools/vararg-unused.py tools/kwarg-unused.py\n"),
            "tests/test_helper_alias.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "keyword_command = ['python3', 'tools/keyword-unused.py']\n"
                "default_command = ['python3', 'tools/default-unused.py']\n"
                "vararg_command = ['python3', 'tools/vararg-unused.py']\n"
                "kwarg_command = ['python3', 'tools/kwarg-unused.py']\n"
                "rebound = ['python3', 'tools/rebound-live.py']\n"
                "def replace(argv):\n"
                "    argv[:] = ['echo']\n"
                "def only_rebind(argv):\n"
                "    argv = ['echo']\n"
                "def replace_default(argv=default_command):\n"
                "    argv[:] = ['echo']\n"
                "def replace_vararg(*argv):\n"
                "    argv[0][:] = ['echo']\n"
                "def replace_kwarg(**argv):\n"
                "    argv['command'][:] = ['echo']\n"
                "replace(command)\n"
                "replace(argv=keyword_command)\n"
                "replace_default()\n"
                "replace_vararg(vararg_command)\n"
                "replace_kwarg(command=kwarg_command)\n"
                "only_rebind(rebound)\n"
                "subprocess.run(command, check=True)\n"
                "subprocess.run(keyword_command, check=True)\n"
                "subprocess.run(default_command, check=True)\n"
                "subprocess.run(vararg_command, check=True)\n"
                "subprocess.run(kwarg_command, check=True)\n"
                "subprocess.run(rebound, check=True)\n"
                "subprocess.run(['python3', 'tools/live.py'], check=True)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/default-unused.py": False,
             "tools/keyword-unused.py": False,
             "tools/kwarg-unused.py": False,
             "tools/live.py": True, "tools/rebound-live.py": True,
             "tools/unused.py": False, "tools/vararg-unused.py": False})

    def test_unittest_setup_effects_precede_each_test_method(self):
        names = ("pre", "live", "post", "setup_direct", "teardown_direct",
                 "class_pre", "class_live", "class_post",
                 "setupclass_direct", "teardownclass_direct")
        tree = {
            **{f"tools/{name}.py": "x = 1\n" for name in names},
            "tools/README.md": " ".join(
                f"tools/{name}.py" for name in names) + "\n",
            "tests/test_lifecycle.py": (
                "import subprocess\n"
                "import unittest\n"
                "command = ['python3', 'tools/pre.py']\n"
                "class_command = ['python3', 'tools/class_pre.py']\n"
                "class CommandTest(unittest.TestCase):\n"
                "    @classmethod\n"
                "    def tearDownClass(cls):\n"
                "        global class_command\n"
                "        subprocess.run(['python3', 'tools/teardownclass_direct.py'])\n"
                "        class_command = ['python3', 'tools/class_post.py']\n"
                "    def tearDown(self):\n"
                "        global command\n"
                "        subprocess.run(['python3', 'tools/teardown_direct.py'])\n"
                "        command = ['python3', 'tools/post.py']\n"
                "    def test_b(self):\n"
                "        subprocess.run(command)\n"
                "        subprocess.run(class_command)\n"
                "    @classmethod\n"
                "    def setUpClass(cls):\n"
                "        global class_command\n"
                "        subprocess.run(['python3', 'tools/setupclass_direct.py'])\n"
                "        class_command = ['python3', 'tools/class_live.py']\n"
                "    def setUp(self):\n"
                "        global command\n"
                "        subprocess.run(['python3', 'tools/setup_direct.py'])\n"
                "        command = ['python3', 'tools/live.py']\n"
                "    def test_a(self):\n"
                "        subprocess.run(command)\n"
                "        subprocess.run(class_command)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/class_live.py": True, "tools/class_post.py": False,
             "tools/class_pre.py": False, "tools/live.py": True,
             "tools/post.py": False, "tools/pre.py": False,
             "tools/setup_direct.py": True,
             "tools/setupclass_direct.py": True,
             "tools/teardown_direct.py": True,
             "tools/teardownclass_direct.py": True})

    def test_global_helper_assignment_invalidates_forwarding_wrapper(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_global_wrapper_replacement.py": (
                "import runpy\n"
                "def change():\n"
                "    global launch\n"
                "    launch = print\n"
                "def launch(script):\n"
                "    runpy.run_path(script)\n"
                "change()\n"
                "launch('tools/unused.py')\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_nonlocal_helper_assignment_invalidates_enclosing_forwarding_wrapper(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_nonlocal_wrapper_replacement.py": (
                "import runpy\n"
                "def outer():\n"
                "    def change():\n"
                "        nonlocal launch\n"
                "        launch = print\n"
                "    def launch(script):\n"
                "        runpy.run_path(script)\n"
                "    change()\n"
                "    launch('tools/unused.py')\n"
                "outer()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_nonlocal_assignment_invalidates_enclosing_helper_identity(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_nonlocal_helper_replacement.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "def outer():\n"
                "    def change():\n"
                "        nonlocal launch\n"
                "        launch = print\n"
                "    def launch():\n"
                "        runpy.run_path(script)\n"
                "    change()\n"
                "    launch()\n"
                "outer()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_forwarding_wrapper_applies_argument_effects_before_forwarding(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_wrapper_argument_effects.py": (
                "import subprocess\n"
                "def launch(command):\n"
                "    subprocess.run(command, check=command.clear())\n"
                "launch(['python3', 'tools/unused.py'])\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_forwarding_wrapper_respects_chained_comparison_short_circuiting(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_wrapper_comparison.py": (
                "import runpy\n"
                "def launch(script):\n"
                "    0 > 1 > runpy.run_path(script)\n"
                "launch('tools/unused.py')\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_try_handlers_require_proven_reachability_but_finally_executes(self):
        tree = {
            "tools/body-live.py": "x = 1\n",
            "tools/handler-unused.py": "x = 1\n",
            "tools/finally-live.py": "x = 1\n",
            "tools/README.md": (
                "tools/body-live.py\n"
                "tools/handler-unused.py\n"
                "tools/finally-live.py\n"
            ),
            "tests/test_try_reachability.py": (
                "import runpy\n"
                "try:\n"
                "    runpy.run_path('tools/body-live.py')\n"
                "except ValueError:\n"
                "    runpy.run_path('tools/handler-unused.py')\n"
                "finally:\n"
                "    runpy.run_path('tools/finally-live.py')\n"
            ),
        }
        self.assertEqual(
            {
                "tools/body-live.py": True,
                "tools/handler-unused.py": False,
                "tools/finally-live.py": True,
            },
            {path: item["referenced"] for path, item in self.tools(tree).items()},
        )

    def test_unproven_exception_branches_invalidate_downstream_bindings(self):
        tree = {
            "tools/handler-unused.py": "x = 1\n",
            "tools/else-unused.py": "x = 1\n",
            "tools/README.md": "tools/handler-unused.py tools/else-unused.py\n",
            "tests/test_try_effects.py": (
                "import subprocess\n"
                "handler_command = ['python3', 'tools/handler-unused.py']\n"
                "else_command = ['python3', 'tools/else-unused.py']\n"
                "try:\n"
                "    raise ValueError()\n"
                "except ValueError:\n"
                "    handler_command.clear()\n"
                "subprocess.run(handler_command)\n"
                "try:\n"
                "    value = 1\n"
                "except ValueError:\n"
                "    pass\n"
                "else:\n"
                "    else_command.clear()\n"
                "subprocess.run(else_command)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/else-unused.py": False, "tools/handler-unused.py": False},
        )

    def test_exception_boundaries_invalidate_callable_identities(self):
        tree = {
            "tools/helper-unused.py": "x = 1\n",
            "tools/wrapper-unused.py": "x = 1\n",
            "tools/README.md": "tools/helper-unused.py tools/wrapper-unused.py\n",
            "tests/test_exception_callable_effects.py": (
                "import runpy\n"
                "def helper():\n"
                "    runpy.run_path('tools/helper-unused.py')\n"
                "def wrapper(script):\n"
                "    runpy.run_path(script)\n"
                "try:\n"
                "    raise ValueError()\n"
                "except ValueError:\n"
                "    helper = print\n"
                "    wrapper = print\n"
                "helper()\n"
                "wrapper('tools/wrapper-unused.py')\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/helper-unused.py": False, "tools/wrapper-unused.py": False},
        )

    def test_exception_target_names_invalidate_static_bindings(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_exception_target.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "try:\n"
                "    raise ValueError()\n"
                "except ValueError as command:\n"
                "    pass\n"
                "try:\n"
                "    subprocess.run(command)\n"
                "except NameError:\n"
                "    pass\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_possible_loop_writes_invalidate_static_bindings(self):
        tree = {
            "tools/body-unused.py": "x = 1\n",
            "tools/else-unused.py": "x = 1\n",
            "tools/target-unused.py": "x = 1\n",
            "tools/wrapper-unused.py": "x = 1\n",
            "tools/README.md": (
                "tools/body-unused.py tools/else-unused.py "
                "tools/target-unused.py tools/wrapper-unused.py\n"
            ),
            "tests/test_loop_effect.py": (
                "import subprocess\n"
                "body = ['python3', 'tools/body-unused.py']\n"
                "body_alias = body\n"
                "for _ in [1]:\n"
                "    body_alias[:] = ['echo']\n"
                "target = ['python3', 'tools/target-unused.py']\n"
                "target_alias = target\n"
                "for target_alias[:] in [['echo']]:\n"
                "    pass\n"
                "other = ['python3', 'tools/else-unused.py']\n"
                "other_alias = other\n"
                "for _ in []:\n"
                "    pass\n"
                "else:\n"
                "    other_alias[:] = ['echo']\n"
                "def launch(command):\n"
                "    alias = command\n"
                "    for _ in [1]:\n"
                "        alias[:] = ['echo']\n"
                "    subprocess.run(command)\n"
                "subprocess.run(body)\n"
                "subprocess.run(target)\n"
                "subprocess.run(other)\n"
                "launch(['python3', 'tools/wrapper-unused.py'])\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/body-unused.py": False, "tools/else-unused.py": False,
             "tools/target-unused.py": False, "tools/wrapper-unused.py": False},
        )

    def test_forwarding_wrapper_try_boundary_and_finally_execution(self):
        tree = {
            "tools/handler-unused.py": "x = 1\n",
            "tools/finally-live.py": "x = 1\n",
            "tools/README.md": "tools/handler-unused.py tools/finally-live.py\n",
            "tests/test_wrapper_try.py": (
                "import runpy\n"
                "def launch(handler_script, finally_script):\n"
                "    try:\n"
                "        value = 1\n"
                "    except ValueError:\n"
                "        runpy.run_path(handler_script)\n"
                "    finally:\n"
                "        runpy.run_path(finally_script)\n"
                "launch('tools/handler-unused.py', 'tools/finally-live.py')\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/finally-live.py": True, "tools/handler-unused.py": False},
        )

    def test_forwarding_wrapper_requires_established_comprehension_iteration(self):
        tree = {
            "tools/list-unused.py": "x = 1\n",
            "tools/generator-unused.py": "x = 1\n",
            "tools/eager-live.py": "x = 1\n",
            "tools/README.md": (
                "tools/list-unused.py tools/generator-unused.py tools/eager-live.py\n"
            ),
            "tests/test_wrapper_comprehensions.py": (
                "import runpy\n"
                "def list_launch(script):\n"
                "    [runpy.run_path(script) for _ in []]\n"
                "def generator_launch(script):\n"
                "    (runpy.run_path(script) for _ in [1])\n"
                "def eager_launch(script):\n"
                "    [runpy.run_path(script) for _ in [1]]\n"
                "list_launch('tools/list-unused.py')\n"
                "generator_launch('tools/generator-unused.py')\n"
                "eager_launch('tools/eager-live.py')\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {
                "tools/eager-live.py": True,
                "tools/generator-unused.py": False,
                "tools/list-unused.py": False,
            },
        )

    def test_finally_termination_stops_following_helper_statements(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_finally_return.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "def exercise():\n"
                "    try:\n"
                "        value = 1\n"
                "    finally:\n"
                "        return\n"
                "    runpy.run_path(script)\n"
                "exercise()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_nonlocal_assignment_updates_enclosing_static_command_binding(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_nonlocal_command.py": (
                "import subprocess\n"
                "def exercise():\n"
                "    command = ['python3', 'tools/unused.py']\n"
                "    def replace():\n"
                "        nonlocal command\n"
                "        command = ['python3', 'tools/live.py']\n"
                "    replace()\n"
                "    subprocess.run(command)\n"
                "exercise()\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.py": True, "tools/unused.py": False},
        )

    def test_unknown_branch_effects_invalidate_downstream_bindings(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_unknown_branch_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "alias = command\n"
                "flag = True\n"
                "if flag:\n"
                "    alias.clear()\n"
                "subprocess.run(command)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unknown_branch_helper_effects_invalidate_downstream_bindings(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_unknown_branch_helper_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "def replace():\n"
                "    command[:] = ['echo']\n"
                "flag = True\n"
                "if flag:\n"
                "    replace()\n"
                "subprocess.run(command, check=True)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unknown_branch_definition_time_effects_invalidate_downstream_bindings(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_unknown_branch_definition_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "flag = True\n"
                "if flag:\n"
                "    def unused(x=command.__setitem__(slice(None), ['echo'])):\n"
                "        return x\n"
                "subprocess.run(command, check=True)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unknown_branch_lambda_defaults_invalidate_downstream_bindings(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_unknown_branch_lambda_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "flag = True\n"
                "if flag:\n"
                "    unused = lambda x=command.__setitem__(slice(None), ['echo']): x\n"
                "subprocess.run(command, check=True)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unknown_branch_class_construction_invalidates_downstream_bindings(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_unknown_branch_class_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "flag = True\n"
                "if flag:\n"
                "    class Settings:\n"
                "        command.__setitem__(slice(None), ['echo'])\n"
                "subprocess.run(command, check=True)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unknown_branch_helper_definition_time_parameter_mutation_invalidates_caller(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_unknown_branch_helper_definition_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "def replace(argv):\n"
                "    unused = lambda x=argv.__setitem__(slice(None), ['echo']): x\n"
                "flag = True\n"
                "if flag:\n"
                "    replace(command)\n"
                "subprocess.run(command, check=True)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unknown_branch_helper_class_construction_parameter_mutation_invalidates_caller(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_unknown_branch_helper_class_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "def replace(argv):\n"
                "    class Change:\n"
                "        x = argv.__setitem__(slice(None), ['echo'])\n"
                "flag = True\n"
                "if flag:\n"
                "    replace(command)\n"
                "subprocess.run(command, check=True)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unknown_branch_helper_parameter_effects_invalidate_caller_binding(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_unknown_branch_parameter_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "def replace(target):\n"
                "    target[:] = ['echo']\n"
                "flag = True\n"
                "if flag:\n"
                "    replace(command)\n"
                "subprocess.run(command, check=True)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unknown_branch_helper_parameter_alias_effects_invalidate_caller_binding(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_unknown_branch_parameter_alias_effect.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "def replace(argv):\n"
                "    alias = argv\n"
                "    alias[:] = ['echo']\n"
                "flag = True\n"
                "if flag:\n"
                "    replace(command)\n"
                "subprocess.run(command, check=True)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unknown_branch_helper_parameter_rebinding_stays_local(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/README.md": "tools/live.py\n",
            "tests/test_unknown_branch_parameter_rebind.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/live.py']\n"
                "def replace(target):\n"
                "    target = ['echo']\n"
                "flag = True\n"
                "if flag:\n"
                "    replace(command)\n"
                "subprocess.run(command, check=True)\n"
            ),
        }
        self.assertTrue(self.tools(tree)["tools/live.py"]["referenced"])

    def test_chained_assignment_interleaves_target_stores_and_effects(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_chained_assignment_order.py": (
                "import subprocess\n"
                "command = ['echo']\n"
                "holder = [None]\n"
                "def replace():\n"
                "    command[:] = ['echo']\n"
                "    return 0\n"
                "command = holder[replace()] = ['python3', 'tools/unused.py']\n"
                "subprocess.run(command, check=True)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_destructuring_assignment_interleaves_element_stores_and_effects(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_destructuring_assignment_order.py": (
                "import subprocess\n"
                "command = ['echo']\n"
                "holder = [None]\n"
                "def replace():\n"
                "    command[:] = ['echo']\n"
                "    return 0\n"
                "command, holder[replace()] = (\n"
                "    ['python3', 'tools/unused.py'], 0)\n"
                "subprocess.run(command, check=True)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_destructuring_snapshots_leaf_values_before_target_effects(self):
        def tree(mutator):
            return {
                "tools/live.py": "x = 1\n",
                "tools/README.md": "tools/live.py\n",
                "tests/test_destructuring_snapshot.py": (
                    "import subprocess\n"
                    "command = ['python3', 'tools/live.py']\n"
                    "source = [0, 0, command]\n"
                    "holder = [None]\n"
                    "def mutate():\n"
                    f"    {mutator}\n"
                    "    return 0\n"
                    "first, holder[mutate()], selected = source\n"
                    "subprocess.run(selected, check=True)\n"
                ),
            }

        self.assertTrue(self.tools(
            tree("source[:] = [0, 0, ['echo']]")
        )["tools/live.py"]["referenced"])
        self.assertFalse(self.tools(
            tree("command[:] = ['echo']")
        )["tools/live.py"]["referenced"])

    def test_handler_free_try_body_termination_stops_following_helper_statements(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_try_return.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "def exercise():\n"
                "    try:\n"
                "        return\n"
                "    finally:\n"
                "        value = 1\n"
                "    runpy.run_path(script)\n"
                "exercise()\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_decorated_python_helper_does_not_retain_original_body(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/old.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/old.py tools/unused.py\n",
            "tests/test_decorated_helper.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "def launch():\n"
                "    runpy.run_path('tools/old.py')\n"
                "def discard(function):\n"
                "    return print\n"
                "@discard\n"
                "def launch():\n"
                "    runpy.run_path(script)\n"
                "def live():\n"
                "    runpy.run_path('tools/live.py')\n"
                "launch()\n"
                "live()\n"
            ),
        }
        report = self.tools(tree)
        self.assertTrue(report["tools/live.py"]["referenced"])
        self.assertFalse(report["tools/old.py"]["referenced"])
        self.assertFalse(report["tools/unused.py"]["referenced"])

    def test_decorated_python_wrapper_is_not_registered(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_decorated_wrapper.py": (
                "import runpy\n"
                "def discard(function):\n"
                "    return print\n"
                "@discard\n"
                "def launch(script):\n"
                "    runpy.run_path(script)\n"
                "launch('tools/unused.py')\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_assignment_expressions_update_static_bindings(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_named_expression.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "(command := ['echo'])\n"
                "subprocess.run(command)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_multiline_calls_preserve_argument_effect_order(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_multiline_call.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "def replace():\n"
                "    command[:] = ['echo']\n"
                "subprocess.run(\n"
                "    command,\n"
                "    check=replace(),\n"
                ")\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_class_body_bindings_do_not_replace_module_bindings(self):
        tree = {
            "tools/class-live.py": "x = 1\n",
            "tools/fallback-unused.py": "x = 1\n",
            "tools/method-live.py": "x = 1\n",
            "tools/module-live.py": "x = 1\n",
            "tools/nested-live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/wrapper-unused.py": "x = 1\n",
            "tools/README.md": (
                "tools/class-live.py tools/fallback-unused.py tools/method-live.py "
                "tools/module-live.py tools/nested-live.py tools/unused.py "
                "tools/wrapper-unused.py\n"
            ),
            "tests/test_class_scope.py": (
                "import subprocess, unittest\n"
                "def dynamic(): return object()\n"
                "def launch(command): subprocess.run(command)\n"
                "command = ['echo']\n"
                "fallback = ['python3', 'tools/fallback-unused.py']\n"
                "module_command = ['python3', 'tools/module-live.py']\n"
                "class Settings:\n"
                "    command = ['python3', 'tools/unused.py']\n"
                "    class_command = ['python3', 'tools/class-live.py']\n"
                "    subprocess.run(module_command)\n"
                "    subprocess.run(class_command)\n"
                "    fallback = dynamic()\n"
                "    subprocess.run(fallback)\n"
                "    launch = print\n"
                "    launch(['python3', 'tools/wrapper-unused.py'])\n"
                "def build():\n"
                "    nested_command = ['python3', 'tools/nested-live.py']\n"
                "    class Nested:\n"
                "        subprocess.run(nested_command)\n"
                "build()\n"
                "class Checks(unittest.TestCase):\n"
                "    def test_method(self):\n"
                "        subprocess.run(['python3', 'tools/method-live.py'])\n"
                "subprocess.run(command)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/class-live.py": True, "tools/method-live.py": True,
             "tools/fallback-unused.py": False, "tools/module-live.py": True,
             "tools/nested-live.py": True, "tools/unused.py": False,
             "tools/wrapper-unused.py": False},
        )

    def test_nested_aggregate_writes_invalidate_reachable_aliases(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/live.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_nested_aggregate.py": (
                "import subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "holder = [command]\n"
                "holder[0][:] = ['echo']\n"
                "subprocess.run(command)\n"
                "live = ['python3', 'tools/live.py']\n"
                "other = [live]\n"
                "other[0] = ['echo']\n"
                "subprocess.run(live)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.py": True, "tools/unused.py": False},
        )

    def test_non_string_executable_value_is_left_unresolved(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_bad_command_shape.py": (
                "import subprocess\n"
                "import unittest\n"
                "class CommandShape(unittest.TestCase):\n"
                "    def test_nested_executable(self):\n"
                "        with self.assertRaises(TypeError):\n"
                "            subprocess.run([['python3', 'tools/unused.py']])\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_lambda_creation_executes_defaults_not_body(self):
        tree = {
            "tools/body-unused.py": "x = 1\n",
            "tools/default-live.py": "x = 1\n",
            "tools/README.md": "tools/body-unused.py tools/default-live.py\n",
            "tests/test_uninvoked_lambda.py": (
                "import runpy\n"
                "body_script = 'tools/body-unused.py'\n"
                "default_script = 'tools/default-live.py'\n"
                "unused = lambda value=runpy.run_path(default_script): runpy.run_path(body_script)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/body-unused.py": False, "tools/default-live.py": True},
        )

    def test_chained_comparison_stops_after_literal_false_prefix(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_comparison.py": (
                "import runpy\n"
                "script = 'tools/unused.py'\n"
                "0 > 1 > runpy.run_path(script)\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])

    def test_unknown_chained_comparison_suffix_effect_invalidates_provenance(self):
        tree = {
            "tools/direct-unused.py": "x = 1\n",
            "tools/wrapper-unused.py": "x = 1\n",
            "tools/README.md": "tools/direct-unused.py tools/wrapper-unused.py\n",
            "tests/test_comparison.py": (
                "import subprocess\n"
                "flag = object()\n"
                "direct = ['python3', 'tools/direct-unused.py']\n"
                "flag == 1 == direct.__setitem__(slice(None), ['echo'])\n"
                "subprocess.run(direct, check=True)\n"
                "def launch(command, condition):\n"
                "    condition == 1 == command.__setitem__(slice(None), ['echo'])\n"
                "    return subprocess.run(command, check=True)\n"
                "launch(['python3', 'tools/wrapper-unused.py'], flag)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/direct-unused.py": False, "tools/wrapper-unused.py": False},
        )

    def test_assignment_target_effects_follow_rhs_and_precede_later_statements(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_assignment_target.py": (
                "import runpy, subprocess\n"
                "command = ['python3', 'tools/unused.py']\n"
                "holder = {}\n"
                "def replace():\n"
                "    runpy.run_path('tools/live.py')\n"
                "    command[:] = ['echo']\n"
                "    return 0\n"
                "holder[replace()] = 1\n"
                "subprocess.run(command, check=True)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.py": True, "tools/unused.py": False},
        )

    def test_assignment_rhs_helper_executes_before_target_rebinding(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/README.md": "tools/live.py\n",
            "tests/test_rhs_helper.py": (
                "import runpy\n"
                "def launch():\n"
                "    runpy.run_path('tools/live.py')\n"
                "launch = launch()\n"
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

    def test_tuple_augmented_assignment_does_not_mutate_alias(self):
        tree = {
            "tools/live.py": "x = 1\n",
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/live.py tools/unused.py\n",
            "tests/test_tuple_alias.py": (
                "import subprocess\n"
                "command = ('python3',)\n"
                "alias = command\n"
                "command += ('tools/unused.py',)\n"
                "subprocess.run(alias, input=b'', check=True)\n"
                "live = ('python3',)\n"
                "live += ('tools/live.py',)\n"
                "subprocess.run(live, input=b'', check=True)\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.py": True, "tools/unused.py": False})

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

    def test_augmented_assignment_captures_target_before_rhs_effects(self):
        tree = {
            "tools/unused.py": "x = 1\n",
            "tools/README.md": "tools/unused.py\n",
            "tests/test_augmented_rhs.py": (
                "import subprocess\n"
                "command = ['echo']\n"
                "def replace():\n"
                "    global command\n"
                "    command = ['python3', 'tools/unused.py']\n"
                "    return 'extra'\n"
                "command += ['python3', 'tools/unused.py', replace()]\n"
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

    def test_js_module_block_declaration_shadows_named_child_process_import(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/block-shadow.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "{\n"
                "  const spawnSync=(...args) => console.log(args);\n"
                "  spawnSync('echo',[script]);\n"
                "}\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_destructured_block_declaration_shadows_child_process_import(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/destructured-block-shadow.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "{\n"
                "  const [spawnSync]=[(...args) => console.log(args)];\n"
                "  spawnSync('echo',[script]);\n"
                "}\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_asi_block_declaration_shadows_child_process_import(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/asi-block-shadow.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "{\n"
                "  let spawnSync\n"
                "  spawnSync=(...args) => console.log(args);\n"
                "  spawnSync('echo',[script]);\n"
                "}\n"
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

    def test_js_default_initializer_executes_only_for_invoked_helper(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/live.mjs tools/unused.mjs\n",
            "tests/default-execution.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const liveScript=new URL('../tools/live.mjs',import.meta.url);\n"
                "const unusedScript=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function unused(value=spawnSync('echo',[unusedScript])) {}\n"
                "function live(value=spawnSync('echo',[liveScript])) {}\n"
                "live();\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True, "tools/unused.mjs": False},
        )

    def test_js_default_initializer_requires_missing_or_undefined_argument(self):
        tree = {
            "tools/missing-live.mjs": "export const live = true;\n",
            "tools/undefined-live.mjs": "export const live = true;\n",
            "tools/supplied-unused.mjs": "export const unused = true;\n",
            "tools/README.md": (
                "tools/missing-live.mjs tools/undefined-live.mjs "
                "tools/supplied-unused.mjs\n"),
            "tests/default-delivery.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const missingLive=new URL('../tools/missing-live.mjs',import.meta.url);\n"
                "const undefinedLive=new URL('../tools/undefined-live.mjs',import.meta.url);\n"
                "const suppliedUnused=new URL('../tools/supplied-unused.mjs',import.meta.url);\n"
                "function missing(value=spawnSync('echo',[missingLive])) {}\n"
                "function explicit(value=spawnSync('echo',[undefinedLive])) {}\n"
                "function supplied(value=spawnSync('echo',[suppliedUnused])) {}\n"
                "missing();explicit(undefined);supplied(1);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/missing-live.mjs": True,
             "tools/supplied-unused.mjs": False,
             "tools/undefined-live.mjs": True},
        )

    def test_js_nested_parameter_scopes_are_deferred(self):
        tree = {
            "tools/arrow-unused.mjs": "export const unused = true;\n",
            "tools/method-unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/arrow-unused.mjs tools/method-unused.mjs\n",
            "tests/nested-parameter-scope.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const arrowScript=new URL('../tools/arrow-unused.mjs',import.meta.url);\n"
                "const methodScript=new URL('../tools/method-unused.mjs',import.meta.url);\n"
                "const launch=(value=spawnSync('echo',[arrowScript]))=>{};\n"
                "const holder={run(value=spawnSync('echo',[methodScript])){}};\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/arrow-unused.mjs": False,
             "tools/method-unused.mjs": False},
        )

    def test_js_unproven_catch_handler_is_not_activated(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/catch-reachability.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "try{void 0;}catch(error){spawnSync('echo',[script]);}\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

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
                "unused();\n"
                "spawnSync('echo',[script]);\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_url_binding_does_not_escape_lexical_block(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/url-block.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script='other.py';\n"
                "{ const script=new URL('../tools/unused.mjs',import.meta.url); }\n"
                "spawnSync('echo',[script]);\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_destructured_lexical_binding_shadows_outer_url_argument(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/url-argument-shadow.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "{\n"
                "  const [script]=['other.py'];\n"
                "  spawnSync('echo',[script]);\n"
                "}\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_destructured_helper_binding_shadows_module_helper(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/helper-shadow.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function launch(){spawnSync('echo',[script]);}\n"
                "function exercise(){\n"
                "  const [launch]=[()=>{}];\n"
                "  launch();\n"
                "}\n"
                "exercise();\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_nested_named_helper_chain_reaches_child_process_call(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/README.md": "tools/live.mjs\n",
            "tests/nested-helper.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/live.mjs',import.meta.url);\n"
                "function outer(){\n"
                "  function inner(){spawnSync('echo',[script]);}\n"
                "  inner();\n"
                "}\n"
                "outer();\n"
            ),
        }
        self.assertTrue(self.tools(tree)["tools/live.mjs"]["referenced"])

    def test_js_url_binding_must_reach_child_process_arguments(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/live.mjs tools/unused.mjs\n",
            "tests/url-delivery.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "const unused=new URL('../tools/unused.mjs',import.meta.url);\n"
                "spawnSync('echo',[live,false && unused]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True, "tools/unused.mjs": False},
        )

    def test_js_url_binding_requires_command_argument_position(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/live.mjs tools/unused.mjs\n",
            "tests/url-position.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "const unused=new URL('../tools/unused.mjs',import.meta.url);\n"
                "spawnSync('echo',[live],unused);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True, "tools/unused.mjs": False},
        )

    def test_js_file_url_to_path_requires_unshadowed_node_url_import(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/live.mjs tools/unused.mjs\n",
            "tests/trusted-converter.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "import {fileURLToPath} from 'node:url';\n"
                "const live=fileURLToPath(new URL('../tools/live.mjs',import.meta.url));\n"
                "spawnSync('echo',[live]);\n"
            ),
            "tests/fake-converter.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "function fileURLToPath(value){return 'other.py';}\n"
                "const unused=fileURLToPath(new URL('../tools/unused.mjs',import.meta.url));\n"
                "spawnSync('echo',[unused]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True, "tools/unused.mjs": False},
        )

    def test_js_url_constructor_requires_unshadowed_builtin_identity(self):
        tree = {
            "tools/block-unused.mjs": "export const unused = true;\n",
            "tools/import-unused.mjs": "export const unused = true;\n",
            "tools/live.mjs": "export const live = true;\n",
            "tools/module-unused.mjs": "export const unused = true;\n",
            "tools/parameter-unused.mjs": "export const unused = true;\n",
            "tools/sibling-live.mjs": "export const live = true;\n",
            "tools/README.md": (
                "tools/block-unused.mjs tools/import-unused.mjs tools/live.mjs "
                "tools/module-unused.mjs tools/parameter-unused.mjs "
                "tools/sibling-live.mjs\n"
            ),
            "tests/url-constructor-live.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "spawnSync('echo',[live]);\n"
                "function unrelated(){ class URL {} }\n"
                "const sibling=new URL('../tools/sibling-live.mjs',import.meta.url);\n"
                "spawnSync('echo',[sibling]);\n"
                "{\n"
                "  class URL { constructor(){ return new globalThis.URL('file:///other.py'); } }\n"
                "  const unused=new URL('../tools/block-unused.mjs',import.meta.url);\n"
                "  spawnSync('echo',[unused]);\n"
                "}\n"
            ),
            "tests/url-constructor-module.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "class URL { constructor(){ return new globalThis.URL('file:///other.py'); } }\n"
                "const unused=new URL('../tools/module-unused.mjs',import.meta.url);\n"
                "spawnSync('echo',[unused]);\n"
            ),
            "tests/url-constructor-parameter.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "function exercise(URL){\n"
                "  const unused=new URL('../tools/parameter-unused.mjs',import.meta.url);\n"
                "  spawnSync('echo',[unused]);\n"
                "}\n"
                "exercise(class URL {});\n"
            ),
            "tests/url-constructor-import.test.mjs": (
                "import URL from 'decoy';\n"
                "import {spawnSync} from 'node:child_process';\n"
                "const unused=new URL('../tools/import-unused.mjs',import.meta.url);\n"
                "spawnSync('echo',[unused]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/block-unused.mjs": False, "tools/import-unused.mjs": False,
             "tools/live.mjs": True, "tools/module-unused.mjs": False,
             "tools/parameter-unused.mjs": False,
             "tools/sibling-live.mjs": True},
        )

    def test_js_url_property_mutation_invalidates_path_provenance(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/live.mjs tools/unused.mjs\n",
            "tests/url-property-mutation.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "spawnSync('echo',[live]);\n"
                "script.pathname='/other.py';\n"
                "spawnSync('echo',[script]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True, "tools/unused.mjs": False},
        )

    def test_js_url_binding_requires_complete_supported_initializer(self):
        tree = {
            "tools/live.mjs": "export const live = true;\n",
            "tools/continued-unused.mjs": "export const unused = true;\n",
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": (
                "tools/live.mjs tools/continued-unused.mjs tools/unused.mjs\n"
            ),
            "tests/url-initializer.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "const unused=new URL('../tools/unused.mjs',import.meta.url).origin;\n"
                "const continued=new URL('../tools/continued-unused.mjs',import.meta.url)\n"
                "  .origin;\n"
                "spawnSync('echo',[live,unused,continued]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/continued-unused.mjs": False, "tools/live.mjs": True,
             "tools/unused.mjs": False},
        )

    def test_js_unreachable_child_process_calls_are_not_references(self):
        tree = {
            "tools/branch.mjs": "export const branch = true;\n",
            "tools/falseor-live.mjs": "export const live = true;\n",
            "tools/helper.mjs": "export const helper = true;\n",
            "tools/helperguard-unused.mjs": "export const unused = true;\n",
            "tools/short.mjs": "export const short = true;\n",
            "tools/trueand-live.mjs": "export const live = true;\n",
            "tools/trueor-unused.mjs": "export const unused = true;\n",
            "tools/unknown-unused.mjs": "export const unused = true;\n",
            "tools/README.md": (
                "tools/branch.mjs tools/falseor-live.mjs tools/helper.mjs "
                "tools/helperguard-unused.mjs tools/short.mjs "
                "tools/trueand-live.mjs tools/trueor-unused.mjs "
                "tools/unknown-unused.mjs\n"
            ),
            "tests/unreachable.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const branch=new URL('../tools/branch.mjs',import.meta.url);\n"
                "const falseor=new URL('../tools/falseor-live.mjs',import.meta.url);\n"
                "const helper=new URL('../tools/helper.mjs',import.meta.url);\n"
                "const helperguard=new URL('../tools/helperguard-unused.mjs',import.meta.url);\n"
                "const short=new URL('../tools/short.mjs',import.meta.url);\n"
                "const trueand=new URL('../tools/trueand-live.mjs',import.meta.url);\n"
                "const trueor=new URL('../tools/trueor-unused.mjs',import.meta.url);\n"
                "const unknown=new URL('../tools/unknown-unused.mjs',import.meta.url);\n"
                "if (false) { spawnSync(process.execPath,[branch]); }\n"
                "function unused() { spawnSync(process.execPath,[helper]); }\n"
                "function launch() { spawnSync(process.execPath,[helperguard]); }\n"
                "false && spawnSync(process.execPath,[short]);\n"
                "true && spawnSync(process.execPath,[trueand]);\n"
                "false || spawnSync(process.execPath,[falseor]);\n"
                "true || spawnSync(process.execPath,[trueor]);\n"
                "flag && spawnSync(process.execPath,[unknown]);\n"
                "false && launch();\n"
            ),
        }
        self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {
            "tools/branch.mjs": False, "tools/falseor-live.mjs": True,
            "tools/helper.mjs": False, "tools/helperguard-unused.mjs": False,
            "tools/short.mjs": False, "tools/trueand-live.mjs": True,
            "tools/trueor-unused.mjs": False, "tools/unknown-unused.mjs": False,
        })

    def test_js_conditional_expression_arms_require_proven_selection(self):
        tree = {
            "tools/false-else-live.mjs": "export const live = true;\n",
            "tools/false-then-unused.mjs": "export const unused = true;\n",
            "tools/true-else-unused.mjs": "export const unused = true;\n",
            "tools/true-then-live.mjs": "export const live = true;\n",
            "tools/unknown-unused.mjs": "export const unused = true;\n",
            "tools/README.md": (
                "tools/false-else-live.mjs tools/false-then-unused.mjs "
                "tools/true-else-unused.mjs tools/true-then-live.mjs "
                "tools/unknown-unused.mjs\n"
            ),
            "tests/conditional-activation.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const falseElse=new URL('../tools/false-else-live.mjs',import.meta.url);\n"
                "const falseThen=new URL('../tools/false-then-unused.mjs',import.meta.url);\n"
                "const trueElse=new URL('../tools/true-else-unused.mjs',import.meta.url);\n"
                "const trueThen=new URL('../tools/true-then-live.mjs',import.meta.url);\n"
                "const unknown=new URL('../tools/unknown-unused.mjs',import.meta.url);\n"
                "false ? spawnSync('echo',[falseThen]) : null;\n"
                "true ? spawnSync('echo',[trueThen]) : null;\n"
                "true ? null : spawnSync('echo',[trueElse]);\n"
                "false ? null : spawnSync('echo',[falseElse]);\n"
                "flag ? spawnSync('echo',[unknown]) : null;\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/false-else-live.mjs": True,
             "tools/false-then-unused.mjs": False,
             "tools/true-else-unused.mjs": False,
             "tools/true-then-live.mjs": True,
             "tools/unknown-unused.mjs": False},
        )

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

    def test_js_uninvoked_method_bodies_are_not_module_execution(self):
        tree = {
            "tools/class-unused.mjs": "export const value = 1;\n",
            "tools/object-unused.mjs": "export const value = 2;\n",
            "tools/live.mjs": "export const value = 3;\n",
            "tools/README.md": (
                "tools/class-unused.mjs tools/object-unused.mjs tools/live.mjs\n"),
            "tests/methods.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const classScript=new URL('../tools/class-unused.mjs',import.meta.url);\n"
                "const objectScript=new URL('../tools/object-unused.mjs',import.meta.url);\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "class Settings { launch() { spawnSync('echo',[classScript]); } }\n"
                "const holder={ launch() { spawnSync('echo',[objectScript]); } };\n"
                "spawnSync('echo',[live]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/class-unused.mjs": False, "tools/live.mjs": True,
             "tools/object-unused.mjs": False},
        )

    def test_js_uninvoked_computed_method_bodies_are_not_module_execution(self):
        tree = {
            "tools/class-unused.mjs": "export const value = 1;\n",
            "tools/object-unused.mjs": "export const value = 2;\n",
            "tools/README.md": "tools/class-unused.mjs tools/object-unused.mjs\n",
            "tests/computed-methods.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const classScript=new URL('../tools/class-unused.mjs',import.meta.url);\n"
                "const objectScript=new URL('../tools/object-unused.mjs',import.meta.url);\n"
                "class Settings { ['launch']() { spawnSync('echo',[classScript]); } }\n"
                "const holder={ ['launch']() { spawnSync('echo',[objectScript]); } };\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/class-unused.mjs": False,
             "tools/object-unused.mjs": False},
        )

    def test_js_uninvoked_nested_computed_method_body_is_not_module_execution(self):
        tree = {
            "tools/unused.mjs": "export const value = 1;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/nested-computed-method.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "class Settings { [['launch'][0]]() { spawnSync('echo',[script]); } }\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_uninvoked_quoted_method_bodies_are_not_module_execution(self):
        tree = {
            "tools/class-unused.mjs": "export const value = 1;\n",
            "tools/object-unused.mjs": "export const value = 2;\n",
            "tools/README.md": "tools/class-unused.mjs tools/object-unused.mjs\n",
            "tests/quoted-methods.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const classScript=new URL('../tools/class-unused.mjs',import.meta.url);\n"
                "const objectScript=new URL('../tools/object-unused.mjs',import.meta.url);\n"
                "class Settings { 'launch'() { spawnSync('echo',[classScript]); } }\n"
                "const holder={ 'launch'() { spawnSync('echo',[objectScript]); } };\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/class-unused.mjs": False,
             "tools/object-unused.mjs": False},
        )

    def test_js_process_call_after_await_is_not_proven_reachable(self):
        tree = {
            "tools/live.mjs": "export const value = 1;\n",
            "tools/unused.mjs": "export const value = 2;\n",
            "tools/README.md": "tools/live.mjs tools/unused.mjs\n",
            "tests/await-suspension.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "const unused=new URL('../tools/unused.mjs',import.meta.url);\n"
                "async function launch() {\n"
                "  spawnSync('echo',[live]);\n"
                "  await new Promise(() => {});\n"
                "  spawnSync('echo',[unused]);\n"
                "}\n"
                "launch();\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True, "tools/unused.mjs": False},
        )

    def test_js_unbraced_literal_false_branch_is_not_activated(self):
        tree = {
            **{"tools/p%d.mjs" % index: "export const value = %d;\n" % index
               for index in range(1, 11)},
            "tools/README.md": " ".join(
                "tools/p%d.mjs" % index for index in range(1, 11)) + "\n",
            "tests/unbraced-false.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const p1=new URL('../tools/p1.mjs',import.meta.url);\n"
                "const p2=new URL('../tools/p2.mjs',import.meta.url);\n"
                "const p3=new URL('../tools/p3.mjs',import.meta.url);\n"
                "const p4=new URL('../tools/p4.mjs',import.meta.url);\n"
                "const p5=new URL('../tools/p5.mjs',import.meta.url);\n"
                "const p6=new URL('../tools/p6.mjs',import.meta.url);\n"
                "const p7=new URL('../tools/p7.mjs',import.meta.url);\n"
                "const p8=new URL('../tools/p8.mjs',import.meta.url);\n"
                "const p9=new URL('../tools/p9.mjs',import.meta.url);\n"
                "const p10=new URL('../tools/p10.mjs',import.meta.url);\n"
                "const passthrough=value=>value;\n"
                "if(false) spawnSync('echo',[p1]);\n"
                "if(false) passthrough\n"
                "  (spawnSync('echo',[p2]));\n"
                "if(false)\n"
                "  if(flag) spawnSync('echo',[p3]);\n"
                "  else spawnSync('echo',[p4]);\n"
                "else spawnSync('echo',[p5]);\n"
                "spawnSync('echo',[p6]);\n"
                "if(true) spawnSync('echo',[p7]);\n"
                "if(true)\n"
                "  if(false) spawnSync('echo',[p8]);\n"
                "  else spawnSync('echo',[p9]);\n"
                "else spawnSync('echo',[p10]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/p1.mjs": False, "tools/p2.mjs": False,
             "tools/p3.mjs": False, "tools/p4.mjs": False,
             "tools/p5.mjs": True, "tools/p6.mjs": True,
             "tools/p7.mjs": True, "tools/p8.mjs": False,
             "tools/p9.mjs": True, "tools/p10.mjs": False})

    def test_js_literal_if_excludes_both_unselected_braced_arms(self):
        tree = {
            "tools/false-then-unused.mjs": "export const value = 1;\n",
            "tools/false-else-live.mjs": "export const value = 2;\n",
            "tools/true-then-live.mjs": "export const value = 3;\n",
            "tools/true-else-unused.mjs": "export const value = 4;\n",
            "tools/README.md": (
                "tools/false-then-unused.mjs tools/false-else-live.mjs "
                "tools/true-then-live.mjs tools/true-else-unused.mjs\n"),
            "tests/braced-literal-if.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const ft=new URL('../tools/false-then-unused.mjs',import.meta.url);\n"
                "const fe=new URL('../tools/false-else-live.mjs',import.meta.url);\n"
                "const tt=new URL('../tools/true-then-live.mjs',import.meta.url);\n"
                "const te=new URL('../tools/true-else-unused.mjs',import.meta.url);\n"
                "if(false){spawnSync('echo',[ft]);}else{spawnSync('echo',[fe]);}\n"
                "if(true){spawnSync('echo',[tt]);}else{spawnSync('echo',[te]);}\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/false-else-live.mjs": True,
             "tools/false-then-unused.mjs": False,
             "tools/true-else-unused.mjs": False,
             "tools/true-then-live.mjs": True},
        )

    def test_js_nested_literal_if_checks_every_enclosing_branch(self):
        tree = {
            "tools/nested-unused.mjs": "export const value = 1;\n",
            "tools/nested-live.mjs": "export const value = 2;\n",
            "tools/README.md": "tools/nested-unused.mjs tools/nested-live.mjs\n",
            "tests/nested-literal-if.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const unused=new URL('../tools/nested-unused.mjs',import.meta.url);\n"
                "const live=new URL('../tools/nested-live.mjs',import.meta.url);\n"
                "if(true){if(false){spawnSync('echo',[unused]);}}\n"
                "if(true){if(true){spawnSync('echo',[live]);}}\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/nested-live.mjs": True,
             "tools/nested-unused.mjs": False},
        )

    def test_js_loop_body_requires_proven_execution(self):
        tree = {
            "tools/false-unused.mjs": "export const value = 1;\n",
            "tools/unknown-unused.mjs": "export const value = 2;\n",
            "tools/true-live.mjs": "export const value = 3;\n",
            "tools/for-false-unused.mjs": "export const value = 4;\n",
            "tools/for-true-live.mjs": "export const value = 5;\n",
            "tools/README.md": (
                "tools/false-unused.mjs tools/unknown-unused.mjs "
                "tools/true-live.mjs tools/for-false-unused.mjs "
                "tools/for-true-live.mjs\n"),
            "tests/loop-reachability.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const falseUnused=new URL('../tools/false-unused.mjs',import.meta.url);\n"
                "const unknownUnused=new URL('../tools/unknown-unused.mjs',import.meta.url);\n"
                "const trueLive=new URL('../tools/true-live.mjs',import.meta.url);\n"
                "const forFalseUnused=new URL('../tools/for-false-unused.mjs',import.meta.url);\n"
                "const forTrueLive=new URL('../tools/for-true-live.mjs',import.meta.url);\n"
                "while(false){spawnSync('echo',[falseUnused]);}\n"
                "while(flag){spawnSync('echo',[unknownUnused]);}\n"
                "while(true){spawnSync('echo',[trueLive]);break;}\n"
                "for(;false;){spawnSync('echo',[forFalseUnused]);}\n"
                "for(;;){spawnSync('echo',[forTrueLive]);break;}\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/false-unused.mjs": False,
             "tools/for-false-unused.mjs": False,
             "tools/for-true-live.mjs": True,
             "tools/true-live.mjs": True,
             "tools/unknown-unused.mjs": False},
        )

    def test_js_loop_termination_precedes_process_activation(self):
        tree = {
            "tools/break-unused.mjs": "export const value = 1;\n",
            "tools/continue-unused.mjs": "export const value = 2;\n",
            "tools/live.mjs": "export const value = 3;\n",
            "tools/README.md": (
                "tools/break-unused.mjs tools/continue-unused.mjs tools/live.mjs\n"),
            "tests/loop-termination.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const breakUnused=new URL('../tools/break-unused.mjs',import.meta.url);\n"
                "const continueUnused=new URL('../tools/continue-unused.mjs',import.meta.url);\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "while(true){break;spawnSync('echo',[breakUnused]);}\n"
                "for(;;){continue;spawnSync('echo',[continueUnused]);}\n"
                "while(true){spawnSync('echo',[live]);break;}\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/break-unused.mjs": False,
             "tools/continue-unused.mjs": False,
             "tools/live.mjs": True},
        )

    def test_js_switch_requires_proven_case_selection(self):
        tree = {
            "tools/unused.mjs": "export const value = 1;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/switch-reachability.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "switch(0){case 1:spawnSync('echo',[script]);}\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_helper_termination_precedes_process_activation(self):
        tree = {
            "tools/return-unused.mjs": "export const value = 1;\n",
            "tools/throw-unused.mjs": "export const value = 2;\n",
            "tools/live.mjs": "export const value = 3;\n",
            "tools/README.md": (
                "tools/return-unused.mjs tools/throw-unused.mjs tools/live.mjs\n"),
            "tests/helper-termination.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const returnUnused=new URL('../tools/return-unused.mjs',import.meta.url);\n"
                "const throwUnused=new URL('../tools/throw-unused.mjs',import.meta.url);\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "function afterReturn(){return;spawnSync('echo',[returnUnused]);}\n"
                "function afterThrow(){throw new Error('stop');spawnSync('echo',[throwUnused]);}\n"
                "function beforeReturn(){spawnSync('echo',[live]);return;}\n"
                "afterReturn();afterThrow();beforeReturn();\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True,
             "tools/return-unused.mjs": False,
             "tools/throw-unused.mjs": False},
        )

    def test_js_nullish_coalescing_requires_rhs_evaluation(self):
        tree = {
            "tools/skipped.mjs": "export const value = 1;\n",
            "tools/live.mjs": "export const value = 2;\n",
            "tools/README.md": "tools/skipped.mjs tools/live.mjs\n",
            "tests/nullish-reachability.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const skipped=new URL('../tools/skipped.mjs',import.meta.url);\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "true ?? spawnSync('echo',[skipped]);\n"
                "undefined ?? spawnSync('echo',[live]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True, "tools/skipped.mjs": False},
        )

    def test_js_optional_call_arguments_require_proven_evaluation(self):
        tree = {
            "tools/unused.mjs": "export const value = 1;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/optional-call.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "const invoke=null;\n"
                "invoke?.(spawnSync('echo',[script]));\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_optional_property_chains_require_proven_evaluation(self):
        tree = {
            "tools/property-unused.mjs": "export const value = 1;\n",
            "tools/computed-unused.mjs": "export const value = 2;\n",
            "tools/README.md": (
                "tools/property-unused.mjs tools/computed-unused.mjs\n"
            ),
            "tests/optional-property.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const propertyScript=new URL('../tools/property-unused.mjs',import.meta.url);\n"
                "const computedScript=new URL('../tools/computed-unused.mjs',import.meta.url);\n"
                "const target=null;\n"
                "target?.launch(spawnSync('echo',[propertyScript]));\n"
                "target?.[spawnSync('echo',[computedScript])];\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/computed-unused.mjs": False,
             "tools/property-unused.mjs": False},
        )

    def test_js_continuous_optional_property_chain_requires_proven_evaluation(self):
        tree = {
            "tools/unused.mjs": "export const value = 1;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/continuous-optional-property.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "const target=null;\n"
                "target?.nested.launch(spawnSync('echo',[script]));\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_continuous_optional_computed_and_call_chains_require_evaluation(self):
        tree = {
            "tools/computed-unused.mjs": "export const value = 1;\n",
            "tools/call-unused.mjs": "export const value = 2;\n",
            "tools/README.md": "tools/computed-unused.mjs tools/call-unused.mjs\n",
            "tests/continuous-optional-segments.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const computed=new URL('../tools/computed-unused.mjs',import.meta.url);\n"
                "const called=new URL('../tools/call-unused.mjs',import.meta.url);\n"
                "const target=null;\n"
                "target?.['nested'].launch(spawnSync('echo',[computed]));\n"
                "target?.nested().launch(spawnSync('echo',[called]));\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/call-unused.mjs": False,
             "tools/computed-unused.mjs": False},
        )

    def test_js_unconstructed_instance_field_initializer_is_deferred(self):
        tree = {
            "tools/unused.mjs": "export const value = 1;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/instance-field.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "class Settings { field=spawnSync('echo',[script]); }\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_unconstructed_computed_instance_field_initializer_is_deferred(self):
        tree = {
            "tools/unused.mjs": "export const value = 1;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/computed-instance-field.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "class Settings { ['field'] = spawnSync('echo',[script]); }\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_unconstructed_nested_computed_instance_field_is_deferred(self):
        tree = {
            "tools/unused.mjs": "export const value = 1;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/nested-computed-instance-field.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "class Settings { [['field'][0]] = spawnSync('echo',[script]); }\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_static_block_computed_assignment_executes_at_class_definition(self):
        tree = {
            "tools/live.mjs": "export const value = 1;\n",
            "tools/README.md": "tools/live.mjs\n",
            "tests/static-block-computed-assignment.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/live.mjs',import.meta.url);\n"
                "const holder=[0];\n"
                "class Settings {\n"
                "  static { [holder[0]] = [spawnSync('echo',[script])]; }\n"
                "}\n"
            ),
        }
        self.assertTrue(self.tools(tree)["tools/live.mjs"]["referenced"])

    def test_js_semicolonless_instance_field_boundary_is_deferred(self):
        tree = {
            "tools/unused.mjs": "export const value = 1;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/semicolonless-instance-fields.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "class Settings { field = 0\n"
                " other = spawnSync('echo',[script]); }\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_regular_expression_literal_is_not_a_process_call(self):
        tree = {
            "tools/regex-unused.mjs": "export const value = 1;\n",
            "tools/live.mjs": "export const value = 2;\n",
            "tools/README.md": "tools/regex-unused.mjs tools/live.mjs\n",
            "tests/regex-literal.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const regexUnused=new URL('../tools/regex-unused.mjs',import.meta.url);\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "const pattern=/spawnSync('echo',[regexUnused])/;\n"
                "spawnSync('echo',[live]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True, "tools/regex-unused.mjs": False},
        )

    def test_js_regular_expression_after_control_header_is_not_a_process_call(self):
        tree = {
            "tools/regex-unused.mjs": "export const value = 1;\n",
            "tools/live.mjs": "export const value = 2;\n",
            "tools/README.md": "tools/regex-unused.mjs tools/live.mjs\n",
            "tests/regex-after-if.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const regexUnused=new URL('../tools/regex-unused.mjs',import.meta.url);\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "if (true) /spawnSync('echo',[regexUnused])/;\n"
                "spawnSync('echo',[live]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True, "tools/regex-unused.mjs": False},
        )

    def test_js_completed_standalone_block_propagates_termination(self):
        tree = {
            "tools/unused.mjs": "export const value = 1;\n",
            "tools/live.mjs": "export const value = 2;\n",
            "tools/README.md": "tools/unused.mjs tools/live.mjs\n",
            "tests/completed-block-termination.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const unused=new URL('../tools/unused.mjs',import.meta.url);\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "function stopped(){{return;}spawnSync('echo',[unused]);}\n"
                "function running(){{spawnSync('echo',[live]);}return;}\n"
                "stopped();running();\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True, "tools/unused.mjs": False},
        )

    def test_js_quoted_url_declaration_is_not_a_binding(self):
        tree = {
            "tools/live.mjs": "export const value = 1;\n",
            "tools/unused.mjs": "export const value = 2;\n",
            "tools/README.md": "tools/live.mjs tools/unused.mjs\n",
            "tests/quoted-url.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=\"const script=new URL('../tools/unused.mjs',import.meta.url);\";\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "spawnSync(process.execPath,[script,live]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True, "tools/unused.mjs": False})

    def test_js_quoted_child_process_import_is_not_trusted(self):
        tree = {
            "tools/unused.mjs": "export const value = 1;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/quoted-import.test.mjs": (
                "const quoted=\"import { spawnSync } from 'node:child_process';\";\n"
                "const spawnSync=()=>{};\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "spawnSync('echo',[script]);\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_dead_url_initializer_does_not_create_provenance(self):
        tree = {
            "tools/unused.mjs": "export const value = 1;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/dead-url-initializer.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "var script='other';\n"
                "if(false){var script=new URL('../tools/unused.mjs',import.meta.url);}\n"
                "spawnSync('echo',[script]);\n"
            ),
        }
        self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])

    def test_js_url_alias_mutation_invalidates_original_provenance(self):
        tree = {
            "tools/live.mjs": "export const value = 1;\n",
            "tools/unused.mjs": "export const value = 2;\n",
            "tools/README.md": "tools/live.mjs tools/unused.mjs\n",
            "tests/url-alias.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "const script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "const alias=script;\n"
                "alias.pathname='/other.py';\n"
                "const live=new URL('../tools/live.mjs',import.meta.url);\n"
                "spawnSync('echo',[script,live]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/live.mjs": True, "tools/unused.mjs": False})

    def test_js_concise_arrow_body_is_deferred_until_invocation(self):
        tree = {
            "tools/arbitrary-unused.mjs": "export const value = 1;\n",
            "tools/callback-live.mjs": "export const value = 2;\n",
            "tools/direct-live.mjs": "export const value = 3;\n",
            "tools/helper-unused.mjs": "export const value = 4;\n",
            "tools/shadowed-unused.mjs": "export const value = 5;\n",
            "tools/README.md": (
                "tools/arbitrary-unused.mjs tools/callback-live.mjs "
                "tools/direct-live.mjs tools/helper-unused.mjs "
                "tools/shadowed-unused.mjs\n"),
            "tests/concise-arrow.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "import test from 'node:test';\n"
                "const arbitraryUnused=new URL('../tools/arbitrary-unused.mjs',import.meta.url);\n"
                "const callbackLive=new URL('../tools/callback-live.mjs',import.meta.url);\n"
                "const directLive=new URL('../tools/direct-live.mjs',import.meta.url);\n"
                "const helperUnused=new URL('../tools/helper-unused.mjs',import.meta.url);\n"
                "const shadowedUnused=new URL('../tools/shadowed-unused.mjs',import.meta.url);\n"
                "const launch=()=>spawnSync('echo',[helperUnused]);\n"
                "const arbitrary=(name,callback)=>callback;\n"
                "arbitrary('x',()=>spawnSync('echo',[arbitraryUnused]));\n"
                "test('x',()=>spawnSync('echo',[callbackLive]));\n"
                "function exercise(test){\n"
                "  test('x',()=>spawnSync('echo',[shadowedUnused]));\n"
                "}\n"
                "exercise((name,callback)=>callback);\n"
                "spawnSync('echo',[directLive]);\n"
            ),
        }
        self.assertEqual(
            {path: item["referenced"] for path, item in self.tools(tree).items()},
            {"tools/arbitrary-unused.mjs": False,
             "tools/callback-live.mjs": True,
             "tools/direct-live.mjs": True,
             "tools/helper-unused.mjs": False,
             "tools/shadowed-unused.mjs": False})

    def test_js_spaced_block_arrow_has_one_block_scope(self):
        source = "const callback=()=> { return true; };"
        searchable, scopes = g11._js_function_scopes(source)
        self.assertEqual(scopes, [(16, 16, 20, 35)])
        self.assertEqual(searchable[scopes[0][2]], "{")

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

    def test_js_deferred_helper_uses_activation_time_url_binding(self):
        tree = {
            "tools/unused.mjs": "export const unused = true;\n",
            "tools/README.md": "tools/unused.mjs\n",
            "tests/deferred-binding.test.mjs": (
                "import {spawnSync} from 'node:child_process';\n"
                "let script=new URL('../tools/unused.mjs',import.meta.url);\n"
                "function exercise() { spawnSync('echo',[script]); }\n"
                "script = 'other.mjs';\n"
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
            "tests/test_runner.py": ("import subprocess\n"
                                     "subprocess.run(['python', '-c', 'tools/runner.py'])\n"),
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
        self.assertEqual(len(cases), 298)
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
