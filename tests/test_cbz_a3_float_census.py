"""Behavioural and mutation self-tests for tools/cbz/a3_float_census.py.

The census under test is loaded from $CBZ_A3_CENSUS when set (the mutation
test below points it at mutated copies), else from tools/cbz/.
"""
import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)
REAL = os.path.join(WORKSPACE, "tools", "cbz", "a3_float_census.py")
TOOL = os.environ.get("CBZ_A3_CENSUS") or REAL


def _load():
    spec = importlib.util.spec_from_file_location("a3_census_under_test", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


def kinds(src):
    r = M.census_text(src, "x.py")
    return r["class"], [(s["line"], s["kind"], s["detail"]) for s in r["signals"]]


class Signals(unittest.TestCase):
    def test_float_literals(self):
        self.assertEqual(kinds("x = 0.5\ny = 1e-9\nz = 2j\n"),
                         ("FLOAT_SIGNAL", [(1, "float_literal", "0.5"), (2, "float_literal", "1e-09"),
                                           (3, "float_literal", "2j")]))

    def test_int_and_bool_literals_are_not_float(self):
        self.assertEqual(kinds("x = 1\ny = True\nz = 10**9\n"), ("NO_FLOAT_SIGNAL", []))

    def test_float_and_complex_calls(self):
        self.assertEqual(kinds("a = float(x)\nb = complex(1, 2)\n"),
                         ("FLOAT_SIGNAL", [(1, "float_call", "float"), (2, "float_call", "complex")]))

    def test_true_division_and_augmented(self):
        self.assertEqual(kinds("a = p / q\nb //= 2\nc /= 3\n"),
                         ("FLOAT_SIGNAL", [(1, "true_division", "/"), (3, "true_division", "/=")]))

    def test_floor_division_is_exact(self):
        self.assertEqual(kinds("a = p // q\nb = p % q\n"), ("NO_FLOAT_SIGNAL", []))

    def test_math_float_and_exact_math(self):
        self.assertEqual(kinds("import math\nr = math.sqrt(2)\nk = math.isqrt(9)\nc = math.comb(5, 2)\n"),
                         ("FLOAT_SIGNAL", [(2, "math_float", "math.sqrt")]))

    def test_math_alias_and_from_import(self):
        self.assertEqual(kinds("import math as m\nr = m.pi\n"), ("FLOAT_SIGNAL", [(2, "math_float", "math.pi")]))
        self.assertEqual(kinds("from math import exp, gcd\n"),
                         ("FLOAT_SIGNAL", [(1, "math_float", "from math import exp")]))

    def test_unrelated_attribute_named_like_math(self):
        self.assertEqual(kinds("obj.sqrt(2)\nmathish.sqrt(2)\n"), ("NO_FLOAT_SIGNAL", []))

    def test_float_modules(self):
        self.assertEqual(kinds("import numpy as np\n"), ("FLOAT_SIGNAL", [(1, "float_module", "numpy")]))
        self.assertEqual(kinds("from decimal import Decimal\n"), ("FLOAT_SIGNAL", [(1, "float_module", "decimal")]))
        self.assertEqual(kinds("import scipy.special\n"), ("FLOAT_SIGNAL", [(1, "float_module", "scipy")]))

    def test_exact_modules_and_mixed(self):
        self.assertEqual(kinds("from fractions import Fraction\nx = Fraction(1, 3)\n"),
                         ("NO_FLOAT_SIGNAL", [(1, "exact_module", "fractions")]))
        self.assertEqual(kinds("from fractions import Fraction\nx = Fraction(1, 3) / 2\n"),
                         ("MIXED", [(1, "exact_module", "fractions"), (2, "true_division", "/")]))
        self.assertEqual(kinds("import sympy\n"), ("NO_FLOAT_SIGNAL", [(1, "exact_module", "sympy")]))

    def test_mpmath_interval(self):
        cls, sig = kinds("import mpmath\nx = mpmath.iv.mpf(1)\n")
        self.assertEqual(cls, "MIXED")
        self.assertIn((1, "float_module", "mpmath"), sig)
        self.assertIn((2, "exact_module", "mpmath.iv"), sig)
        self.assertIn((1, "exact_module", "mpmath.iv"), kinds("from mpmath import iv\n")[1])

    def test_interval_layer(self):
        cls, sig = kinds("import math\n_na = math.nextafter\nx = 0.5\n")
        self.assertEqual(cls, "MIXED")
        self.assertIn((2, "interval_layer", "math.nextafter"), sig)
        cls, sig = kinds("from ia import dn, up\nx = 0.5\n")
        self.assertEqual((cls, sig[0]), ("MIXED", (1, "interval_layer", "from ia import")))
        cls, sig = kinds("import ia\nx = 1 / 3\n")
        self.assertEqual((cls, sig[0]), ("MIXED", (1, "interval_layer", "import ia")))
        cls, sig = kinds("from math import ulp\n")
        self.assertIn((1, "interval_layer", "math.ulp"), sig)
        self.assertEqual(kinds("import iax\nx = 0.5\n")[0], "FLOAT_SIGNAL")

    def test_strings_and_comments_are_not_code(self):
        self.assertEqual(kinds("s = '0.5 / 2 float(x)'\n# x = 0.5\n"), ("NO_FLOAT_SIGNAL", []))

    def test_unparseable_fails_closed(self):
        r = M.census_text("def f(:\n", "bad.py")
        self.assertEqual(r["class"], "UNPARSEABLE")


class CommandLine(unittest.TestCase):
    def run_cli(self, args):
        return subprocess.run([sys.executable, "-I", TOOL] + list(args), capture_output=True, text=True)

    def test_directory_census_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.py"), "w") as fh:
                fh.write("x = 1 / 3\n")
            with open(os.path.join(d, "b.py"), "w") as fh:
                fh.write("x = 1\n")
            out = os.path.join(d, "o.json")
            r = self.run_cli([d, "--json", out])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            with open(out) as fh:
                data = json.load(fh)
            self.assertEqual(data["summary"], {"files": 2, "by_class": {"FLOAT_SIGNAL": 1, "NO_FLOAT_SIGNAL": 1}})

    def test_unparseable_exit_one(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.py"), "w") as fh:
                fh.write("def f(:\n")
            self.assertEqual(self.run_cli([d]).returncode, 1)

    def test_git_mode(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
            git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@x",
                                             "-c", "commit.gpgsign=false"] + list(a), cwd=d,
                                            check=True, capture_output=True, env=env)
            git("init", "-q")
            with open(os.path.join(d, "a.py"), "w") as fh:
                fh.write("x = 0.25\n")
            git("add", "-A")
            git("commit", "-q", "-m", "c")
            out = os.path.join(d, "o.json")
            r = self.run_cli(["--git", d, "--rev", "HEAD", "--json", out])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            with open(out) as fh:
                data = json.load(fh)
            self.assertEqual(data["summary"]["files"], 1)
            self.assertEqual(data["files"][0]["class"], "FLOAT_SIGNAL")
            self.assertEqual(len(data["files"][0]["blob"]), 40)

    def test_usage(self):
        self.assertEqual(self.run_cli([]).returncode, 2)


MUTANTS = [
    ("float literals ignored", "if isinstance(node.value, (float, complex)) and not isinstance(node.value, bool):",
     "if False:"),
    ("float() calls ignored", 'node.func.id in ("float", "complex")', 'node.func.id in ()'),
    ("true division ignored", "        if isinstance(node.op, ast.Div):\n            self.add(node, \"true_division\", \"/\")",
     "        if False:\n            self.add(node, \"true_division\", \"/\")"),
    ("augmented division ignored", 'self.add(node, "true_division", "/=")', "pass"),
    ("math attributes ignored", "if base in self.math_aliases and node.attr not in EXACT_MATH:", "if False:"),
    ("from-math imports ignored", 'self.add(node, "math_float", f"from {mod} import {a.name}")', "pass"),
    ("sqrt treated as exact", '"comb", "factorial",', '"comb", "factorial", "sqrt",'),
    ("float modules ignored", 'FLOAT_MODULES = {"numpy", "scipy", "mpmath", "decimal", "cmath"}',
     'FLOAT_MODULES = set()'),
    ("exact modules ignored", 'EXACT_MODULES = {"fractions", "sympy", "flint", "gmpy2"}', "EXACT_MODULES = set()"),
    ("interval layer ignored", 'counts.get("exact_module") or counts.get("interval_layer")',
     'counts.get("exact_module")'),
    ("nextafter attribute ignored", "if base in self.math_aliases and node.attr in INTERVAL_MATH:", "if False:"),
    ("ia import ignored", 'self.add(node, "interval_layer", f"from {mod} import")', "pass"),
    ("mixed collapsed", '("MIXED" if has_exact else "FLOAT_SIGNAL")', '"FLOAT_SIGNAL"'),
    ("unparseable passes", 'return 1 if by_class.get("UNPARSEABLE") else 0', "return 0"),
    ("git mode lists nothing", 'if typ == b"blob" and p.endswith(".py"):', "if False:"),
]


# Additional source mutations for the canonical input/lexer repairs.
MUTANTS.extend([('missing root ignored', 'raise OSError(f"source path does not exist: {p}")', 'pass'), ('source symlink accepted', 'raise OSError(f"source symlink requires explicit target admission: {p}")', 'pass'), ('directory symlink skipped', 'raise OSError(f"source directory symlink requires explicit target admission: {d}")', 'pass'), ('gitlink skipped', 'raise OSError(f"gitlink requires a separately pinned scan: {p}")', 'pass'), ('git source symlink accepted', 'if mode not in (b"100644", b"100755"):', 'if False:'), ('git batch header unchecked', 'if len(header) != 3 or header[0] != sha.encode() or header[1] != b"blob" or not header[2].isdigit():', 'if False:'), ('git batch body unchecked', 'if len(data) != size or trailer != b"\\n":', 'if False:'), ('git batch exit unchecked', 'if code != 0:', 'if False:')])

EXPECTED_FAILURE = {'float literals ignored': 'test_float_literals', 'float() calls ignored': 'test_float_and_complex_calls', 'true division ignored': 'test_true_division_and_augmented', 'augmented division ignored': 'test_true_division_and_augmented', 'math attributes ignored': 'test_math_float_and_exact_math', 'from-math imports ignored': 'test_math_alias_and_from_import', 'sqrt treated as exact': 'test_math_float_and_exact_math', 'float modules ignored': 'test_float_modules', 'exact modules ignored': 'test_exact_modules_and_mixed', 'interval layer ignored': 'test_interval_layer', 'nextafter attribute ignored': 'test_interval_layer', 'ia import ignored': 'test_interval_layer', 'mixed collapsed': 'test_exact_modules_and_mixed', 'unparseable passes': 'test_unparseable_exit_one', 'git mode lists nothing': 'test_git_mode'}

EXPECTED_FAILURE.update({'missing root ignored': 'test_missing_root_fails_for_both_tools', 'source symlink accepted': 'test_source_symlink_fails_for_both_tools', 'directory symlink skipped': 'test_symlink_directory_is_not_silently_skipped', 'gitlink skipped': 'test_gitlink_is_explicitly_rejected_by_both_tools', 'git source symlink accepted': 'test_git_source_symlink_fails_for_both_tools', 'git batch header unchecked': 'test_malformed_blob_transport_fails_closed', 'git batch body unchecked': 'test_malformed_blob_transport_fails_closed', 'git batch exit unchecked': 'test_malformed_blob_transport_fails_closed'})


# ULW#29 review parity (main#698/6545133621)
MUTANTS.extend([
    ('aliases not pre-collected', '    v.collect_aliases(tree)\n', ''),
    ('parent-package ia ignored', '        elif any(a.name == "ia" for a in node.names):', '        elif False:'),
    ('source decoded as utf-8 text', '            records.append(census_text(data, rel))', "            records.append(census_text(data.decode('utf-8', 'replace').lstrip('\\ufeff'), rel))"),
    ('git source decoded strictly', '                records.append(census_text(data, p, sha))', "                records.append(census_text(data.decode('utf-8'), p, sha))"),
])
EXPECTED_FAILURE.update({'aliases not pre-collected': 'test_census_alias_imported_after_use', 'parent-package ia ignored': 'test_census_interval_module_from_parent_package', 'source decoded as utf-8 text': 'test_census_declared_encodings', 'git source decoded strictly': 'test_census_declared_encodings'})

TEST_MODULES = ('tests.test_cbz_a3_float_census', 'tests.test_cbz_auditor_hardening')


def designated_test_ids(modules=TEST_MODULES):
    """Map each test method name to its unique module.Class.method id."""
    ids = {}
    for module in modules:
        with open(os.path.join(WORKSPACE, *module.split(".")) + ".py", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef):
                for node in cls.body:
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                        if node.name in ids:
                            raise ValueError("ambiguous designated test: " + node.name)
                        ids[node.name] = ".".join((module, cls.name, node.name))
    return ids


def run_suite(tool_path, test_id=None):
    env = dict(os.environ, CBZ_A3_CENSUS=tool_path, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, *(["-O"] if sys.flags.optimize else []), "-B", "-S", "-m", "unittest", "-v",
                           *([test_id] if test_id else TEST_MODULES)],
                          cwd=WORKSPACE, env=env, capture_output=True, text=True, timeout=600)


def killed(r, expected_test=None):
    """A kill is exit 1 from the designated test run alone, failing only by its own assertions.

    Launch/import/syntax failures, errors, skips, zero or extra tests, foreign
    failures and failures raised outside the designated method are not kills.
    Nor is an AssertionError raised inside the tool under test: the innermost
    traceback frame must be test code, so a crash counts the same whatever its
    exception type.
    """
    out = r.stdout + r.stderr
    if expected_test is None or r.returncode != 1:
        return False
    runs = re.findall(r"^Ran [^\n]*$", out, re.M)
    summary = re.findall(r"^FAILED \(failures=([1-9][0-9]*)\)$", out, re.M)
    if (len(runs) != 1 or not re.fullmatch(r"Ran 1 test in [0-9]+(?:\.[0-9]+)?s", runs[0])
            or len(summary) != 1 or re.search(r"^FAILED \((?!failures=[1-9][0-9]*\)$)", out, re.M)):
        return False
    failures = 0
    for block in re.split(r"(?m)^={70}$", out)[1:]:
        header = re.match(r"\n(FAIL|ERROR): (test_[A-Za-z0-9_]+) \(", block)
        if header is None:
            continue
        if (header.group(1) != "FAIL" or header.group(2) != expected_test
                or not re.search(r'^  File "[^"\n]+", line [0-9]+, in %s$' % re.escape(expected_test), block, re.M)
                or not re.search(r"^AssertionError\b", block, re.M)):
            return False
        frames = re.findall(r'^  File "([^"\n]+)", line [0-9]+, in ', block, re.M)
        if not frames or not re.search(r"[/\\]tests[/\\][^/\\]+\.py$", frames[-1]):
            return False
        failures += 1
    return failures == int(summary[0])


@unittest.skipIf(os.environ.get("CBZ_A3_CENSUS"), "mutant run: do not recurse")
class SelfMutation(unittest.TestCase):
    def test_harness_control_unmutated_copy_passes(self):
        # without this control, a harness that cannot load the tool would "kill" every mutant
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "control.py")
            shutil.copyfile(REAL, path)
            r = run_suite(path)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_every_mutant_is_killed(self):
        with open(REAL, encoding="utf-8") as fh:
            src = fh.read()
        survivors = []
        ids = designated_test_ids()
        with tempfile.TemporaryDirectory() as d:
            for i, (name, old, new) in enumerate(MUTANTS):
                self.assertEqual(src.count(old), 1, f"mutation point drifted: {name}")
                path = os.path.join(d, f"m{i:02d}.py")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(src.replace(old, new))
                if not killed(run_suite(path, ids[EXPECTED_FAILURE[name]]), EXPECTED_FAILURE[name]):
                    survivors.append(name)
        self.assertEqual(survivors, [])


if __name__ == "__main__":
    unittest.main()
