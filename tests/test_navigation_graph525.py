"""Adverse controls for the navigation job's pinned #525 composition.

The shell fixtures simulate process results; actual Git and test execution
are separately required and are not established by these controls.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/navigation.yml"
SETUP = "      - name: Recover pinned graph-binding source\n        run: |\n"
GUARD = "      - name: Verify pinned graph-binding composition\n        run: |\n"
MAIN_COMMIT = "29b7d5969e2cd707debb7449c69a71bbcd39cf63"
MATH_COMMIT = "dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde"
MATH_URL = "https://github.com/d6g8k5htny-coder/Math-.git"
NORMAL_LINE = ("test_525_composition (test_graph_binding_check.Composition525Test."
               "test_525_composition) ... ok\n")
OPTIMIZED_LINE = ("test_525_composition (tests.test_graph_binding_check.Composition525Test."
                  "test_525_composition) ... ok\n")
GOOD = OPTIMIZED_LINE + "\nRan 1 test in 0.001s\n\nOK\n"

DRIVER = r'''import json, os, pathlib, sys
args = sys.argv[1:]
with pathlib.Path(os.environ["FIXTURE_TRACE"]).open("a") as stream:
    stream.write(json.dumps(args) + "\n")
if os.environ.get("FIXTURE_KIND") == "git":
    op = args[2] if args and args[0] == "-C" else args[0]
    key = ("math-" if args and args[0] == "-C" else "main-") + op
    if os.environ.get("FIXTURE_FAIL") in (op, key):
        raise SystemExit(23)
    if op == "init":
        pathlib.Path(args[-1]).mkdir(exist_ok=True)
else:
    data = json.loads(pathlib.Path(os.environ["FIXTURE_RESULT"]).read_text())
    sys.stdout.write(data["output"])
    raise SystemExit(data["exit"])
'''


class NavigationGraph525Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.bash = shutil.which("bash")
        if not cls.bash:
            raise RuntimeError("Bash is required for workflow controls")

    def block(self, marker):
        self.assertEqual(self.workflow.count(marker), 1, "workflow step missing or ambiguous")
        block = self.workflow.split(marker, 1)[1].split("\n      - ", 1)[0]
        lines = block.splitlines()
        self.assertTrue(lines and all(not line or line.startswith("          ") for line in lines))
        return "\n".join(line[10:] if line else "" for line in lines) + "\n"

    def run_block(self, marker, *, failure="", existing=False, normal=NORMAL_LINE,
                  output=GOOD, exit_status=0, script_text=None):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "navigation-evidence").mkdir()
            (root / "navigation-evidence/tests-normal.log").write_text(normal)
            runner = root / "runner"
            runner.mkdir()
            if existing:
                (runner / "cbz-graph525-math.git").mkdir()
            env_file = root / "github-env"
            env_file.write_text("")
            binary = root / "bin"
            binary.mkdir()
            driver = root / "driver.py"
            driver.write_text(DRIVER)
            name = "git" if marker == SETUP else "python"
            shim = binary / name
            shim.write_text('#!/bin/sh\nexec "$FIXTURE_PYTHON" -B -S "$FIXTURE_DRIVER" "$@"\n')
            shim.chmod(0o700)
            fixture = root / "result.json"
            fixture.write_text(json.dumps({"output": output, "exit": exit_status}))
            script = root / "run.sh"
            script.write_text(self.block(marker) if script_text is None else script_text)
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("PYTHON") and k not in ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS")}
            env.update(PATH=str(binary) + os.pathsep + os.defpath,
                       RUNNER_TEMP=str(runner), GITHUB_ENV=str(env_file),
                       FIXTURE_PYTHON=sys.executable, FIXTURE_DRIVER=str(driver),
                       FIXTURE_TRACE=str(root / "trace.jsonl"), FIXTURE_RESULT=str(fixture),
                       FIXTURE_KIND=name, FIXTURE_FAIL=failure)
            run = subprocess.run([self.bash, "--noprofile", "--norc", "-e", str(script)],
                                 cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 capture_output=True, text=True, timeout=15)
            trace_path = root / "trace.jsonl"
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
            log = root / "navigation-evidence/graph525-optimized.log"
            return run, trace, env_file.read_text(), log.read_text() if log.exists() else None

    def test_setup_fetches_exact_objects_before_export(self):
        run, trace, exported, _ = self.run_block(SETUP)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(len(trace), 4)
        path = trace[0][-1]
        self.assertEqual(trace[0], ["init", "--bare", path])
        self.assertEqual(trace[1], ["-C", path, "fetch", "--no-tags", "--no-write-fetch-head",
                                    "--depth=1", MATH_URL, MATH_COMMIT])
        self.assertEqual(trace[2], ["-C", path, "cat-file", "-e", MATH_COMMIT + "^{commit}"])
        self.assertEqual(trace[3], ["cat-file", "-e", MAIN_COMMIT + "^{commit}"])
        self.assertEqual(exported, "GB_MATH_CHECKOUT=" + path + "\n")

    def test_missing_source_objects_never_export_success(self):
        for failure in ("fetch", "math-cat-file", "main-cat-file"):
            with self.subTest(failure=failure):
                run, _, exported, _ = self.run_block(SETUP, failure=failure)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(exported, "")

    def test_existing_destination_is_preserved_and_refused(self):
        run, trace, exported, _ = self.run_block(SETUP, existing=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(trace, [])
        self.assertEqual(exported, "")

    def test_complete_actual_test_shape_is_accepted(self):
        run, trace, _, log = self.run_block(GUARD)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(trace, [["-B", "-O", "-S", "-m", "unittest", "-v",
                                  "tests.test_graph_binding_check.Composition525Test"]])
        self.assertEqual(log, GOOD)

    def test_missing_normal_or_skipped_optimized_is_rejected(self):
        for normal, output in (("", GOOD),
                               (NORMAL_LINE, GOOD.replace(" ... ok\n", " ... skipped 'missing'\n", 1)),
                               (NORMAL_LINE, GOOD.replace("\nOK\n", "\nOK (skipped=1)\n"))):
            with self.subTest(normal=normal, output=output):
                run, _, _, _ = self.run_block(GUARD, normal=normal, output=output)
                self.assertNotEqual(run.returncode, 0)

    def test_child_error_or_malformed_final_count_is_rejected(self):
        run, _, _, log = self.run_block(GUARD, exit_status=23)
        self.assertEqual(run.returncode, 23)
        self.assertEqual(log, GOOD)
        for bad in (GOOD.replace("Ran 1 test", "Ran 0 tests"),
                    GOOD.replace("\nOK\n", "\nOK (expected failures=1)\n")):
            with self.subTest(output=bad):
                run, _, _, _ = self.run_block(GUARD, output=bad)
                self.assertNotEqual(run.returncode, 0)

    def test_skipped_subtest_is_rejected_even_with_clean_summary(self):
        bad = GOOD.replace("\nRan 1 test", "  subtest (source='missing') ... skipped 'missing'\n\nRan 1 test")
        run, _, _, log = self.run_block(GUARD, output=bad)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(log, bad)

    def test_source_guard_removal_mutants_reopen_their_refusals(self):
        def mutate(source, needle, suffix=""):
            lines = source.splitlines(keepends=True)
            matched = [i for i, line in enumerate(lines) if needle in line]
            self.assertEqual(len(matched), 1)
            i = matched[0]
            lines[i] = lines[i].rstrip("\n") + suffix + "\n" if suffix else "true\n"
            return "".join(lines)

        setup = self.block(SETUP)
        for needle, failure, existing, suffix in (
            ("test ! -e", "", True, ""),
            ("fetch --no-tags", "fetch", False, " || true"),
            ("git -C \"$math_repo\" cat-file", "math-cat-file", False, ""),
            ("git cat-file -e", "main-cat-file", False, ""),
        ):
            with self.subTest(step=needle):
                baseline, _, _, _ = self.run_block(SETUP, failure=failure, existing=existing)
                mutant, _, exported, _ = self.run_block(
                    SETUP, failure=failure, existing=existing,
                    script_text=mutate(setup, needle, suffix))
                self.assertNotEqual(baseline.returncode, 0)
                self.assertEqual(mutant.returncode, 0, mutant.stdout + mutant.stderr)
                self.assertTrue(exported.startswith("GB_MATH_CHECKOUT="))

        guard = self.block(GUARD)
        skipped_subtest = GOOD.replace("\nRan 1 test", "  subtest (source='missing') ... skipped 'missing'\n\nRan 1 test")
        for needle, normal, output in (
            ("normal graph-binding composition did not pass", "", GOOD),
            ("optimized graph-binding test count invalid", NORMAL_LINE, GOOD.replace("Ran 1 test", "Ran 0 tests")),
            ("optimized graph-binding test did not finish cleanly", NORMAL_LINE, GOOD.replace("\nOK\n", "\nOK (expected failures=1)\n")),
            ("graph-binding test or subtest skipped", NORMAL_LINE, skipped_subtest),
        ):
            with self.subTest(step=needle):
                baseline, _, _, _ = self.run_block(GUARD, normal=normal, output=output)
                mutant, _, _, _ = self.run_block(
                    GUARD, normal=normal, output=output,
                    script_text=mutate(guard, needle))
                self.assertNotEqual(baseline.returncode, 0)
                self.assertEqual(mutant.returncode, 0, mutant.stdout + mutant.stderr)


if __name__ == "__main__":
    unittest.main()
