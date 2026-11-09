"""Adverse controls for the navigation workflow's history-backed tests.

The shell fixture supplies synthetic process results; it does not represent
GitHub execution or an actual historical-source check.
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
MARKER = "      - name: Verify history-backed reader controls\n        run: |\n"
CLASSES = (
    "tests.test_reader_links.FiniteH0LifetimeReading",
    "tests.test_reader_links.UniversalitySourcesReading",
)
REQUIRED = (
    ("FiniteH0LifetimeReading", "test_pinned_items_carry_only_their_digest_and_the_digests_match_the_pinned_bytes"),
    ("FiniteH0LifetimeReading", "test_quoted_phrases_are_verbatim_in_their_pinned_sources"),
    ("UniversalitySourcesReading", "test_pinned_items_carry_only_their_digest_and_the_digests_match_the_pinned_bytes"),
    ("UniversalitySourcesReading", "test_process_marks_and_open_gaps_follow_the_records"),
    ("UniversalitySourcesReading", "test_quoted_phrases_are_verbatim_in_their_pinned_sources"),
)
GOOD = "".join(
    f"{method} (tests.test_reader_links.{cls}.{method}) ... ok\n"
    for cls, method in REQUIRED
) + "\nRan 20 tests in 0.001s\n\nOK\n"
DRIVER = '''import json, os, sys
from pathlib import Path
with Path(os.environ["FIXTURE_TRACE"]).open("a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
data = json.loads(Path(os.environ["FIXTURE_RESULT"]).read_text())
sys.stdout.write(data["output"])
raise SystemExit(data["exit"])
'''


class NavigationHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.bash = shutil.which("bash")
        if not cls.bash:
            raise RuntimeError("Bash is required for navigation workflow controls")

    def block(self):
        self.assertEqual(self.workflow.count(MARKER), 1, "history guard is missing or ambiguous")
        block = self.workflow.split(MARKER, 1)[1].split("\n      - ", 1)[0]
        lines = block.splitlines()
        self.assertTrue(lines)
        self.assertTrue(all(not line or line.startswith("          ") for line in lines))
        return "\n".join(line[10:] if line else "" for line in lines) + "\n"

    def run_block(self, output=GOOD, exit_status=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "navigation-evidence").mkdir()
            binary = root / "bin"
            binary.mkdir()
            driver = root / "fixture.py"
            driver.write_text(DRIVER, encoding="utf-8")
            shim = binary / "python"
            shim.write_text('#!/bin/sh\nexec "$FIXTURE_PYTHON" -B -S "$FIXTURE_DRIVER" "$@"\n', encoding="utf-8")
            shim.chmod(0o700)
            result = root / "result.json"
            result.write_text(json.dumps({"output": output, "exit": exit_status}), encoding="utf-8")
            script = root / "run.sh"
            script.write_text(self.block(), encoding="utf-8")
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("PYTHON") and key not in ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS")}
            env.update(PATH=str(binary) + os.pathsep + os.defpath,
                       FIXTURE_PYTHON=sys.executable, FIXTURE_DRIVER=str(driver),
                       FIXTURE_RESULT=str(result), FIXTURE_TRACE=str(root / "trace.jsonl"))
            run = subprocess.run([self.bash, "--noprofile", "--norc", "-e", str(script)],
                                 cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 capture_output=True, text=True, timeout=15)
            trace = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
            log = (root / "navigation-evidence/history-reader.log").read_text(encoding="utf-8")
            return run, trace, log

    def test_checkout_requests_full_history(self):
        self.assertIn("      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1\n"
                      "        with:\n          fetch-depth: 0\n", self.workflow)

    def test_complete_history_suite_is_accepted_and_retained(self):
        run, trace, log = self.run_block()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(trace, [["-B", "-S", "-m", "unittest", "-v", *CLASSES]])
        self.assertEqual(log, GOOD)

    def test_missing_required_test_is_rejected(self):
        first = GOOD.splitlines(keepends=True)[0]
        run, _, log = self.run_block(output=GOOD.replace(first, "", 1))
        self.assertNotEqual(run.returncode, 0)
        self.assertNotIn(first, log)

    def test_selected_test_skip_is_rejected(self):
        bad = GOOD.replace(" ... ok\n", " ... skipped 'missing history'\n", 1)
        run, _, log = self.run_block(output=bad)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(log, bad)

    def test_subtest_skip_is_rejected_even_with_ok_summary(self):
        bad = GOOD.replace("\nRan 20", "  test_pinned (tests.test_reader_links.UniversalitySourcesReading.test_pinned) (url='x') ... skipped 'missing'\n\nRan 20")
        run, _, log = self.run_block(output=bad)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(log, bad)

    def test_empty_or_malformed_summary_is_rejected(self):
        for replacement in ("Ran 0 tests in 0.001s", "no census"):
            with self.subTest(replacement=replacement):
                run, _, _ = self.run_block(output=GOOD.replace("Ran 20 tests in 0.001s", replacement))
                self.assertNotEqual(run.returncode, 0)

    def test_nonzero_child_exit_is_not_masked(self):
        run, _, log = self.run_block(exit_status=23)
        self.assertEqual(run.returncode, 23)
        self.assertEqual(log, GOOD)

    def test_skipped_summary_is_not_accepted(self):
        run, _, _ = self.run_block(output=GOOD.replace("\nOK\n", "\nOK (skipped=1)\n"))
        self.assertNotEqual(run.returncode, 0)

    def test_earlier_ok_cannot_mask_nonclean_final_summary(self):
        forged = GOOD.replace("\nRan 20 tests in 0.001s\n\nOK\n",
                              "\nOK\nRan 20 tests in 0.001s\n\nOK (expected failures=1)\n")
        run, _, _ = self.run_block(output=forged)
        self.assertNotEqual(run.returncode, 0)

    def test_earlier_positive_count_cannot_mask_empty_final_census(self):
        forged = GOOD.replace("Ran 20 tests in 0.001s", "Ran 20 tests in 0.001s\n\nRan 0 tests in 0.001s")
        run, _, _ = self.run_block(output=forged)
        self.assertNotEqual(run.returncode, 0)


if __name__ == "__main__":
    unittest.main()
