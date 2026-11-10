#!/usr/bin/env python3
"""Exercise bounded mutations of G11's live assertion-origin guard.

Each case runs an unmodified designated origin test and an isolated mutation of
the real grader. The normal G11 parent runner must observe an authorized
assertion at the designated test line. This does not certify arbitrary hostile
Python code or replace the full G11 census/mutation gate.
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import g11_mutation_selftest as g11

ROOT = Path(__file__).resolve().parents[2]
ORIGIN_TEST = "tests/test_g11_origin_provenance.py"
PREFIX = "tests.test_g11_origin_provenance.ActualG11Origin."
CASES = (
    ("missing live-origin guard",
     (("        and outcome.get(\"origin_authorized\") is True\n", "        and True\n"),),
     PREFIX + "test_formatted_traceback_without_live_origin_is_not_a_kill",
     "self.assertFalse(g11m.qualifies_kill(parsed, SELECTED, 123))"),
    ("subject and helper suffix accepted",
     (("len(positions) == 1 and all(is_oracle(code) for code in codes[positions[0] + 1:]))",
       "len(positions) == 1 and True)"),
      ("and all(f[\"oracle\"] is True for f in frames[positions[0] + 1:])", "and True")),
     PREFIX + "test_direct_subject_assertion_is_not_a_kill",
     "self.assertFalse(g11m.qualifies_kill(outcome, SELECTED, self.line))"),
    ("cleanup phase accepted",
     (("state.get(\"phase\") in (\"body\", \"body-exited\")", "True"),),
     PREFIX + "test_lifecycle_and_spoofed_nested_code_are_not_authorized",
     'self.assertFalse(outcome.get("origin_authorized"), outcome)'),
    ("untrusted suffix report accepted",
     (("and all(f[\"oracle\"] is True for f in frames[positions[0] + 1:])", "and True"),),
     PREFIX + "test_origin_report_rejects_inconsistent_and_non_oracle_frames",
     "self.assertFalse(g11m.qualifies_kill(outcome, target, 7))"),
    ("missing runner completion accepted",
     (("if not lines or lines[-1] != marker or lines.count(marker) != 1:", "if False:"),),
     PREFIX + "test_precreated_report_without_runner_completion_is_rejected",
     "self.assertFalse(g11m.qualifies_kill(outcome, target, 123), outcome)"),
)


def run(root):
    paths = [root / g11.TOOL, root / g11.TESTS, root / g11.HARNESS, root / ORIGIN_TEST]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    tool_source, test_source, harness_source, origin_source = [
        path.read_text(encoding="utf-8") for path in paths]
    if len({case[0] for case in CASES}) != len(CASES):
        raise ValueError("duplicate mutation name")
    results = []
    with tempfile.TemporaryDirectory(prefix="g11-origin-mutants-") as tmp:
        for index, (name, edits, test_id, fragment) in enumerate(CASES):
            mutated = harness_source
            for old, new in edits:
                if old == new or mutated.count(old) != 1:
                    raise ValueError("mutation target missing or ambiguous: " + name)
                mutated = mutated.replace(old, new, 1)
            line = g11._assertion_line(origin_source, test_id, fragment)
            outcomes = []
            for label, selected_source in (("baseline", harness_source), ("mutant", mutated)):
                work = Path(tmp) / (str(index) + "-" + label)
                g11._sandbox(tool_source, test_source, selected_source, str(work))
                (work / ORIGIN_TEST).write_text(origin_source, encoding="utf-8")
                outcomes.append(g11._run(str(work), test_id))
            baseline, mutant = outcomes
            clean = (baseline.get("parsed") is True and baseline.get("exit") == 0
                     and baseline.get("status") == "OK" and baseline.get("ran") == 1
                     and baseline.get("failures") == 0 and baseline.get("errors") == 0
                     and baseline.get("skipped") == 0)
            killed = clean and g11.qualifies_kill(mutant, test_id, line)
            results.append({"name": name, "test": test_id, "line": line,
                            "baseline": baseline, "mutant": mutant, "killed": killed})
    after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    if before != after:
        raise ValueError("source changed during mutation self-test")
    return {"mode": "optimized" if sys.flags.optimize else "normal", "inputs_sha256": before,
            "results": results, "executed": len(results), "total": len(CASES),
            "status": "PASS" if len(results) == len(CASES) and all(r["killed"] for r in results) else "FAIL"}


def main():
    try:
        report = run(ROOT)
    except (OSError, ValueError, g11.HarnessError) as exc:
        print("G11_ORIGIN_MUTATION_HARNESS_ERROR: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
