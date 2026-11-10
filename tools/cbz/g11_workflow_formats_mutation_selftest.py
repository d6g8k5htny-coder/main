#!/usr/bin/env python3
"""Bounded mutations of G11 env-wrapper and JSON workflow reference handling."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import g11_mutation_selftest as g11

ROOT = Path(__file__).resolve().parents[2]
FORMAT_TEST = "tests/test_g11_workflow_formats.py"
PREFIX = "tests.test_g11_workflow_formats.WorkflowFormats."
CASES = (
    ("env wrapper not recognized",
     '            if tokens and tokens[0] == "env":',
     '            if False:',
     PREFIX + "test_env_unset_preserves_direct_execution_reference",
     "self.assertEqual(g11.workflow_tool_references(text, TOOLS), {TOOL})"),
    ("env assignments implicitly credited",
     '        if token.startswith("-") or "=" in token:\n            return []',
     '        if token.startswith("-"):\n            return []\n        if "=" in token:\n            index += 1\n            continue',
     PREFIX + "test_env_unsupported_or_nonexecution_is_not_credited",
     "self.assertEqual(g11.workflow_tool_references(yaml_run(command), TOOLS), set())"),
    ("JSON job runs ignored",
     '    if text.lstrip().startswith(("{", "[")):',
     '    if False:',
     PREFIX + "test_json_workflow_recognizes_only_job_step_runs",
     "self.assertEqual(g11.workflow_tool_references(text, TOOLS), {PUBLISHER})"),
    ("nested JSON with run credited",
     '                if isinstance(step, dict) and isinstance(step.get("run"), str):\n                    commands =',
     '                if isinstance(step, dict) and isinstance(step.get("run", step.get("with", {}).get("run")), str):\n                    step = dict(step, run=step.get("run", step.get("with", {}).get("run")))\n                    commands =',
     PREFIX + "test_json_workflow_recognizes_only_job_step_runs",
     "self.assertEqual(g11.workflow_tool_references(text, TOOLS), {PUBLISHER})"),
    ("duplicate JSON keys accepted",
     '        if key in result:\n            raise ValueError("duplicate JSON workflow key")',
     '        if False:\n            raise ValueError("duplicate JSON workflow key")',
     PREFIX + "test_json_duplicate_keys_and_non_run_paths_are_refused",
     "self.assertEqual(g11.workflow_tool_references(duplicate, TOOLS), set())"),
    ("JSON no-op scan ignored",
     '    for number, commands in _workflow_reference_commands(text):\n        if not commands:',
     '    for number, commands in _workflow_commands(text):\n        if not commands:',
     PREFIX + "test_json_noop_step_is_failing",
     'self.assertEqual(report["status"], "FAIL")'),
    ("invalid JSON workflow ignored",
     '                    findings.append({"path": path, "area": kind, "class": WORKFLOW_JSON_INVALID,\n                                     "line": 0, "detail": "invalid or unsupported JSON workflow"})',
     '                    pass',
     PREFIX + "test_duplicate_json_workflow_is_unparseable",
     'self.assertEqual(report["status"], "FAIL")'),
    ("non-string JSON run accepted",
     '                if "run" in step and not isinstance(step["run"], str):\n                    raise ValueError("JSON workflow run must be a string")',
     '                if False:\n                    raise ValueError("JSON workflow run must be a string")',
     PREFIX + "test_nonstring_json_run_is_unparseable",
     'self.assertEqual(report["status"], "FAIL")'),
)

def run(root):
    paths = [root / g11.TOOL, root / g11.TESTS, root / g11.HARNESS, root / FORMAT_TEST]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    tool_source, test_source, harness_source, format_source = [
        path.read_text(encoding="utf-8") for path in paths]
    if len({case[0] for case in CASES}) != len(CASES):
        raise ValueError("duplicate mutation name")
    results = []
    with tempfile.TemporaryDirectory(prefix="g11-formats-mutants-") as tmp:
        for index, (name, old, new, test_id, fragment) in enumerate(CASES):
            if old == new or tool_source.count(old) != 1:
                raise ValueError("mutation target missing or ambiguous: " + name)
            mutated = tool_source.replace(old, new, 1)
            line = g11._assertion_line(format_source, test_id, fragment)
            outcomes = []
            for label, source in (("baseline", tool_source), ("mutant", mutated)):
                work = Path(tmp) / (str(index) + "-" + label)
                g11._sandbox(source, test_source, harness_source, str(work))
                (work / FORMAT_TEST).write_text(format_source, encoding="utf-8")
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
    return {"mode": "optimized" if sys.flags.optimize else "normal",
            "inputs_sha256": before, "results": results, "executed": len(results),
            "total": len(CASES), "status": "PASS" if len(results) == len(CASES)
            and all(result["killed"] for result in results) else "FAIL"}

def main():
    try:
        report = run(ROOT)
    except (OSError, ValueError, g11.HarnessError) as exc:
        print("G11_FORMATS_MUTATION_HARNESS_ERROR: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
