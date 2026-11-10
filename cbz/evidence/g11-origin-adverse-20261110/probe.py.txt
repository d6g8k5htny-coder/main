"""Probe the actual G11 selected unittest and parser against subject-origin failures."""
import hashlib
import json
import pathlib
import sys
import tempfile

root = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2]).resolve()
output.mkdir(parents=True, exist_ok=False)
paths = {
    "harness": root / "tools/cbz/g11_mutation_selftest.py",
    "subject": root / "tools/cbz/g11_census.py",
    "test": root / "tests/test_g11_census.py",
}
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
before = {name: digest(path) for name, path in paths.items()}
sys.path.insert(0, str(root / "tools/cbz"))
import g11_mutation_selftest as harness
test_source = paths["test"].read_text(encoding="utf-8")
tool_source = paths["subject"].read_text(encoding="utf-8")
harness_source = paths["harness"].read_text(encoding="utf-8")
selected = "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion"
fragment = "self.assertFalse(g11m.qualifies_kill(wrong_line, expected, 17))"
line = harness._assertion_line(test_source, selected, fragment)
needle = "def qualifies_kill(outcome, expected_test, expected_line):\n"
if harness_source.count(needle) != 1:
    raise ValueError('qualifies_kill definition mismatch')
old = '        and outcome.get("traceback_line") == expected_line\n        and outcome.get("exception_type")'
new = '        and True\n        and outcome.get("exception_type")'
if harness_source.count(old) != 1:
    raise ValueError('declared mutation target mismatch')
injections = {
    "subject_assert": "    if outcome.get('traceback_line') == 18:\n        raise AssertionError('PROBE_SUBJECT_ASSERT')\n",
    "helper_assert": "    if outcome.get('traceback_line') == 18:\n        def helper():\n            raise AssertionError('PROBE_HELPER_ASSERT')\n        return helper()\n",
    "callback_assert": "    if outcome.get('traceback_line') == 18:\n        def invoke(callback):\n            return callback()\n        def callback():\n            raise AssertionError('PROBE_CALLBACK_ASSERT')\n        return invoke(callback)\n",
    "runtime_error": "    if outcome.get('traceback_line') == 18:\n        raise RuntimeError('PROBE_RUNTIME_ERROR')\n",
}
cases = [("baseline", harness_source, ""), ("selected_oracle", harness_source.replace(old, new, 1), "declared mutation: replace exact line guard with True")]
cases += [(name, harness_source.replace(needle, needle + injection, 1), injection)
          for name, injection in injections.items()]
outcomes = []
with tempfile.TemporaryDirectory(prefix="cbz-g11-origin-probe-") as tmp:
    for name, variant, mutation in cases:
        work = pathlib.Path(tmp) / name
        harness._sandbox(tool_source, test_source, variant, str(work))
        observed = []
        real_run = harness.subprocess.run
        def capture(*args, **kwargs):
            proc = real_run(*args, **kwargs)
            observed.append(proc)
            return proc
        harness.subprocess.run = capture
        try:
            outcome = harness._run(str(work), selected)
        finally:
            harness.subprocess.run = real_run
        if len(observed) != 1:
            raise ValueError("actual child was not uniquely observed")
        proc = observed[0]
        accepted = harness.qualifies_kill(outcome, selected, line)
        case = {"name": name, "expected_test": selected, "expected_line": line,
                "outcome": outcome, "accepted_as_kill": accepted,
                "variant_sha256": hashlib.sha256(variant.encode("utf-8")).hexdigest(),
                "mutation": mutation, "command": proc.args}
        (output / (name + ".json")).write_text(json.dumps(case, indent=2, sort_keys=True) + "\n")
        (output / (name + ".stdout")).write_text(proc.stdout)
        (output / (name + ".stderr")).write_text(proc.stderr)
        print("CASE " + json.dumps(case, sort_keys=True), flush=True)
        print("ORIGINAL_STDERR " + name + " " + json.dumps(proc.stderr), flush=True)
        outcomes.append(case)
after = {name: digest(path) for name, path in paths.items()}
if before != after:
    raise ValueError("selected source changed during probe")
by_name = {case["name"]: case for case in outcomes}
if by_name["baseline"]["outcome"]["status"] != "OK" or by_name["baseline"]["accepted_as_kill"]:
    raise ValueError("clean selected baseline did not pass")
if not by_name["selected_oracle"]["accepted_as_kill"]:
    raise ValueError("genuine selected oracle failure was not credited")
if by_name["runtime_error"]["accepted_as_kill"]:
    raise ValueError("ordinary runtime error was wrongly credited")
for name in ("subject_assert", "helper_assert", "callback_assert"):
    if "PROBE_" not in (output / (name + ".stderr")).read_text():
        raise ValueError("subject-origin marker absent")
false_kills = [name for name in ("subject_assert", "helper_assert", "callback_assert")
               if by_name[name]["accepted_as_kill"]]
result = {"mode": "optimized" if sys.flags.optimize else "normal", "source_sha256": before,
          "source_after": after, "test": selected, "expected_line": line,
          "false_kills": false_kills, "status": "FAIL" if false_kills else "PASS"}
(output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("G11_ORIGIN_PROBE " + json.dumps(result, sort_keys=True), flush=True)
raise SystemExit(1 if false_kills else 0)
