#!/usr/bin/env python3
"""Run the unmodified black-box suite against isolated source-level mutants.

This is test tooling. Production code has no test flags or detector toggles.
Compilation failure, launch failure and timeout are harness errors, not kills.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/cbz_source_audit.py"
TESTS = ROOT / "tests/test_cbz_source_audit.py"
SUITE_ARGUMENTS = ["-m", "unittest", "discover", "-s", "tests",
                   "-p", "test_cbz_source_audit.py", "-v"]


def function(tree, name):
    found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1, (name, len(found))
    return found[0]


def exact_statement(tree, fragment, replacement):
    matches = 0
    for parent in ast.walk(tree):
        for attr, value in ast.iter_fields(parent):
            if not isinstance(value, list):
                continue
            for index, child in enumerate(value):
                if isinstance(child, ast.stmt) and ast.unparse(child).startswith(fragment):
                    value[index:index + 1] = ast.parse(replacement).body
                    matches += 1
    assert matches == 1, (fragment, matches)


def disable_token(tree, token):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REVIEW_TOKENS"
                                                for t in node.targets):
            pairs = list(zip(node.value.keys, node.value.values))
            matches = [index for index, (key, _) in enumerate(pairs) if key.value == token]
            assert len(matches) == 1, (token, matches)
            del pairs[matches[0]]
            node.value.keys = [key for key, _ in pairs]
            node.value.values = [value for _, value in pairs]
            return
    raise AssertionError("Missing token mapping")


def suppress_finding(tree, code):
    method = function(tree, "finding")
    method.body[:0] = ast.parse(f"if code == {code!r}:\n    return").body


def disable_native_form(tree, form):
    targets = [node for node in ast.walk(function(tree, "native_review_form"))
               if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant)
               and node.value.value == form]
    if len(targets) != 1:
        raise RuntimeError("Native review-form mutation anchor is missing or ambiguous: " + form)
    targets[0].value = ast.Constant(None)


def omit_missing_file(tree):
    target = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "path not in discovered"]
    assert len(target) == 1
    target[0].body = [ast.Continue()]


def disable_hash_check(tree):
    target = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "record['sha256'] != entry['sha256']"]
    assert len(target) == 1
    target[0].test = ast.Constant(False)


def fake_hash_identity(tree):
    exact_statement(tree, "record['sha256'] = hashlib.sha256(source).hexdigest()",
                    "record['sha256'] = entry['sha256']")


def omit_unlisted_files(tree):
    target = [node for node in ast.walk(tree) if isinstance(node, ast.For)
              and ast.unparse(node.target) == "path" and ast.unparse(node.iter) == "unlisted"]
    assert len(target) == 1
    target[0].body = [ast.Pass()]


def ignore_permissions(tree):
    target = [node for node in ast.walk(function(tree, "check_entry")) if isinstance(node, ast.If)
              and "mode & 292" in ast.unparse(node.test)]
    assert len(target) == 1, [ast.unparse(n.test) for n in ast.walk(function(tree, "check_entry"))
                              if isinstance(n, ast.If)]
    target[0].test = ast.Constant(False)


def ignore_symlink_classification(tree):
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Call):
            call = node.test.func
            if isinstance(call, ast.Attribute) and call.attr == "S_ISLNK":
                node.test = ast.Constant(False)
                count += 1
    assert count == 2, count


def ignore_utf8_errors(tree):
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "decode":
            for keyword in node.keywords:
                if keyword.arg == "errors":
                    keyword.value = ast.Constant("ignore")
                    count += 1
    assert count == 2


def ignore_comment_nesting(tree):
    target = [node for node in ast.walk(function(tree, "block_comment"))
              if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add)
              and isinstance(node.target, ast.Name) and node.target.id == "depth"]
    assert len(target) == 1
    target[0].value = ast.Constant(0)


def ignore_literal_mask(tree, name):
    function(tree, name).body = ast.parse("self.advance(self.i + 1)\nreturn").body


def ignore_raw_strings(tree):
    function(tree, "raw_string").body = [ast.Return(value=ast.Constant(False))]


def omit_semantic_gaps(tree):
    for node in ast.walk(function(tree, "empty_report")):
        if isinstance(node, ast.Dict):
            for index, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == "blocking_coverage":
                    node.values[index] = ast.List(elts=[], ctx=ast.Load())
                    return
    raise AssertionError("Missing semantic gap report")


def fake_unconditional_success(tree):
    function(tree, "main").body = ast.parse(
        "report = empty_report()\n"
        "report['aggregate_status'] = 'ESTABLISHED'\n"
        "report['gate_claims']['G1'] = 'ESTABLISHED'\n"
        "sys.stdout.write(render_report(report))\nreturn 0"
    ).body


def fake_unconditional_clear_audit(tree):
    function(tree, "audit_manifest").body = ast.parse(
        "report = empty_report()\nreport['input_integrity']['status'] = 'CHECKED'\nreturn report"
    ).body


def ignore_duplicate_json_keys(tree):
    target = [node for node in ast.walk(function(tree, "unique_object")) if isinstance(node, ast.If)]
    assert len(target) == 1
    target[0].test = ast.Constant(False)


def python_command(arguments):
    return [sys.executable, "-B", "-S", *(["-O"] * sys.flags.optimize), *arguments]


def child_environment(target):
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"PYTHONHOME", "PYTHONPATH", "PYTHONOPTIMIZE",
                                  "PYTHONSTARTUP", "PYTHONINSPECT"}}
    environment["CBZ_AUDITOR_SCRIPT"] = str(Path(target).resolve())
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def write_log(evidence_dir, name, output):
    path = Path(evidence_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")
    return {"log": str(path.resolve()),
            "log_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest()}


def partial_output(error):
    def decoded(value):
        return value.decode("utf-8", errors="backslashreplace") if isinstance(value, bytes) else value or ""
    return decoded(error.stdout) + decoded(error.stderr)


def test_target(target, log_name, evidence_dir, *, suite_root=ROOT, timeout=60):
    command = python_command(SUITE_ARGUMENTS)
    result = {"returncode": None, "tests_run": 0, "assertion_failures": 0,
              "test_errors": 0, "tests_skipped": 0, "failing_tests": [],
              "suite_argv": command, "harness_error": None}
    try:
        completed = subprocess.run(command, cwd=suite_root, env=child_environment(target),
                                   text=True, encoding="utf-8", capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        result["harness_error"] = {"stage": "suite", "kind": "TIMEOUT"}
        result.update(write_log(evidence_dir, log_name, partial_output(error)))
        return result
    except (OSError, UnicodeError) as error:
        result["harness_error"] = {"stage": "suite", "kind": type(error).__name__}
        result.update(write_log(evidence_dir, log_name, str(error) + "\n"))
        return result
    output = completed.stdout + completed.stderr
    tests = re.search(r"^Ran (\d+) tests? in .*$", output, flags=re.MULTILINE)
    summary = re.search(r"^(?:FAILED \([^\n]*\)|OK(?: \([^\n]*\))?)$", output, flags=re.MULTILINE)
    counts = dict(re.findall(r"(failures|errors|skipped)=(\d+)", summary.group(0))) if summary else {}
    result.update({
        "returncode": completed.returncode,
        "tests_run": int(tests.group(1)) if tests else 0,
        "assertion_failures": int(counts.get("failures", 0)),
        "test_errors": int(counts.get("errors", 0)),
        "tests_skipped": int(counts.get("skipped", 0)),
        "failing_tests": sorted(set(re.findall(r"^FAIL: (\S+)", output, flags=re.MULTILINE))),
        **write_log(evidence_dir, log_name, output),
    })
    if tests is None or summary is None:
        result["harness_error"] = {"stage": "suite", "kind": "INCOMPLETE_RESULT"}
    return result


def reject_json_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def smoke_target(target, evidence_dir, *, timeout=5):
    """Check launch health only; semantic wrongness remains the suite's job."""
    Path(evidence_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke-", dir=evidence_dir) as temporary:
        base = Path(temporary)
        root = base / "source"
        root.mkdir()
        source = b"theorem smoke_ok : True := by trivial\n"
        (root / "Main.lean").write_bytes(source)
        manifest = base / "manifest.json"
        manifest.write_text(json.dumps({
            "version": 1, "root": str(root),
            "files": [{"path": "Main.lean", "sha256": hashlib.sha256(source).hexdigest()}],
        }), encoding="utf-8")
        command = python_command([str(Path(target).resolve()), "--manifest", str(manifest)])
        try:
            completed = subprocess.run(command, env=child_environment(target), text=True,
                                       encoding="utf-8", capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            return {"status": "TIMEOUT", "stdout_and_stderr": partial_output(error)}
        except (OSError, UnicodeError) as error:
            return {"status": "LAUNCH_ERROR", "detail": str(error)}
    result = {"returncode": completed.returncode, "stdout": completed.stdout,
              "stderr": completed.stderr}
    # Zero is allowed here so a functioning false-success mutant is still testable.
    if completed.stderr or completed.returncode not in (0, 1, 2):
        return {"status": "LAUNCH_ERROR", **result}
    try:
        report = json.loads(completed.stdout, parse_constant=reject_json_constant)
    except (ValueError, RecursionError):
        return {"status": "INVALID_JSON", **result}
    if not isinstance(report, dict):
        return {"status": "INVALID_JSON_OBJECT", **result}
    return {"status": "VALID_JSON_REPORT", **result}


def evaluate_target(target, baseline_count, log_name, evidence_dir, *, suite_root=ROOT,
                    smoke_timeout=5, suite_timeout=60):
    """Separate invalid programs/executions from complete behavioral assertion kills."""
    result = {"returncode": None, "tests_run": 0, "assertion_failures": 0,
              "test_errors": 0, "tests_skipped": 0, "failing_tests": [],
              "harness_error": None, "smoke": None, "killed": False}
    try:
        compile(Path(target).read_bytes(), str(target), "exec")
    except (OSError, SyntaxError, ValueError) as error:
        result["harness_error"] = {"stage": "compile", "kind": type(error).__name__, "detail": str(error)}
        result["result"] = "HARNESS_ERROR"
        result.update(write_log(evidence_dir, log_name, str(error) + "\n"))
        return result
    result["smoke"] = smoke_target(target, evidence_dir, timeout=smoke_timeout)
    result["smoke_log"] = write_log(evidence_dir, log_name + ".smoke.json",
                                     json.dumps(result["smoke"], sort_keys=True, indent=2) + "\n")
    if result["smoke"]["status"] != "VALID_JSON_REPORT":
        result["harness_error"] = {"stage": "smoke", "kind": result["smoke"]["status"]}
        result["result"] = "HARNESS_ERROR"
        result.update(write_log(evidence_dir, log_name, "Suite not run: invalid smoke execution.\n"))
        return result
    result.update(test_target(target, log_name, evidence_dir, suite_root=suite_root, timeout=suite_timeout))
    killed = (result["harness_error"] is None and result["returncode"] == 1
              and result["assertion_failures"] > 0 and result["test_errors"] == 0
              and result["tests_skipped"] == 0 and result["tests_run"] > 0
              and result["tests_run"] == baseline_count)
    result["killed"] = killed
    result["result"] = ("HARNESS_ERROR" if result["harness_error"] else
                        "KILLED" if killed else "NOT_KILLED")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path,
                        help="New directory for this cut's receipts, logs and isolated mutants.")
    args = parser.parse_args(argv)
    evidence_dir = args.evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=False)
    mutant_dir = evidence_dir / "mutants"
    mutant_dir.mkdir()
    original = SOURCE.read_bytes()
    original_tests = TESTS.read_bytes()
    original_runner = Path(__file__).read_bytes()
    source_hash = hashlib.sha256(original).hexdigest()
    parsed = ast.parse(original.decode("utf-8"))
    baseline = evaluate_target(SOURCE, None, "mutation-baseline.log", evidence_dir)
    if (baseline["harness_error"] or baseline["returncode"] != 0 or baseline["tests_run"] == 0
            or baseline["assertion_failures"] or baseline["test_errors"] or baseline["tests_skipped"]):
        raise RuntimeError("Unmodified auditor suite/smoke did not pass: " + repr(baseline))

    mutants = []
    for token in ("sorry", "admit", "axiom", "native_decide", "unsafe", "Float", "Float32",
                  "sorryAx", "ofReduceBool", "ofReduceNat", "trustCompiler", "skipKernelTC", "addDeclWithoutChecking",
                  "run_tac", "run_elab", "eval_expr", "implemented_by", "extern", "macro",
                  "macro_rules", "syntax", "elab", "elab_rules", "declare_syntax_cat",
                  "initialize", "builtin_initialize"):
        mutants.append(("disable_token_" + token, lambda tree, token=token: disable_token(tree, token)))
    for code in ("QUOTED_IDENTIFIER", "STRING_INTERPOLATION", "UNSUPPORTED_SYNTAX", "UNSUPPORTED_ESCAPE"):
        mutants.append(("suppress_" + code.lower(), lambda tree, code=code: suppress_finding(tree, code)))
    for name, form in (("disable_plus_native_form", "+native"), ("disable_native_assignment_form", "native:=")):
        mutants.append((name, lambda tree, form=form: disable_native_form(tree, form)))
    mutants.extend([
        ("skip_missing_file", omit_missing_file),
        ("disable_hash_check", disable_hash_check),
        ("fake_hash_identity_from_manifest", fake_hash_identity),
        ("skip_unlisted_files", omit_unlisted_files),
        ("ignore_permission_denial", ignore_permissions),
        ("disable_symlink_classification", ignore_symlink_classification),
        ("ignore_invalid_utf8", ignore_utf8_errors),
        ("ignore_nested_comments", ignore_comment_nesting),
        ("disable_string_mask", lambda tree: ignore_literal_mask(tree, "string")),
        ("disable_character_mask", lambda tree: ignore_literal_mask(tree, "character")),
        ("disable_raw_string_mask", ignore_raw_strings),
        ("omit_unestablished_semantic_coverage", omit_semantic_gaps),
        ("fake_unconditional_success", fake_unconditional_success),
        ("fake_unconditional_clear_audit", fake_unconditional_clear_audit),
        ("ignore_duplicate_json_keys", ignore_duplicate_json_keys),
    ])
    results = []
    for name, mutate in mutants:
        tree = copy.deepcopy(parsed)
        mutate(tree)
        ast.fix_missing_locations(tree)
        changed = (ast.unparse(tree) + "\n").encode("utf-8")
        assert changed != original
        target = mutant_dir / (name + ".py")
        target.write_bytes(changed)
        result = evaluate_target(target, baseline["tests_run"], "mutation-" + name + ".log", evidence_dir)
        results.append({"name": name, "source_sha256": hashlib.sha256(changed).hexdigest(), **result})
        print(f"{name}: {result['result']}", flush=True)
    if (SOURCE.read_bytes() != original or TESTS.read_bytes() != original_tests
            or Path(__file__).read_bytes() != original_runner):
        raise RuntimeError("Source, tests or runner changed during the mutation cut.")
    receipt = {
        "schema_version": 2, "source": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": source_hash, "tests_sha256": hashlib.sha256(original_tests).hexdigest(),
        "runner_sha256": hashlib.sha256(original_runner).hexdigest(),
        "command": shlex.join(["python", "tests/run_scanner_mutations.py", "--evidence-dir", str(evidence_dir)]),
        "suite_command": shlex.join(["python", *python_command(SUITE_ARGUMENTS)[1:]]),
        "suite_argv": python_command(SUITE_ARGUMENTS),
        "evidence_dir": str(evidence_dir),
        "negative_control": {**baseline, "result": "UNMODIFIED_SUITE_PASSES"},
        "mutants_tested": len(results), "mutants_killed": sum(r["result"] == "KILLED" for r in results),
        "harness_errors": sum(r["result"] == "HARNESS_ERROR" for r in results),
        "mutants": results,
        "interpretation": "Finite regression evidence only; no semantic or complete scanner-correctness proof.",
    }
    (evidence_dir / "mutation-receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n",
                                                      encoding="utf-8")
    return 0 if receipt["mutants_killed"] == receipt["mutants_tested"] else 1


if __name__ == "__main__":
    sys.exit(main())
