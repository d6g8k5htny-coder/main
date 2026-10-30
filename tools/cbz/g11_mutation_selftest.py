#!/usr/bin/env python3
"""Mutation self-test of the G11 census (tools/cbz/g11_census.py).

Each mutation weakens the census or its attribution guard in one way. For
every mutation the harness copies the census, test module and harness into an
isolated directory, applies exactly one mutation, and runs one designated test.
A mutation is KILLED only when the clean full baseline passes and that one test
reports exactly one AssertionError at its exact designated assertion line,
with the exact module/class/method identity, FAILED status, zero errors and
zero skips. Setup, import, teardown, cleanup, process and unrelated assertion
failures never count as kills. The self-test passes only when the unmutated
baseline passes and every mutation is killed.

A mutation whose target text does not occur exactly once is a harness error,
never a silent no-op.

Usage: python3 -B -S tools/cbz/g11_mutation_selftest.py [--root DIR] [--json]
Exit status: 0 pass, 1 a mutation survived or the baseline failed, 2 harness error.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = os.path.join("tools", "cbz", "g11_census.py")
TESTS = os.path.join("tests", "test_g11_census.py")
HARNESS = os.path.join("tools", "cbz", "g11_mutation_selftest.py")

MUTATIONS = [
    ("pass not detected", '    if isinstance(stmt, ast.Pass):\n        return "pass"\n',
     '    if False:\n        return "pass"\n'),
    ("ellipsis not detected", "and stmt.value.value is Ellipsis:", "and stmt.value.value is None:"),
    ("NotImplementedError not detected", 'exc.id == "NotImplementedError"', 'exc.id == "NotImplemented"'),
    ("raise call not unwrapped", "exc = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc",
     "exc = stmt.exc"),
    ("docstring-only not reported", '        if not rest:\n            return "docstring only"\n',
     "        if not rest:\n            return None\n"),
    ("docstring not stripped", "        rest = rest[1:]\n", "        rest = rest[0:]\n"),
    ("decorator context ignored", "        if _trusted_decorator(dec, imports, scope_shadowed):", "        if False:"),
    ("Protocol/ABC context ignored", "            if _trusted_abstract_base(base, imports):", "            if False:"),
    ("exception context ignored", "            if _builtin_exception_base(base, imports):", "            if False:"),
    ("log_message context ignored", '            if _http_request_handler_base(base, imports) and func.name == "log_message":', "            if False:"),
    ("log_message context too broad", '            if _http_request_handler_base(base, imports) and func.name == "log_message":',
     '            if _http_request_handler_base(base, imports):'),
    ("module docstring not recorded", "                docstring = bool(ast.get_docstring(ast.parse(files[path])))",
     "                docstring = True"),
    ("class bodies not visited", "                visit(child, classes + [child])", "                pass"),
    ("nested functions not visited", "                visit(child, [])\n            else:",
     "                pass\n            else:"),
    ("comments read as data", "        if tok.type == tokenize.COMMENT:\n            target = markers",
     "        if tok.type == tokenize.COMMENT:\n            target = mentions"),
    ("strings read as comments", "            target = mentions\n        else:\n            continue",
     "            target = markers\n        else:\n            continue"),
    ("marker words case-insensitive", r'MARKER = re.compile(r"\b(?:TODO|FIXME|XXX|TBD|STUB)\b|\b(?i:not implemented)\b")',
     r'MARKER = re.compile(r"(?i)\b(?:TODO|FIXME|XXX|TBD|STUB)\b|\bnot implemented\b")'),
    ("marker word boundary dropped", r'MARKER = re.compile(r"\b(?:TODO|FIXME|XXX|TBD|STUB)\b|\b(?i:not implemented)\b")',
     r'MARKER = re.compile(r"(?:TODO|FIXME|XXX|TBD|STUB)|\b(?i:not implemented)\b")'),
    ("quotes not tracked", "        elif ch in \"\\\"'`\":\n            quote = ch",
     "        elif False:\n            quote = ch"),
    ("string escapes ignored", '            if ch == "\\\\":\n                i += 2\n',
     "            if False:\n                i += 2\n"),
    ("hash comment needs no whitespace", '(opener != "#" or i == 0 or line[i - 1].isspace())', "True"),
    ("block comments not entered", "                in_block, rest = True, rest[first + 2:]",
     "                in_block, rest = False, rest[first + 2:]"),
    ("block comment end ignored", "                        in_block, rest = False, rest[end + 2:]",
     "                        in_block, rest = True, rest[end + 2:]"),
    ("mentions dropped", "for m in mentions)", "for m in [])"),
    ("noop detection off", "        if commands and all(c in NOOP_RUN for c in commands):", "        if False:"),
    ("block scalar not read", '        if not value or value[0] in "|>":', "        if False:"),
    ("block scalar ends late", "                if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= key_column:",
     "                if nxt.strip() and len(nxt) - len(nxt.lstrip()) < key_column:"),
    ("inline comment kept", '            commands = [re.sub(r"\\s+#.*$", "", value).strip("\'\\"")]',
     '            commands = [value.strip("\'\\"")]'),
    ("tests counted as tools", '                  and area(p) == "tooling" and not p.endswith("__init__.py"))',
     '                  and True and not p.endswith("__init__.py"))'),
    ("__init__ counted as tool", '                  and area(p) == "tooling" and not p.endswith("__init__.py"))',
     '                  and area(p) == "tooling" and True)'),
    ("basename ambiguity ignored", '    unique = {p: basenames[p.rsplit("/", 1)[-1]] == 1 for p in tools}',
     "    unique = {p: True for p in tools}"),
    ("path reference not recognised", '    pattern = r"\\b" + re.escape(path) + r"\\b"\n',
     '    pattern = r"\\bNOMATCH\\b"\n'),
    ("import not recognised", "        if isinstance(node, ast.Import):", "        if False:"),
    ("from-import list not recognised", "        elif isinstance(node, ast.ImportFrom) and node.level == 0:", "        elif False:"),
    ("basename not recognised", '                    r"|\\b" + re.escape(path.rsplit("/", 1)[-1]) + r"\\b")',
     '                    r"|\\bNOMATCH")'),
    ("transitive exercise off", '                if other != path and report[other]["exercised"] and other in texts and path in tool_dependencies.get(other, set()):',
     '                if False and other != path and report[other]["exercised"] and other in texts and path in tool_dependencies.get(other, set()):'),
    ("site not production", 'PRODUCTION = {"tooling", "site"}', 'PRODUCTION = {"tooling"}'),
    ("research counted as production", 'PRODUCTION = {"tooling", "site"}', 'PRODUCTION = {"tooling", "site", "research"}'),
    ("vendor counted as site", '    if path.startswith("docs/site/vendor/"):\n        return "vendor"',
     '    if False:\n        return "vendor"'),
    ("undecodable decoded leniently", '            text = raw.decode("utf-8")', '            text = raw.decode("utf-8", "replace")'),
    ("unparseable not a failure class", '"class": "unparseable"', '"class": "mention"'),
    ("legitimate context still fails", '"class": "stub" if not stub["context"] else "stub-context"', '"class": "stub"'),
    ("markers do not fail", 'f["class"] in {"stub", "marker", "noop", "undecodable", "unparseable"}]',
     'f["class"] in {"stub", "noop", "undecodable", "unparseable"}]'),
    ("noop steps do not fail", 'f["class"] in {"stub", "marker", "noop", "undecodable", "unparseable"}]',
     'f["class"] in {"stub", "marker", "undecodable", "unparseable"}]'),
    ("tool findings do not fail", '"status": "FAIL" if failing or failing_tools else "PASS"}',
     '"status": "FAIL" if failing else "PASS"}'),
    ("exit status always zero", '    return 1 if report["status"] == "FAIL" else 0\n', "    return 0\n"),
    ("unreadable revision exits zero",
     '        print("G11_CENSUS_ERROR: cannot read revision %s: %s" % (args.rev, exc), file=sys.stderr)\n        return 2\n',
     '        print("G11_CENSUS_ERROR: cannot read revision %s: %s" % (args.rev, exc), file=sys.stderr)\n        return 0\n'),
]

RESULT = re.compile(r"^(FAILED|OK)\b(?: \(([^)]*)\))?\s*$", re.M)
SUMMARY = re.compile(r"Ran (\d+) tests? in [^\n]+")
FAIL_HEADER = re.compile(r"^FAIL: ([^(]+) \(([^)]+)\)$", re.M)
TRACE_FRAME = re.compile(r'File "([^"]*test_g11_census\.py)", line (\d+), in ([A-Za-z_]\w*)')
EXCEPTION = re.compile(r"(?m)^(AssertionError):")


class HarnessError(Exception):
    pass


TARGETS = {
    "pass not detected": ("tests.test_g11_census.PythonStubs.test_each_stub_body", r'stubs("def a():\n    pass\n")'),
    "ellipsis not detected": ("tests.test_g11_census.PythonStubs.test_each_stub_body", r'stubs("def a():\n    ...\n")'),
    "NotImplementedError not detected": ("tests.test_g11_census.PythonStubs.test_each_stub_body", r"raise NotImplementedError\n"),
    "raise call not unwrapped": ("tests.test_g11_census.PythonStubs.test_each_stub_body", "NotImplementedError('later')"),
    "docstring-only not reported": ("tests.test_g11_census.PythonStubs.test_each_stub_body", "Only a docstring."),
    "docstring not stripped": ("tests.test_g11_census.PythonStubs.test_docstring_then_stub_body", "Doc."),
    "decorator context ignored": ("tests.test_g11_census.PythonStubs.test_legitimate_contexts_are_reported_but_do_not_fail", "self.assertEqual(contexts,"),
    "Protocol/ABC context ignored": ("tests.test_g11_census.PythonStubs.test_legitimate_contexts_are_reported_but_do_not_fail", "self.assertEqual(contexts,"),
    "exception context ignored": ("tests.test_g11_census.PythonStubs.test_legitimate_contexts_are_reported_but_do_not_fail", "self.assertEqual(contexts,"),
    "log_message context ignored": ("tests.test_g11_census.PythonStubs.test_request_handler_log_message_override_is_legitimate", "self.assertEqual(found,"),
    "log_message context too broad": ("tests.test_g11_census.PythonStubs.test_request_handler_log_message_override_is_legitimate", "self.assertEqual(found,"),
    "module docstring not recorded": ("tests.test_g11_census.ToolChecks.test_module_docstring_is_recorded", "self.assertEqual((report["),
    "class bodies not visited": ("tests.test_g11_census.PythonStubs.test_methods_async_and_nested", "self.assertEqual(stubs(text),"),
    "nested functions not visited": ("tests.test_g11_census.PythonStubs.test_methods_async_and_nested", "self.assertEqual(stubs(text),"),
    "comments read as data": ("tests.test_g11_census.Markers.test_python_comment_is_marker_and_string_is_mention", "self.assertEqual(classes(run("),
    "strings read as comments": ("tests.test_g11_census.Markers.test_python_comment_is_marker_and_string_is_mention", "self.assertEqual(classes(run("),
    "marker words case-insensitive": ("tests.test_g11_census.Markers.test_lowercase_words_and_lookalikes_are_not_markers", "self.assertEqual(classes(run("),
    "marker word boundary dropped": ("tests.test_g11_census.Markers.test_lowercase_words_and_lookalikes_are_not_markers", "self.assertEqual(classes(run("),
    "quotes not tracked": ("tests.test_g11_census.Markers.test_javascript_comments_and_strings", "self.assertEqual(classes(run("),
    "string escapes ignored": ("tests.test_g11_census.Markers.test_escaped_quote_inside_string", "self.assertEqual(classes(run("),
    "hash comment needs no whitespace": ("tests.test_g11_census.Markers.test_shell_comments_and_strings", "self.assertEqual(classes(run("),
    "block comments not entered": ("tests.test_g11_census.Markers.test_javascript_comments_and_strings", "self.assertEqual(classes(run("),
    "block comment end ignored": ("tests.test_g11_census.Markers.test_javascript_comments_and_strings", "self.assertEqual(classes(run("),
    "mentions dropped": ("tests.test_g11_census.Markers.test_python_comment_is_marker_and_string_is_mention", "self.assertEqual(classes(run("),
    "noop detection off": ("tests.test_g11_census.Workflows.test_noop_steps", "self.assertEqual(found,"),
    "block scalar not read": ("tests.test_g11_census.Workflows.test_noop_steps", "self.assertEqual(found,"),
    "block scalar ends late": ("tests.test_g11_census.Workflows.test_noop_steps", "self.assertEqual(found,"),
    "inline comment kept": ("tests.test_g11_census.Workflows.test_noop_steps", "self.assertEqual(found,"),
    "tests counted as tools": ("tests.test_g11_census.ToolChecks.test_tests_and_init_files_are_not_tools", "self.assertEqual(sorted(report),"),
    "__init__ counted as tool": ("tests.test_g11_census.ToolChecks.test_tests_and_init_files_are_not_tools", "self.assertEqual(sorted(report),"),
    "basename ambiguity ignored": ("tests.test_g11_census.ToolChecks.test_ambiguous_basename_needs_the_full_path", 'self.assertFalse(report["tools/run.py"]["documented"])'),
    "path reference not recognised": ("tests.test_g11_census.ToolChecks.test_full_path_counts_even_when_basename_is_ambiguous", 'self.assertTrue(report["tools/run.py"]["documented"])'),
    "import not recognised": ("tests.test_g11_census.ToolChecks.test_documented_and_exercised_by_path_import_and_basename", 'self.assertTrue(report[path]["exercised"])'),
    "from-import list not recognised": ("tests.test_g11_census.ToolChecks.test_documented_and_exercised_by_path_import_and_basename", 'self.assertTrue(report[path]["exercised"])'),
    "basename not recognised": ("tests.test_g11_census.ToolChecks.test_documented_and_exercised_by_path_import_and_basename", 'self.assertTrue(report[path]["documented"])'),
    "transitive exercise off": ("tests.test_g11_census.ToolChecks.test_exercised_through_an_exercised_tool", 'self.assertEqual(report["tools/helper.py"]["exercised_by"],'),
    "site not production": ("tests.test_g11_census.Markers.test_site_marker_fails_and_vendor_marker_does_not", 'self.assertEqual(run({"docs/site/app.mjs":'),
    "research counted as production": ("tests.test_g11_census.PythonStubs.test_production_stub_fails_but_research_stub_does_not", 'self.assertEqual(research["status"],'),
    "vendor counted as site": ("tests.test_g11_census.Markers.test_site_marker_fails_and_vendor_marker_does_not", 'self.assertEqual(run({"docs/site/vendor/lib.js":'),
    "undecodable decoded leniently": ("tests.test_g11_census.FailClosed.test_undecodable_and_unparseable_production_files_fail", 'self.assertEqual(report["status"], "FAIL")'),
    "unparseable not a failure class": ("tests.test_g11_census.FailClosed.test_undecodable_and_unparseable_production_files_fail", 'self.assertEqual(report["status"], "FAIL")'),
    "legitimate context still fails": ("tests.test_g11_census.PythonStubs.test_legitimate_contexts_are_reported_but_do_not_fail", 'self.assertEqual(report["failing"], [])'),
    "markers do not fail": ("tests.test_g11_census.Markers.test_production_marker_fails", 'self.assertEqual(report["status"], "FAIL")'),
    "noop steps do not fail": ("tests.test_g11_census.Workflows.test_production_noop_fails", 'self.assertEqual(run({".github/workflows/w.yml": "      - run: true'),
    "tool findings do not fail": ("tests.test_g11_census.ToolChecks.test_undocumented_and_unexercised_tools_fail", 'self.assertEqual(result["status"], "FAIL")'),
    "exit status always zero": ("tests.test_g11_census.CommandLine.test_exit_codes_and_exact_committed_bytes", "self.assertEqual(result.returncode, 1)"),
    "unreadable revision exits zero": ("tests.test_g11_census.CommandLine.test_exit_codes_and_exact_committed_bytes", 'self.assertEqual(self.cli(root, "--rev", "no-such-revision").returncode, 2)'),
}


NEW_MUTATIONS = [
    ("decoy decorator names accepted", "census", "        if _trusted_decorator(dec, imports, scope_shadowed):", "        if _decorator_name(dec) in LEGIT_DECORATORS:",
     "tests.test_g11_census.AdversarialControls.test_decoy_names_do_not_exempt_stubs", 'self.assertEqual(sorted(f["detail"].split(":")[0] for f in report["failing"]),'),
    ("decoy abstract-base names accepted", "census", "            if _trusted_abstract_base(base, imports):", "            if _base_name(base) in LEGIT_BASES:",
     "tests.test_g11_census.AdversarialControls.test_decoy_names_do_not_exempt_stubs", 'self.assertEqual(sorted(f["detail"].split(":")[0] for f in report["failing"]),'),
    ("decoy exception suffix accepted", "census", "            if _builtin_exception_base(base, imports):",
     '            if _base_name(base).endswith(("Error", "Exception", "Warning")):',
     "tests.test_g11_census.AdversarialControls.test_decoy_names_do_not_exempt_stubs", 'self.assertEqual(sorted(f["detail"].split(":")[0] for f in report["failing"]),'),
    ("decoy request-handler suffix accepted", "census", '            if _http_request_handler_base(base, imports) and func.name == "log_message":',
     '            if _base_name(base).endswith("RequestHandler") and func.name == "log_message":',
     "tests.test_g11_census.AdversarialControls.test_decoy_names_do_not_exempt_stubs", 'self.assertEqual(sorted(f["detail"].split(":")[0] for f in report["failing"]),'),
    ("comment and string mentions count as exercise", "census",
     "def python_tool_references(text, tools, unique):\n    references = set()",
     "def python_tool_references(text, tools, unique):\n    references = {p for p in tools if p in text}",
     "tests.test_g11_census.AdversarialControls.test_comment_and_string_mentions_do_not_exercise_tools",
     'self.assertFalse(report["tools"]["tools/runner.py"]["exercised"])'),
    ("empty workflow run block not detected", "census", "        if not commands:", "        if False:",
     "tests.test_g11_census.AdversarialControls.test_empty_and_comment_only_run_blocks_fail", "self.assertEqual(found,"),
    ("YAML comments omitted", "census", '        elif path.endswith((".yml", ".yaml")):', "        elif False:",
     "tests.test_g11_census.AdversarialControls.test_yaml_comments_are_scanned_for_markers", "self.assertEqual(classes(report),"),
    ("shadowed builtin exception name accepted", "census",
     'elif isinstance(node, ast.Name) and node.id in builtins.__dict__ and node.id not in imports["shadowed"]:',
     'elif isinstance(node, ast.Name) and node.id in builtins.__dict__:',
     "tests.test_g11_census.AdversarialControls.test_decoy_names_do_not_exempt_stubs",
     'self.assertEqual(sorted(f["detail"].split(":")[0] for f in report["failing"]),'),
    ("mutation attribution accepts wrong assertion", "harness", '        and outcome.get("traceback_line") == expected_line', "        and True",
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     "self.assertFalse(g11m.qualifies_kill(wrong_line, expected, 17))"),
    ("function-local decorator shadow accepted", "census",
     "    if binding and binding in scope_shadowed:\n        return False",
     "    if False:\n        return False",
     "tests.test_g11_census.AdversarialControls.test_decoy_names_do_not_exempt_stubs",
     'self.assertEqual([(f["class"], f["detail"]) for f in local_report["findings"]'),
    ("python -c path mention counts as execution", "census",
     '                        if "-c" in tokens[1:]:\n                            continue',
     "                        if False:\n                            continue",
     "tests.test_g11_census.AdversarialControls.test_comment_and_string_mentions_do_not_exercise_tools",
     'self.assertFalse(python_cmd["tools"]["tools/runner.py"]["exercised"])'),
    ("shell -c path mention counts as execution", "census",
     '                if "-c" in tokens[1:]:\n                    continue',
     "                if False:\n                    continue",
     "tests.test_g11_census.AdversarialControls.test_comment_and_string_mentions_do_not_exercise_tools",
     'self.assertFalse(shell_cmd["tools"]["tools/runner.py"]["exercised"])'),
    ("template literal import text counts as static import", "census",
     '            if quote == "`":\n                if ch == "\\\\" and i + 1 < len(text):\n                    out.extend((" ", " "))\n                    i += 2\n                    continue\n                if ch == quote:\n                    quote = None\n                out.append("\\n" if ch == "\\n" else " ")\n                i += 1\n                continue',
     '            if quote == "`":\n                if ch == "\\\\" and i + 1 < len(text):\n                    out.extend((" ", " "))\n                    i += 2\n                    continue\n                if ch == quote:\n                    quote = None\n                out.append(ch)\n                i += 1\n                continue',
     "tests.test_g11_census.AdversarialControls.test_comment_and_string_mentions_do_not_exercise_tools",
     'self.assertFalse(js_cmd["tools"]["tools/runner.mjs"]["exercised"])'),
    ("parser drops method identity", "harness",
     '        result["test_id"] = header.group(2) + "." + header.group(1)',
     '        result["test_id"] = header.group(2)',
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     'self.assertEqual(parsed["test_id"], expected_parser)'),
    ("wrong test id credited", "harness",
     '        and outcome.get("test_id") == expected_test',
     "        and True",
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     "self.assertFalse(g11m.qualifies_kill(wrong_test, expected, 17))"),
    ("nonfailure status credited", "harness",
     '        and outcome.get("status") == "FAILED"',
     "        and True",
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     "self.assertFalse(g11m.qualifies_kill(not_failed, expected, 17))"),
    ("error outcome credited", "harness",
     '        and outcome.get("errors") == 0',
     "        and True",
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     "self.assertFalse(g11m.qualifies_kill(has_errors, expected, 17))"),
    ("skipped outcome credited", "harness",
     '        and outcome.get("skipped") == 0',
     "        and True",
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     "self.assertFalse(g11m.qualifies_kill(has_skips, expected, 17))"),
]


def _assertion_line(test_source, test_id, fragment):
    parts = test_id.split(".")
    class_name, method_name = parts[-2], parts[-1]
    tree = ast.parse(test_source)
    class_node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if class_node is None:
        raise HarnessError("target class missing: " + class_name)
    method = next((n for n in class_node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == method_name), None)
    if method is None:
        raise HarnessError("target test missing: " + test_id)
    matches = []
    for node in ast.walk(method):
        if isinstance(node, ast.Call):
            segment = ast.get_source_segment(test_source, node) or ""
            if fragment in segment:
                matches.append(node.lineno)
    if len(matches) != 1:
        raise HarnessError("assertion fragment %r matched %d calls in %s" % (fragment, len(matches), test_id))
    return matches[0]


def _parse_run(proc):
    output = proc.stdout + "\n" + proc.stderr
    summary = SUMMARY.search(output)
    ran = int(summary.group(1)) if summary else -1
    result = {"exit": proc.returncode, "ran": ran, "failures": 0, "errors": 0, "skipped": 0,
              "test_id": None, "traceback_method": None, "traceback_line": None,
              "exception_type": None, "parsed": bool(summary)}
    status = RESULT.findall(output)
    if status:
        result["status"] = status[-1][0]
        detail = status[-1][1] or ""
        counts = dict(re.findall(r"(failures|errors|skipped)=(\d+)", detail))
        result["failures"] = int(counts.get("failures", 0))
        result["errors"] = int(counts.get("errors", 0))
        result["skipped"] = int(counts.get("skipped", 0))
    header = FAIL_HEADER.search(output)
    if header:
        result["test_id"] = header.group(2) + "." + header.group(1)
        frames = TRACE_FRAME.findall(output)
        if frames:
            _path, line, method = frames[-1]
            result["traceback_line"] = int(line)
            result["traceback_method"] = method
        exc = EXCEPTION.search(output)
        if exc:
            result["exception_type"] = exc.group(1)
    return result


def qualifies_kill(outcome, expected_test, expected_line):
    expected_method = expected_test.rsplit(".", 1)[-1]
    return (
        outcome.get("parsed") is True
        and outcome.get("status") == "FAILED"
        and outcome.get("exit") == 1
        and outcome.get("ran") == 1
        and outcome.get("failures") == 1
        and outcome.get("errors") == 0
        and outcome.get("skipped") == 0
        and outcome.get("test_id") == expected_test
        and outcome.get("traceback_method") == expected_method
        and outcome.get("traceback_line") == expected_line
        and outcome.get("exception_type") == "AssertionError"
    )


def _run(workdir, test_id=None):
    command = [sys.executable, "-B", "-S"]
    if sys.flags.optimize:
        command.append("-O")
    command.extend(["-m", "unittest"])
    command.append(test_id if test_id else "tests.test_g11_census")
    command.append("-v")
    proc = subprocess.run(command, cwd=workdir, capture_output=True, text=True, timeout=300)
    return _parse_run(proc)


def _sandbox(tool_source, test_source, harness_source, workdir):
    os.makedirs(os.path.join(workdir, "tests"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "tools", "cbz"), exist_ok=True)
    Path(workdir, TESTS).write_text(test_source, encoding="utf-8")
    Path(workdir, TOOL).write_text(tool_source, encoding="utf-8")
    Path(workdir, HARNESS).write_text(harness_source, encoding="utf-8")


def run_selftest(root):
    tool_path = os.path.join(root, TOOL)
    test_path = os.path.join(root, TESTS)
    harness_path = os.path.join(root, HARNESS)
    with open(tool_path, encoding="utf-8") as fh:
        original_tool = fh.read()
    with open(test_path, encoding="utf-8") as fh:
        test_source = fh.read()
    with open(harness_path, encoding="utf-8") as fh:
        original_harness = fh.read()
    cases = []
    for name, old, new in MUTATIONS:
        target_test, fragment = TARGETS[name]
        cases.append((name, "census", old, new, target_test, fragment))
    cases.extend(NEW_MUTATIONS)
    names = [case[0] for case in cases]
    if len(names) != len(set(names)):
        raise HarnessError("duplicate mutation names")
    for name, target, old, new, test_id, fragment in cases:
        source = original_tool if target == "census" else original_harness
        if source.count(old) != 1 or old == new:
            raise HarnessError("mutation %r: target occurs %d times in %s (expected exactly 1)" %
                               (name, source.count(old), target))
        _assertion_line(test_source, test_id, fragment)
    baseline = None
    results = []
    with tempfile.TemporaryDirectory(prefix="g11-mut-") as tmp:
        base = os.path.join(tmp, "baseline")
        _sandbox(original_tool, test_source, original_harness, base)
        baseline = _run(base)
        for index, (name, target, old, new, test_id, fragment) in enumerate(cases):
            work = os.path.join(tmp, "m%02d" % index)
            tool_source = original_tool
            harness_source = original_harness
            if target == "census":
                tool_source = original_tool.replace(old, new, 1)
            else:
                harness_source = original_harness.replace(old, new, 1)
            _sandbox(tool_source, test_source, harness_source, work)
            outcome = _run(work, test_id)
            expected_line = _assertion_line(test_source, test_id, fragment)
            results.append(dict(outcome, mutation=name, target=target, expected_test=test_id,
                                expected_line=expected_line,
                                killed=qualifies_kill(outcome, test_id, expected_line)))
    baseline_ok = (baseline.get("parsed") and baseline.get("exit") == 0 and baseline.get("ran") == 33
                   and baseline.get("failures") == 0 and baseline.get("errors") == 0
                   and baseline.get("skipped") == 0 and baseline.get("status") == "OK")
    killed = sum(r["killed"] for r in results)
    return {"baseline": baseline, "mutations": results, "killed": killed, "total": len(results),
            "status": "PASS" if baseline_ok and killed == len(results) else "FAIL"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mutation self-test of the G11 census and its kill-attribution guard.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_selftest(os.path.abspath(args.root))
    except (HarnessError, OSError, subprocess.SubprocessError) as exc:
        print("G11_SELFTEST_HARNESS_ERROR: %s" % exc, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        for r in report["mutations"]:
            print("%-8s %s (target %s, exit %d, failures %d, errors %d, skips %d, line %s/%s)" %
                  ("KILLED" if r["killed"] else "SURVIVED", r["mutation"], r["target"], r["exit"],
                   r["failures"], r["errors"], r["skipped"], r["traceback_line"], r["expected_line"]))
        print("G11_SELFTEST_%s: baseline %d tests, %d/%d mutations killed" %
              (report["status"], report["baseline"].get("ran", -1), report["killed"], report["total"]))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
