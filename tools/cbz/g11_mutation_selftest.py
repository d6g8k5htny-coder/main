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
    ("Protocol/ABC context ignored", '            if base_kind == "Protocol":', "            if False:"),
    ("exception context ignored",
     "            if _builtin_exception_base(base, imports, scope_shadowed):\n                continue\n",
     "            if _builtin_exception_base(base, imports, scope_shadowed):\n                return \"method of an exception class\"\n"),
    ("log_message context ignored", '            if (_http_request_handler_base(base, imports, scope_shadowed)\n                    and func.name == "log_message"):', "            if False:"),
    ("log_message context too broad", '            if (_http_request_handler_base(base, imports, scope_shadowed)\n                    and func.name == "log_message"):',
     '            if _http_request_handler_base(base, imports, scope_shadowed):'),
    ("module docstring not recorded", "                docstring = bool(ast.get_docstring(ast.parse(files[path])))",
     "                docstring = True"),
    ("class bodies not visited", "                visit(child, classes + [child], scope_shadowed | _scope_bindings(child), child.body)", "                pass"),
    ("nested functions not visited", "                visit(child, [], scope_shadowed | _scope_bindings(child), child.body)\n            else:",
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
    ("inline comment kept",
     "            commands = [_strip_shell_comment(value).strip()]\n            commands = [_strip_shell_comment(c).strip() for c in commands]",
     "            commands = [value.strip()]\n            commands = [c.strip() for c in commands]"),
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
    ("transitive exercise off", '                if other != path and report[other]["referenced"] and other in texts and path in tool_dependencies.get(other, set()):',
     '                if False and other != path and report[other]["referenced"] and other in texts and path in tool_dependencies.get(other, set()):'),
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
FAIL_HEADER = re.compile(r"^FAIL: ([^(]+) \(([^)]+)\)(?: \[.*\])?(?: \(.*\))?$", re.M)
TRACE_FRAME = re.compile(r'File "([^"]*test_g11_census\.py)", line (\d+), in ([A-Za-z_]\w*)')
EXCEPTION = re.compile(r"(?m)^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning|Interrupt|Exit|StopIteration)):")


class HarnessError(Exception):
    pass


TARGETS = {
    "pass not detected": ("tests.test_g11_census.PythonStubs.test_each_stub_body", 'self.assertEqual(stubs("def a():\\n    pass\\n"), [(1, "a: pass")])'),
    "ellipsis not detected": ("tests.test_g11_census.PythonStubs.test_each_stub_body", 'self.assertEqual(stubs("def a():\\n    ...\\n"), [(1, "a: ellipsis")])'),
    "NotImplementedError not detected": ("tests.test_g11_census.PythonStubs.test_each_stub_body", 'self.assertEqual(stubs("def a():\\n    raise NotImplementedError\\n"), [(1, "a: raise NotImplementedError")])'),
    "raise call not unwrapped": ("tests.test_g11_census.PythonStubs.test_each_stub_body", 'self.assertEqual(stubs("def a():\\n    raise NotImplementedError(\'later\')\\n"),\n                         [(1, "a: raise NotImplementedError")])'),
    "docstring-only not reported": ("tests.test_g11_census.PythonStubs.test_each_stub_body", 'self.assertEqual(stubs(\'def a():\\n    """Only a docstring."""\\n\'), [(1, "a: docstring only")])'),
    "docstring not stripped": ("tests.test_g11_census.PythonStubs.test_docstring_then_stub_body", '[(1, "a: pass")])'),
    "decorator context ignored": ("tests.test_g11_census.PythonStubs.test_legitimate_contexts_are_reported_but_do_not_fail", "self.assertEqual(contexts,"),
    "Protocol/ABC context ignored": ("tests.test_g11_census.PythonStubs.test_legitimate_contexts_are_reported_but_do_not_fail", "self.assertEqual(contexts,"),
    "exception context ignored": ("tests.test_g11_census.ReviewResidualControls.test_stub_exemptions_are_immediate_and_scope_safe", 'self.assertIn(name, stub_names)'),
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
    "basename ambiguity ignored": ("tests.test_g11_census.ToolChecks.test_ambiguous_basename_needs_the_full_path", 'self.assertFalse(ambiguous["tools/run.py"]["documented"])'),
    "path reference not recognised": ("tests.test_g11_census.ToolChecks.test_full_path_counts_even_when_basename_is_ambiguous", 'self.assertTrue(report["tools/run.py"]["documented"])'),
    "import not recognised": ("tests.test_g11_census.ToolChecks.test_documented_and_exercised_by_path_import_and_basename", 'self.assertTrue(report[path]["referenced"])'),
    "from-import list not recognised": ("tests.test_g11_census.ToolChecks.test_documented_and_exercised_by_path_import_and_basename", 'self.assertTrue(report[path]["referenced"])'),
    "basename not recognised": ("tests.test_g11_census.ToolChecks.test_documented_and_exercised_by_path_import_and_basename", 'self.assertTrue(report[path]["documented"])'),
    "transitive exercise off": ("tests.test_g11_census.ToolChecks.test_exercised_through_an_exercised_tool", 'self.assertEqual(report["tools/helper.py"]["referenced_by"],'),
    "site not production": ("tests.test_g11_census.Markers.test_site_marker_fails_and_vendor_marker_does_not", 'self.assertEqual(run({"docs/site/app.mjs":'),
    "research counted as production": ("tests.test_g11_census.PythonStubs.test_production_stub_fails_but_research_stub_does_not", 'self.assertEqual(research["status"],'),
    "vendor counted as site": ("tests.test_g11_census.Markers.test_site_marker_fails_and_vendor_marker_does_not", 'self.assertEqual(run({"docs/site/vendor/lib.js":'),
    "undecodable decoded leniently": ("tests.test_g11_census.FailClosed.test_undecodable_and_unparseable_production_files_fail", 'self.assertEqual([f["class"] for f in report["failing"]], [expected])'),
    "unparseable not a failure class": ("tests.test_g11_census.FailClosed.test_undecodable_and_unparseable_production_files_fail", 'self.assertEqual(report["status"], "FAIL")'),
    "legitimate context still fails": ("tests.test_g11_census.PythonStubs.test_legitimate_contexts_are_reported_but_do_not_fail", "self.assertEqual(contexts,"),
    "markers do not fail": ("tests.test_g11_census.Markers.test_production_marker_fails", 'self.assertEqual(report["status"], "FAIL")'),
    "noop steps do not fail": ("tests.test_g11_census.Workflows.test_production_noop_fails", 'self.assertEqual(run({".github/workflows/w.yml": "      - run: true'),
    "tool findings do not fail": ("tests.test_g11_census.ToolChecks.test_undocumented_and_unexercised_tools_fail", 'self.assertEqual(result["status"], "FAIL")'),
    "exit status always zero": ("tests.test_g11_census.CommandLine.test_exit_codes_and_exact_committed_bytes", "self.assertEqual(result.returncode, 1)"),
    "unreadable revision exits zero": ("tests.test_g11_census.CommandLine.test_exit_codes_and_exact_committed_bytes", 'self.assertEqual(self.cli(root, "--rev", "no-such-revision").returncode, 2)'),
}


NEW_MUTATIONS = [
    ("decoy decorator names accepted", "census",
     "        if _trusted_decorator(dec, imports, scope_shadowed):\n            resolved = _resolve_scoped_import(dec, imports, scope_shadowed)",
     '        if _decorator_name(dec) in LEGIT_DECORATORS:\n            resolved = ("abc", "abstractmethod") if _decorator_name(dec) == "abstractmethod" else ("typing", "overload")',
     "tests.test_g11_census.AdversarialControls.test_decoy_names_do_not_exempt_stubs", 'self.assertEqual(sorted(f["detail"].split(":")[0] for f in report["failing"]),'),
    ("decoy abstract-base names accepted", "census", '            if base_kind == "Protocol":', "            if _base_name(base) in LEGIT_BASES:",
     "tests.test_g11_census.AdversarialControls.test_decoy_names_do_not_exempt_stubs", 'self.assertEqual(sorted(f["detail"].split(":")[0] for f in report["failing"]),'),
    ("decoy request-handler suffix accepted", "census", '            if (_http_request_handler_base(base, imports, scope_shadowed)\n                    and func.name == "log_message"):',
     '            if _base_name(base).endswith("RequestHandler") and func.name == "log_message":',
     "tests.test_g11_census.AdversarialControls.test_decoy_names_do_not_exempt_stubs", 'self.assertEqual(sorted(f["detail"].split(":")[0] for f in report["failing"]),'),
    ("comment and string mentions count as exercise", "census",
     'def python_tool_references(text, source_path, tools, unique, mode="script"):\n    references = set()',
     'def python_tool_references(text, source_path, tools, unique, mode="script"):\n    references = {p for p in tools if p in text}',
     "tests.test_g11_census.AdversarialControls.test_comment_and_string_mentions_do_not_exercise_tools",
     'self.assertFalse(report["tools"]["tools/runner.py"]["referenced"])'),
    ("empty workflow run block not detected", "census", "        if not commands:", "        if False:",
     "tests.test_g11_census.AdversarialControls.test_empty_and_comment_only_run_blocks_fail", "self.assertEqual(found,"),
    ("YAML comments omitted", "census", '        elif path.endswith((".yml", ".yaml")):', "        elif False:",
     "tests.test_g11_census.AdversarialControls.test_yaml_comments_are_scanned_for_markers", "self.assertEqual(classes(report),"),
    ("mutation attribution accepts wrong assertion", "harness", '        and outcome.get("traceback_line") == expected_line\n        and outcome.get("exception_type")',
     '        and True\n        and outcome.get("exception_type")',
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     "self.assertFalse(g11m.qualifies_kill(wrong_line, expected, 17))"),
    ("function-local decorator shadow accepted", "census",
     "    if binding and binding in scope_shadowed:\n        return None",
     "    if False:\n        return None",
     "tests.test_g11_census.AdversarialControls.test_decoy_names_do_not_exempt_stubs",
     'self.assertEqual([(f["class"], f["detail"]) for f in local_report["findings"]'),
    ("python -c path mention counts as execution", "census",
     '        if parsing_options and token in terminating_options:\n            return set()',
     '        if parsing_options and token in terminating_options and token != "-c":\n            return set()\n        if parsing_options and token == "-c":\n            continue',
     "tests.test_g11_census.AdversarialControls.test_comment_and_string_mentions_do_not_exercise_tools",
     'self.assertFalse(python_cmd["tools"]["tools/runner.py"]["referenced"])'),
    ("shell -c path mention counts as execution", "census",
     '                    if "-c" not in tokens[1:]:',
     "                    if True:",
     "tests.test_g11_census.AdversarialControls.test_comment_and_string_mentions_do_not_exercise_tools",
     'self.assertFalse(shell_cmd["tools"]["tools/runner.py"]["referenced"])'),
    ("template literal import text counts as static import", "census",
     '            if quote == "`":\n                if ch == "\\\\" and i + 1 < len(text):\n                    out.extend((" ", " "))\n                    i += 2\n                    continue\n                if ch == quote:\n                    quote = None\n                out.append("\\n" if ch == "\\n" else " ")\n                i += 1\n                continue',
     '            if quote == "`":\n                if ch == "\\\\" and i + 1 < len(text):\n                    out.extend((" ", " "))\n                    i += 2\n                    continue\n                if ch == quote:\n                    quote = None\n                out.append(ch)\n                i += 1\n                continue',
     "tests.test_g11_census.AdversarialControls.test_comment_and_string_mentions_do_not_exercise_tools",
     'self.assertFalse(js_cmd["tools"]["tools/runner.mjs"]["referenced"])'),
    ("parser drops method identity", "harness",
     '            test_id += "." + method\n        result["test_id"] = test_id',
     '            pass\n        result["test_id"] = test_id',
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     'self.assertEqual(parsed["test_id"], expected_parser)'),
    ("parser repeats qualified method identity", "harness",
     '        if not test_id.endswith("." + method):\n            test_id += "." + method',
     '        if True:\n            test_id += "." + method',
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     'self.assertEqual(parsed_current["test_id"], expected_parser)'),
    ("subTest failure header not parsed", "harness",
     'FAIL_HEADER = re.compile(r"^FAIL: ([^(]+) \\(([^)]+)\\)(?: \\[.*\\])?(?: \\(.*\\))?$", re.M)\nTRACE_FRAME',
     'FAIL_HEADER = re.compile(r"^FAIL: ([^(]+) \\(([^)]+)\\)$", re.M)\nTRACE_FRAME',
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     'self.assertEqual(parsed_subtest["test_id"], expected_parser)'),
    ("wrong test id credited", "harness",
     '        and outcome.get("test_id") == expected_test\n        and outcome.get("traceback_method")',
     '        and True\n        and outcome.get("traceback_method")',
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     "self.assertFalse(g11m.qualifies_kill(wrong_test, expected, 17))"),
    ("nonfailure status credited", "harness",
     '        and outcome.get("status") == "FAILED"\n        and outcome.get("exit")',
     '        and True\n        and outcome.get("exit")',
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     "self.assertFalse(g11m.qualifies_kill(not_failed, expected, 17))"),
    ("error outcome credited", "harness",
     '        and outcome.get("failures") == 1\n        and outcome.get("errors") == 0',
     '        and outcome.get("failures") == 1\n        and True',
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     "self.assertFalse(g11m.qualifies_kill(has_errors, expected, 17))"),
    ("skipped outcome credited", "harness",
     '        and outcome.get("skipped") == 0\n        and outcome.get("test_id")',
     '        and True\n        and outcome.get("test_id")',
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     "self.assertFalse(g11m.qualifies_kill(has_skips, expected, 17))"),
    ("quoted workflow command boundary ignored", "census",
     "                value = value[1:-1]",
     "                value = value",
     "tests.test_g11_census.AdversarialControls.test_comment_and_string_mentions_do_not_exercise_tools",
     "self.assertEqual(g11._workflow_commands(workflow_text)[0][1], [\"sh -c 'echo tools/runner.py'\"])"),
    ("FAIL header mistaken for exception", "harness",
     'EXCEPTION = re.compile(r"(?m)^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning|Interrupt|Exit|StopIteration)):")\n\n\nclass HarnessError',
     'EXCEPTION = re.compile(r"(?m)^([A-Za-z_][A-Za-z0-9_]*):")\n\n\nclass HarnessError',
     "tests.test_g11_census.MutationAttribution.test_mutation_kill_requires_exact_target_assertion",
     "self.assertTrue(g11m.qualifies_kill(parsed, expected_parser, 123))"),
    ("outer class exemption leaks", "census",
      "    for cls in classes[-1:]:", "    for cls in reversed(classes):",
      "tests.test_g11_census.ReviewResidualControls.test_stub_exemptions_are_immediate_and_scope_safe",
      "self.assertIn(name, stub_names)"),
    ("local base shadow accepted", "census",
      "    if binding and binding in scope_shadowed:\n        return None",
      "    if False:\n        return None",
      "tests.test_g11_census.ReviewResidualControls.test_stub_exemptions_are_immediate_and_scope_safe",
      'self.assertIn("shadowed_abc", stub_names)'),
    ("uncalled helper counts as exercise", "census",
      "        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):\n            # A definition is not a call: only decorators/defaults execute here.",
      "        if False:\n            # A definition is not a call: only decorators/defaults execute here.",
      "tests.test_g11_census.ReviewResidualControls.test_tool_reachability_and_heredoc_boundaries",
      'self.assertFalse(uncalled["tools"]["tools/runner.py"]["referenced"])'),
    ("heredoc body counted as command", "census",
      "            if heredoc:", "            if False:",
      "tests.test_g11_census.ReviewResidualControls.test_tool_reachability_and_heredoc_boundaries",
      'self.assertFalse(heredoc["tools"]["tools/runner.py"]["referenced"])'),
    ("mutation inventory shrink accepted", "harness",
      "\n    if tuple(cases) != declared:\n        raise HarnessError(\"mutation inventory mismatch: complete case tuples differ\")\n", "\n    if False:\n        raise HarnessError(\"mutation inventory mismatch: complete case tuples differ\")\n",
      "tests.test_g11_census.ReviewResidualControls.test_mutation_inventory_is_sealed",
      'self.assertRaisesRegex(g11m.HarnessError, "mutation inventory")'),
    ("hostile Python environment survives", "harness",
      "    for name in PYTHON_ENVIRONMENT_KEYS:\n        env.pop(name, None)", "    for name in ():\n        env.pop(name, None)",
      "tests.test_g11_census.ReviewResidualControls.test_mutation_child_environment_is_sanitized",
      'self.assertEqual(outcome.get("status"), "OK")'),
    ("in-scope tracked symlink silently skipped", "census",
      "        if path.endswith(SCANNED) and (kind != b\"blob\" or mode == b\"120000\"):",
      "        if False:",
      "tests.test_g11_census.ReviewResidualControls.test_in_scope_tracked_symlink_is_rejected",
      'self.assertRaisesRegex(ValueError, "in-scope non-regular tracked path")'),
    ("decoy TestCase base credited", "census",
     'and any(_resolve_import(base, imports) == ("unittest", "TestCase")',
     'and any(_base_name(base) == "TestCase"',
     "tests.test_g11_census.ReviewResidualControls.test_tool_reachability_and_heredoc_boundaries",
     'self.assertFalse(decoy["tools"]["tools/runner.py"]["referenced"])'),
    ("workflow defaults accepted as steps", "census",
     "        if not _is_step_run(lines, index):\n            continue",
     "        if False:\n            continue",
     "tests.test_g11_census.Workflows.test_job_defaults_run_mapping_is_not_a_step",
     "self.assertEqual(g11.noop_steps(text), [])"),
    ("static path assignments not bound", "census",
     "            _bind_python_value(\n                node, value, local, module_bindings, scope, global_names,\n                destination_for_name)",
     "            _bind_python_value(\n                node, None, local, module_bindings, scope, global_names,\n                destination_for_name)",
     "tests.test_g11_census.ToolChecks.test_reachable_python_path_values_are_execution_references",
     "self.assertEqual(referenced,"),
    ("spec loader path ignored", "census",
     "            if call_name == \"importlib.util.spec_from_file_location\" and len(node.args) >= 2:\n                value = argument_value(1)\n                references.update(_tool_values(value, tools))",
     "            if call_name == \"importlib.util.spec_from_file_location\" and len(node.args) >= 2:\n                value = argument_value(1)\n                references.update(())",
     "tests.test_g11_census.ToolChecks.test_reachable_python_path_values_are_execution_references",
     "self.assertEqual(referenced,"),
    ("runpy path ignored", "census",
     "            if call_name == \"runpy.run_path\" and node.args:\n                value = argument_value(0)\n                references.update(_tool_values(value, tools))",
     "            if call_name == \"runpy.run_path\" and node.args:\n                value = argument_value(0)\n                references.update(())",
     "tests.test_g11_census.ToolChecks.test_reachable_python_path_values_are_execution_references",
     "self.assertEqual(referenced,"),
    ("subprocess path list ignored", "census",
     "                    if isinstance(value, (list, _StaticTuple, _StaticAlternatives)):\n                        references.update(_command_tool_values(value, tools))\n                        continue",
     "                    if isinstance(value, (list, _StaticTuple, _StaticAlternatives)):\n                        references.update(())\n                        continue",
     "tests.test_g11_census.ToolChecks.test_reachable_python_path_values_are_execution_references",
     "self.assertEqual(referenced,"),
    ("environment path default ignored", "census",
     "    if resolved_call == \"os.environ.get\" and len(node.args) >= 2:\n        return _python_static_value(node.args[1], local, module, source_path, imports, scope_shadowed)",
     "    if resolved_call == \"os.environ.get\" and len(node.args) >= 2:\n        return None",
     "tests.test_g11_census.ToolChecks.test_reachable_python_path_values_are_execution_references",
     "self.assertEqual(referenced,"),
    ("legacy os.path construction ignored", "census",
     "    if resolved_call in {\"os.path.dirname\", \"posixpath.dirname\"} and node.args:\n        value = _python_static_value(node.args[0], local, module, source_path, imports, scope_shadowed)\n        return posixpath.dirname(value) if isinstance(value, str) else None",
     "    if resolved_call in {\"os.path.dirname\", \"posixpath.dirname\"} and node.args:\n        value = _python_static_value(node.args[0], local, module, source_path, imports, scope_shadowed)\n        return None",
     "tests.test_g11_census.ToolChecks.test_reachable_python_path_values_are_execution_references",
     "self.assertEqual(referenced,"),
    ("concatenated command lists ignored", "census",
     "            if isinstance(left_value, list) and isinstance(right_value, list):\n                results.append(left_value + right_value)",
     "            if isinstance(left_value, list) and isinstance(right_value, list):\n                return None",
     "tests.test_g11_census.ToolChecks.test_reachable_python_path_values_are_execution_references",
     "self.assertEqual(referenced,"),
    ("lazy-load none guard ignored", "census",
     "                    if not _is_none_guard(node.test):\n                        if names:",
     "                    if True:\n                        if names:",
     "tests.test_g11_census.ToolChecks.test_reachable_python_path_values_are_execution_references",
     "self.assertEqual(referenced,"),
    ("arbitrary unknown branch accepted", "census",
     "                    if not _is_none_guard(node.test):\n                        if names:",
     "                    if False:\n                        if names:",
     "tests.test_g11_census.ToolChecks.test_path_values_need_reachable_execution_not_mentions",
     "self.assertEqual({path: item[\"referenced\"] for path, item in report.items()},"),
    ("process wrapper forwarding ignored", "census",
     "                positions = (wrapper[0] if wrapper is not None\n                             and not reassigned and call_order > wrapper_order\n                             and _python_call_signature_binds(wrapper[2], node) else ())",
     "                positions = ()",
     "tests.test_g11_census.ToolChecks.test_reachable_python_path_values_are_execution_references",
     "self.assertEqual(referenced,"),
    ("nonprocess wrapper accepted", "census",
     "                    forwarded = (node.args[0] if resolved in process_calls\n"
     "                                 and (resolved not in subprocess_calls\n"
     "                                      or _subprocess_uses_supported_argv(node))\n"
     "                                 else None)",
     "                    forwarded = (node.args[0] if True\n"
     "                                 and (resolved not in subprocess_calls\n"
     "                                      or _subprocess_uses_supported_argv(node))\n"
     "                                 else None)",
     "tests.test_g11_census.ToolChecks.test_path_values_need_reachable_execution_not_mentions",
     "self.assertEqual({path: item[\"referenced\"] for path, item in report.items()},"),
    ("sibling with_name path ignored", "census",
     "        if node.func.attr == \"with_name\" and isinstance(base, str) and node.args:",
     "        if False:",
     "tests.test_g11_census.ToolChecks.test_sibling_tool_path_execution_is_a_reference",
     "self.assertTrue(report[\"tools/family/worker.py\"][\"referenced\"])"),
    ("JavaScript URL tool binding ignored", "census",
     "        if target in tools:\n            function_blocks = {(scope[2], scope[3]) for scope in function_scopes}",
     "        if False:\n            function_blocks = {(scope[2], scope[3]) for scope in function_scopes}",
     "tests.test_g11_census.ToolChecks.test_js_url_child_process_execution_is_a_reference",
     "self.assertEqual({path: item[\"referenced\"] for path, item in report.items()},"),
    ("JavaScript URL mention accepted without child process", "census",
     "    if not names:\n        return []",
     "    if not names:\n        names = {\"spawnSync\", \"spawn\", \"execFileSync\", \"execFile\"}",
     "tests.test_g11_census.ToolChecks.test_js_url_child_process_execution_is_a_reference",
     "self.assertEqual({path: item[\"referenced\"] for path, item in report.items()},"),
    ("decoy Python loader APIs accepted", "census",
     "            if call_name == \"importlib.util.spec_from_file_location\" and len(node.args) >= 2:",
     "            if _dotted_name(node.func).endswith(\".spec_from_file_location\") and len(node.args) >= 2:",
     "tests.test_g11_census.ToolChecks.test_path_values_need_reachable_execution_not_mentions",
     "self.assertEqual({path: item[\"referenced\"] for path, item in report.items()},"),
    ("decoy run_path API accepted", "census",
     "            if call_name == \"runpy.run_path\" and node.args:",
     "            if _dotted_name(node.func).endswith(\".run_path\") and node.args:",
     "tests.test_g11_census.ToolChecks.test_path_values_need_reachable_execution_not_mentions",
     "self.assertEqual({path: item[\"referenced\"] for path, item in report.items()},"),
    ("decoy JavaScript child process accepted", "census",
     "    child_names = set()",
     "    child_names = {\"spawnSync\", \"spawn\", \"execFileSync\", \"execFile\"}",
     "tests.test_g11_census.ToolChecks.test_js_url_child_process_execution_is_a_reference",
     "self.assertEqual({path: item[\"referenced\"] for path, item in report.items()},"),
    ("absolute parent traversal accepts the filesystem root", "census",
     "            if parent == current:\n                return None",
     "            if parent == current:\n                return current",
     "tests.test_g11_census.ToolChecks.test_absolute_parent_paths_stop_at_the_root",
     "self.assertIsNone(g11._python_static_value(beyond,"),
    ("resolved command list credits command-mode arguments", "census",
     "        if parsing_options and token in terminating_options:\n            return set()",
     "        if parsing_options and token in terminating_options:\n            return _tool_values(arguments[-1], tools)",
     "tests.test_g11_census.ToolChecks.test_resolved_command_lists_reject_command_mode_arguments",
     "self.assertEqual({path: item[\"referenced\"] for path, item in report.items()},"),
    ("unreachable wrapper process call accepted", "census",
     "                branch = node.body if truth else node.orelse\n                return visit_body(branch)",
     "                branch = [*node.body, *node.orelse]\n                return visit_body(branch)",
     "tests.test_g11_census.ToolChecks.test_wrapper_process_call_must_be_reachable",
     "self.assertFalse(self.tools(tree)[\"tools/dead.py\"][\"referenced\"])"),
    ("function-local process import ignored", "census",
     "        if scope is not None:\n            _update_local_import_bindings(node, local_imports)",
     "        if False:\n            _update_local_import_bindings(node, local_imports)",
     "tests.test_g11_census.ToolChecks.test_function_local_process_import_is_trusted",
     "self.assertTrue(self.tools(tree)[\"tools/worker.py\"][\"referenced\"])"),
    ("JavaScript same-line mention accepted outside call arguments", "census",
     "    for call_start, callee, arguments in _js_call_arguments(visible, child_names):",
     "    for arguments in visible.splitlines():\n        call_start = len(visible)\n        callee = ''",
     "tests.test_g11_census.ToolChecks.test_js_url_child_process_execution_is_a_reference",
     "self.assertEqual({path: item[\"referenced\"] for path, item in report.items()},"),
    ("destructured path values rebound as aggregate", "census",
     "            assign(element, element_value)",
     "            assign(element, target_value)",
     "tests.test_g11_census.ToolChecks.test_destructured_paths_bind_corresponding_values",
     "self.assertEqual({path: item[\"referenced\"] for path, item in report.items()},"),
    ("rebound wrapper parameter still forwarded", "census",
     "            if target.id in parameters:\n                rebound.add(target.id)\n            aliases.pop(target.id, None)",
     "            if False:\n                rebound.add(target.id)\n            if target.id not in parameters:\n                aliases.pop(target.id, None)",
     "tests.test_g11_census.ToolChecks.test_wrapper_rebinding_invalidates_forwarding",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("local static shadow inherits module value", "census",
     "        if node.id in scope_shadowed:\n            return None",
     "        if False:\n            return None",
     "tests.test_g11_census.ToolChecks.test_local_path_shadows_do_not_inherit_module_values",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("JavaScript quoted and property names credited", "census",
     "        identifiers = _js_direct_child_argument_identifiers(arguments)",
     "        identifiers = set(bindings)",
     "tests.test_g11_census.ToolChecks.test_js_quoted_and_property_names_do_not_credit_bindings",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("forwarded command-mode arguments credited", "census",
     "                        references.update(_command_tool_values(value, tools))\n            if call_name == \"importlib.util.spec_from_file_location\"",
     "                        references.update(_tool_values(value, tools))\n            if call_name == \"importlib.util.spec_from_file_location\"",
     "tests.test_g11_census.ToolChecks.test_wrapper_forwarding_rejects_command_mode_arguments",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("augmented assignment retains stale static value", "census",
     "        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):\n            augmented_prior = None",
     "        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):\n            augmented_prior = None",
     "tests.test_g11_census.ToolChecks.test_augmented_assignment_replaces_static_path_value",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript property call accepted as imported callee", "census",
     '        r"(?<![\\w$.])(" + "|".join(map(re.escape, sorted(names))) + r")(?![\\w$])\\s*\\(")',
     '        r"\\b(" + "|".join(map(re.escape, sorted(names))) + r")\\s*\\(")',
     "tests.test_g11_census.ToolChecks.test_js_property_calls_do_not_use_named_child_process_imports",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("unresolved command element dropped", "census",
     "        else:\n            values.append(_UNRESOLVED_STATIC if value is None else value)",
     "        else:\n            if value is not None:\n                values.append(value)",
     "tests.test_g11_census.ToolChecks.test_unresolved_command_elements_do_not_shift_the_executable",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("boolean or evaluated in reverse order", "census",
     "        for value_node in node.values:\n            value = _python_static_value(value_node, local, module, source_path, imports, scope_shadowed)",
     "        for value_node in reversed(node.values):\n            value = _python_static_value(value_node, local, module, source_path, imports, scope_shadowed)",
     "tests.test_g11_census.ToolChecks.test_boolean_or_static_values_follow_short_circuit_order",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("shadowed callee inherits module wrapper", "census",
     "                if wrapper is None and node.func.id not in static_shadowed:\n                    key = (None, node.func.id)\n                    wrapper = wrappers.get(key)",
     "                if wrapper is None:\n                    key = (None, node.func.id)\n                    wrapper = wrappers.get(key)",
     "tests.test_g11_census.ToolChecks.test_local_callee_shadows_do_not_inherit_module_wrappers",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("quoted JavaScript text accepted as child process call", "census",
     "    for match in starts.finditer(searchable):",
     "    for match in starts.finditer(text):",
     "tests.test_g11_census.ToolChecks.test_js_quoted_call_text_does_not_execute_bindings",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("resolved starred command elements ignored", "census",
     "        if isinstance(element, ast.Starred):\n            if isinstance(value, (list, _StaticTuple)):\n                values.extend(value)\n            else:\n                values.append(_UNRESOLVED_STATIC)",
     "        if isinstance(element, ast.Starred):\n            return None",
     "tests.test_g11_census.ToolChecks.test_unresolved_command_elements_do_not_shift_the_executable",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("dynamic command suffix discards proven prefix", "census",
     "        else:\n            values.append(_UNRESOLVED_STATIC if value is None else value)",
     "        else:\n            if value is None:\n                return None\n            values.append(value)",
     "tests.test_g11_census.ToolChecks.test_dynamic_command_suffix_does_not_hide_a_proven_tool_prefix",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("unittest lifecycle methods ignored", "census",
     "                    and (node.name.startswith(\"test_\")\n                         or node.name in {\"setUp\", \"tearDown\", \"setUpClass\", \"tearDownClass\"})):",
     "                    and node.name.startswith(\"test_\")):",
     "tests.test_g11_census.ToolChecks.test_unittest_lifecycle_methods_are_reachable",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("import loader wrapper forwarding ignored", "census",
     '    path_calls = {"importlib.util.spec_from_file_location": 1, "runpy.run_path": 0}',
     '    path_calls = {"runpy.run_path": 0}\n    for candidate in ast.walk(tree):\n        if isinstance(candidate, ast.FunctionDef):\n            candidate.decorator_list = [ast.copy_location(ast.Name(id="mutated", ctx=ast.Load()), candidate)]',
     "tests.test_g11_census.ToolChecks.test_import_loader_wrapper_forwards_a_proven_tool_path",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("class method registered as module wrapper", "census",
     "            elif isinstance(statement, ast.ClassDef):\n                register(statement.body, id(statement))",
     "            elif isinstance(statement, ast.ClassDef):\n                register(statement.body, containing_scope)",
     "tests.test_g11_census.ToolChecks.test_class_methods_do_not_register_as_module_wrappers",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("conditional command alternatives merged", "census",
     "        if truth is not None:\n            selected = node.body if truth else node.orelse\n            return _python_static_value(selected, local, module, source_path, imports,\n                                        scope_shadowed)",
     "        if truth is not None:\n            body = _python_static_value(node.body, local, module, source_path, imports, scope_shadowed)\n            other = _python_static_value(node.orelse, local, module, source_path, imports, scope_shadowed)\n            if isinstance(body, list) and isinstance(other, list):\n                return body + [value for value in other if value not in body]",
     "tests.test_g11_census.ToolChecks.test_conditional_command_lists_select_only_the_taken_branch",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("reassigned JavaScript URL binding retains stale path", "census",
     "            if any(not _js_binding_or_property_written(\n                       visible[declaration_end:cutoff], name, mutable_object)\n                   for cutoff in cutoffs if cutoff >= declaration_end):\n                references.add(target)",
     "            if True:\n                references.add(target)",
     "tests.test_g11_census.ToolChecks.test_js_reassigned_url_binding_does_not_keep_its_old_path",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("path constructor ignores later components", "census",
     "    if resolved_call == \"pathlib.Path\" and node.args:\n        values = [_python_static_value(arg, local, module, source_path, imports, scope_shadowed)\n                  for arg in node.args]\n        if values and all(isinstance(value, str) for value in values):\n            return posixpath.normpath(posixpath.join(*values))\n        return None",
     "    if resolved_call == \"pathlib.Path\" and node.args:\n        return _python_static_value(node.args[0], local, module, source_path, imports, scope_shadowed)",
     "tests.test_g11_census.ToolChecks.test_path_constructor_resolves_every_component",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("command element assignment retains stale sequence", "census",
     "                _invalidate_python_binding(\n                    root.id, local, module, scope, global_names,\n                    destination_for_name, recursive=recursive)",
     "                pass",
     "tests.test_g11_census.ToolChecks.test_command_element_assignment_invalidates_static_sequence",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript parameter shadow accepted as imported callee", "census",
     "        if _js_parameter_shadows(visible, call_start, callee):\n            continue",
     "        if False:\n            continue",
     "tests.test_g11_census.ToolChecks.test_js_parameter_shadows_named_child_process_import",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("mode switch after script mistaken for interpreter mode", "census",
     "    arguments = value[1:]\n    if first in python_interpreters:",
     "    arguments = value[1:]\n    if any(token in {\"-c\", \"-m\"} for token in arguments if isinstance(token, str)):\n        return set()\n    if first in python_interpreters:",
     "tests.test_g11_census.ToolChecks.test_interpreter_mode_switch_after_script_is_program_argument",
     'self.assertTrue(self.tools(tree)["tools/live.py"]["referenced"])'),
    ("wrapper element write retains caller provenance", "census",
     "                if isinstance(root, ast.Name):\n                    invalidate(root.id)",
     "                if False:\n                    invalidate(root.id)",
     "tests.test_g11_census.ToolChecks.test_wrapper_element_write_invalidates_forwarding",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("helper element write falls back to module binding", "census",
     "                _invalidate_python_binding(\n                    root.id, local, module, scope, global_names,\n                    destination_for_name, recursive=recursive)",
     "                local.pop(root.id, None)",
     "tests.test_g11_census.ToolChecks.test_helper_element_write_invalidates_module_binding",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript default initializer treated as a binding", "census",
     "        if name in _js_parameter_bindings(searchable[param_start:param_end]):",
     "        if re.search(r\"(?<![\\w$])\" + re.escape(name) + r\"(?![\\w$])\", searchable[param_start:param_end]):",
     "tests.test_g11_census.ToolChecks.test_js_default_initializer_reference_is_not_a_parameter_binding",
     'self.assertTrue(self.tools(tree)["tools/live.mjs"]["referenced"])'),
    ("interpreter option operand mistaken for script", "census",
     "        if consume_operand:\n            consume_operand = False\n            continue",
     "        if False:\n            consume_operand = False\n            continue",
     "tests.test_g11_census.ToolChecks.test_interpreter_option_operands_do_not_hide_the_script",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("command alias survives element write", "census",
     "    if isinstance(prior, (list, _StaticTuple, dict)):\n        # Previously evaluated arguments hold the same aggregate objects.",
     "    if False:\n        # Previously evaluated arguments hold the same aggregate objects.",
     "tests.test_g11_census.ToolChecks.test_command_alias_element_write_invalidates_all_aliases",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript URL binding parameter shadow ignored", "census",
     "            if _js_parameter_shadows(visible, call_start, name):\n                continue",
     "            if False:\n                continue",
     "tests.test_g11_census.ToolChecks.test_js_parameter_shadows_url_binding",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("list mutator calls retain stale command provenance", "census",
     "    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)\n            or node.func.attr not in _MUTATING_LIST_METHODS):\n        return None",
     "    if True:\n        return None",
     "tests.test_g11_census.ToolChecks.test_command_mutator_calls_invalidate_aliases_and_forwarding",
     "self.assertEqual({path: item[\"referenced\"] for path, item in self.tools(tree).items()},"),
    ("JavaScript local declaration shadow ignored", "census",
     "        if _js_local_declaration_shadows(visible, call_start, callee):\n            continue",
     "        if False:\n            continue",
     "tests.test_g11_census.ToolChecks.test_js_local_declaration_shadows_named_child_process_import",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("JavaScript URL declaration escapes function scope", "census",
     "                         if declaration[0] <= call_start\n                         and (declaration[2] is None or declaration[2] == call_scope)\n                         and (declaration[3] is None\n                              or declaration[3][0] < call_start < declaration[3][1])]",
     "                         if declaration[0] <= call_start\n                         and (declaration[3] is None\n                              or declaration[3][0] < call_start < declaration[3][1])]",
     "tests.test_g11_census.ToolChecks.test_js_url_binding_does_not_escape_function_scope",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("wrapper parameter alias mutation retains caller provenance", "census",
     "            parameter = aliases.get(name)",
     "            parameter = name if name in parameters else None",
     "tests.test_g11_census.ToolChecks.test_wrapper_parameter_alias_mutation_invalidates_forwarding",
     'self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()}, {'),
    ("reassigned wrapper remains registered", "census",
     "                             and not reassigned and call_order > wrapper_order\n                             and _python_call_signature_binds(wrapper[2], node) else ())",
     "                             and call_order > wrapper_order\n                             and _python_call_signature_binds(wrapper[2], node) else ())",
     "tests.test_g11_census.ToolChecks.test_reassigned_wrapper_binding_is_not_execution_evidence",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript enclosing declaration shadow ignored", "census",
     "    for block in containing:\n        for declaration in direct_pattern.finditer(searchable, block[0] + 1, block[1]):",
     "    for block in containing[:1]:\n        for declaration in direct_pattern.finditer(searchable, block[0] + 1, block[1]):",
     "tests.test_g11_census.ToolChecks.test_js_enclosing_declaration_shadows_named_child_process_import",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("wrapper traversal continues after return", "census",
     "            return isinstance(node, (ast.Return, ast.Raise))",
     "            return False",
     "tests.test_g11_census.ToolChecks.test_wrapper_process_call_after_return_is_unreachable",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("deferred wrapper availability uses source line", "census",
     '            execution_line = (getattr(node, "lineno", 0)\n                              if activation_line is None else activation_line)\n            nodes.append((node, scope, execution_line))\n            if isinstance(target, ast.Name):',
     '            execution_line = getattr(node, "lineno", 0)\n            nodes.append((node, scope, execution_line))\n            if False:',
     "tests.test_g11_census.ToolChecks.test_deferred_caller_sees_wrapper_defined_before_invocation",
     'self.assertTrue(self.tools(tree)["tools/live.py"]["referenced"])'),
    ("later nonforwarding wrapper definition retains registration", "census",
     "                else:\n                    wrappers.pop(key, None)",
     "                elif False:\n                    wrappers.pop(key, None)",
     "tests.test_g11_census.ToolChecks.test_later_nonforwarding_wrapper_definition_replaces_registration",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("list augmented assignment breaks alias identity", "census",
     "                if isinstance(prior, list) and isinstance(value, list):\n                    prior[:] = value\n                    value = prior",
     "                if False:\n                    prior[:] = value\n                    value = prior",
     "tests.test_g11_census.ToolChecks.test_list_augmented_assignment_preserves_alias_identity",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript unreachable child-process call accepted", "census",
     "        activation_points = _js_call_activation_points(visible, call_start)\n        if not activation_points:\n            continue",
     "        activation_points = {call_start}\n        if not activation_points:\n            continue",
     "tests.test_g11_census.ToolChecks.test_js_unreachable_child_process_calls_are_not_references",
     'self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()},'),
    ("deferred caller uses final module binding", "census",
     "                    execute(captured_function, None, execution_line, node)",
     '                    lifecycle.setdefault("__mutation_deferred__", {"tests": [], "setUp": None, "tearDown": None, "setUpClass": None, "tearDownClass": None})["tests"].append(captured_function)',
     "tests.test_g11_census.ToolChecks.test_deferred_caller_uses_module_value_at_invocation",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript uncalled helper chain accepted", "census",
     "            activations.update(reachable(\n                call.start(), seen | {identity}, binding_guards + (guard,)))",
     "            activations.add(call.start())",
     "tests.test_g11_census.ToolChecks.test_js_uncalled_helper_chain_is_not_reachable",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("node test callback root ignored", "census",
     "                    return (\"test-callback\", call_start, open_brace, close_brace, name)",
     "                    return ()",
     "tests.test_g11_census.ToolChecks.test_js_node_test_callback_is_a_reachable_root",
     'self.assertEqual({path: item["referenced"] for path, item in self.tools(tree).items()},'),
    ("none guard branch states combined", "census",
     "                    if names:\n                        marker = ast.Delete(\n",
     "                    if False:\n                        marker = ast.Delete(\n",
     "tests.test_g11_census.ToolChecks.test_none_guard_join_invalidates_alternative_values",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("node test registration reachability ignored", "census",
     "            return reachable(call_start, seen | {identity}, binding_guards)",
     "            return {call_start}",
     "tests.test_g11_census.ToolChecks.test_js_node_test_registration_must_be_reachable",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("node test registrar shadow ignored", "census",
     "                    if (_js_parameter_shadows(searchable, call_start, name)\n                            or _js_local_declaration_shadows(searchable, call_start, name)):\n                        return ()",
     "                    if False:\n                        return ()",
     "tests.test_g11_census.ToolChecks.test_js_node_test_registrar_must_resolve_to_import",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("none guard branch environments merged", "census",
     "                    for alternative in (node.body, node.orelse):",
     "                    for alternative in ([*node.body, *node.orelse],):",
     "tests.test_g11_census.ToolChecks.test_none_guard_branch_calls_use_isolated_bindings",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("nested helper effects deferred", "census",
     "                    execute(captured_function, None, execution_line, node)",
     '                    lifecycle.setdefault("__mutation_deferred__", {"tests": [], "setUp": None, "tearDown": None, "setUpClass": None, "tearDownClass": None})["tests"].append(captured_function)',
     "tests.test_g11_census.ToolChecks.test_nested_helper_effects_precede_caller_continuation",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("global assignment stored in helper local binding", "census",
     "        if scope is None or name in scope_globals.get(scope, set()):\n            return None",
     "        if scope is None:\n            return None",
     "tests.test_g11_census.ToolChecks.test_nested_helper_global_assignment_updates_module_binding",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("command element deletion retains stale sequence", "census",
     "                if not names and isinstance(target, (ast.Subscript, ast.Attribute)):",
     "                if False:",
     "tests.test_g11_census.ToolChecks.test_command_element_delete_invalidates_aggregate_and_aliases",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript helper rebinding ignored", "census",
     "            return ({candidate} if all(binding_active(*guard, candidate)\n                                       for guard in binding_guards) else set())",
     "            return {candidate}",
     "tests.test_g11_census.ToolChecks.test_js_rebound_helper_is_not_reachable",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("wrapper element deletion retains forwarding", "census",
     "            elif isinstance(node, ast.Delete):\n                for target in node.targets:\n                    update_target(target)",
     "            elif False:\n                for target in node.targets:\n                    update_target(target)",
     "tests.test_g11_census.ToolChecks.test_wrapper_element_delete_invalidates_forwarding",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("declared global remains a local static shadow", "census",
     "        static_shadowed = ((visible_shadowed | set(local_imports[\"shadowed\"]))\n                           - locally_imported - global_names\n                           - scope_nonlocals.get(scope, set()))",
     "        static_shadowed = ((visible_shadowed | set(local_imports[\"shadowed\"]))\n                           - locally_imported\n                           - scope_nonlocals.get(scope, set()))",
     "tests.test_g11_census.ToolChecks.test_nested_helper_global_assignment_is_read_in_same_scope",
     'self.assertTrue(self.tools(tree)["tools/live.py"]["referenced"])'),
    ("JavaScript lexical helper shadow ignored", "census",
     "            if _js_helper_call_shadowed(\n                    searchable, call.start(), name, definition_start):\n                continue",
     "            if False:\n                continue",
     "tests.test_g11_census.ToolChecks.test_js_lexically_shadowed_helper_is_not_reachable",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("keyword argument helper effects deferred", "census",
     "                nodes.append((marker, scope, execution_line))\n            for keyword in node.keywords:\n                visit(keyword.value, owner, scope, activation_line)",
     "                nodes.append((marker, scope, execution_line))\n            for keyword in ():\n                visit(keyword.value, owner, scope, activation_line)",
     "tests.test_g11_census.ToolChecks.test_call_argument_helper_effects_precede_process_attribution",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("reassigned Python helper identity ignored", "census",
     "                captured_function = active_functions.get((scope, target.id))",
     "                captured_function = functions.get(target.id)",
     "tests.test_g11_census.ToolChecks.test_reassigned_python_helper_is_not_executed",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript deferred binding checked at definition", "census",
     "            cutoffs = ({call_start} if declaration_scope is not None\n                       else activation_points)",
     "            cutoffs = {call_start}",
     "tests.test_g11_census.ToolChecks.test_js_deferred_helper_uses_activation_time_url_binding",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("earlier argument value reevaluated after later effects", "census",
     "                if value is not missing_argument:\n                    return value",
     "                if False:\n                    return value",
     "tests.test_g11_census.ToolChecks.test_call_preserves_earlier_argument_value_before_later_rebinding",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("assignment target invalidated before RHS helper", "census",
     "            visit(node.value, owner, scope, activation_line)\n"
     "            marker = ast.copy_location(ast.Expr(value=node.value), node.value)",
     "            for target in node.targets:\n"
     "                for name in _target_names(target):\n"
     "                    active_functions.pop(\n"
     "                        (active_function_destination(scope, name), name), None)\n"
     "            visit(node.value, owner, scope, activation_line)\n"
     "            marker = ast.copy_location(ast.Expr(value=node.value), node.value)",
     "tests.test_g11_census.ToolChecks.test_assignment_rhs_helper_executes_before_target_rebinding",
     'self.assertTrue(self.tools(tree)["tools/live.py"]["referenced"])'),
    ("sequence element value reevaluated after later effects", "census",
     "        value = evaluated_elements.get((id(node), index), missing)",
     "        value = missing",
     "tests.test_g11_census.ToolChecks.test_sequence_preserves_earlier_element_value_before_later_effects",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript module block shadow ignored", "census",
     "    searchable, function_scopes = _js_function_scopes(text)\n"
     "    blocks = _js_block_scopes(searchable)\n"
     "    containing = sorted(",
     "    searchable, function_scopes = _js_function_scopes(text)\n"
     "    blocks = [(scope[2], scope[3]) for scope in function_scopes]\n"
     "    containing = sorted(",
     "tests.test_g11_census.ToolChecks.test_js_module_block_declaration_shadows_named_child_process_import",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("captured sequence values lost through concatenation", "census",
     "    missing = object()\n    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):\n        left = evaluated_elements.get",
     "    missing = object()\n    if False and isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):\n        left = evaluated_elements.get",
     "tests.test_g11_census.ToolChecks.test_sequence_capture_survives_list_concatenation",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("Python helper traversal continues after termination", "census",
     "            if statement_terminates(statement):\n                break",
     "            if False:\n                break",
     "tests.test_g11_census.ToolChecks.test_python_helper_stops_after_unconditional_termination",
     "self.assertEqual("),
    ("JavaScript destructured block shadow ignored", "census",
     "            if name in bindings:\n                return True",
     "            if False and name in bindings:\n                return True",
     "tests.test_g11_census.ToolChecks.test_js_destructured_block_declaration_shadows_child_process_import",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("named binary operand reevaluated after later effects", "census",
     '        left = evaluated_elements.get(("binary", id(node), 0), missing)',
     "        left = missing",
     "tests.test_g11_census.ToolChecks.test_concatenation_preserves_named_operand_before_later_effects",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("selected branch traversal continues after termination", "census",
     "            visit_statement_sequence(branch, owner, scope, activation_line)\n            return",
     "            for statement in branch:\n                visit(statement, owner, scope, activation_line)\n            return",
     "tests.test_g11_census.ToolChecks.test_selected_python_branch_stops_after_termination",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript ASI declaration shadow ignored", "census",
     "            if simple is not None and simple.group(1) == name:\n                if start + simple.start(1) == permitted_declaration:\n                    continue\n                return True",
     "            if False:\n                if start + simple.start(1) == permitted_declaration:\n                    continue\n                return True",
     "tests.test_g11_census.ToolChecks.test_js_asi_block_declaration_shadows_child_process_import",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("repeated helper invocation suppressed", "census",
     "            active_calls.remove(identity)",
     "            pass",
     "tests.test_g11_census.ToolChecks.test_repeated_helper_invocations_apply_each_mutation",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("global helper reassignment invalidates local identity", "census",
     "        if name in function_scope_globals.get(scope, set()):\n            return None",
     "        if False:\n            return None",
     "tests.test_g11_census.ToolChecks.test_global_assignment_invalidates_module_helper_identity",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("Boolean short-circuit reachability ignored", "census",
     "        if isinstance(node, ast.BoolOp):\n            # Boolean operands execute left-to-right",
     "        if False:\n            # Boolean operands execute left-to-right",
     "tests.test_g11_census.ToolChecks.test_boolean_short_circuit_skips_unreachable_loader",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("conditional expression reachability ignored", "census",
     "        if isinstance(node, ast.IfExp):\n            # Conditional expressions evaluate the condition first",
     "        if False:\n            # Conditional expressions evaluate the condition first",
     "tests.test_g11_census.ToolChecks.test_conditional_expression_skips_unreachable_loader",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("async helper creation treated as execution", "census",
     "            if (owner is None and isinstance(node, ast.FunctionDef)\n                    and not _function_is_generator(node)):",
     "            if (owner is None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))\n                    and not _function_is_generator(node)):",
     "tests.test_g11_census.ToolChecks.test_unawaited_async_helper_body_is_not_executed",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("async forwarding wrapper treated as execution", "census",
     "            if isinstance(statement, ast.FunctionDef):\n                key = (containing_scope, statement.name)",
     "            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):\n                key = (containing_scope, statement.name)",
     "tests.test_g11_census.ToolChecks.test_async_forwarding_wrapper_is_not_executed",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("generator helper creation treated as execution", "census",
     "            if (owner is None and isinstance(node, ast.FunctionDef)\n                    and not _function_is_generator(node)):",
     "            if (owner is None and isinstance(node, ast.FunctionDef)):",
     "tests.test_g11_census.ToolChecks.test_uniterated_generator_helper_body_is_not_executed",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("comprehension body treated as unconditional", "census",
     "        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,\n                             ast.GeneratorExp)):",
     "        if False:",
     "tests.test_g11_census.ToolChecks.test_comprehension_bodies_need_established_iteration",
     "self.assertEqual("),
    ("chained comparison short circuit ignored", "census",
     "        if isinstance(node, ast.Compare):\n            # Chained comparisons evaluate one comparator at a time",
     "        if False:\n            # Chained comparisons evaluate one comparator at a time",
     "tests.test_g11_census.ToolChecks.test_chained_comparison_stops_after_literal_false_prefix",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript URL lexical block ignored", "census",
     "                         and (declaration[3] is None\n                              or declaration[3][0] < call_start < declaration[3][1])]",
     "                         and True]",
     "tests.test_g11_census.ToolChecks.test_js_url_binding_does_not_escape_lexical_block",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("generator method creation treated as execution", "census",
     "                if (isinstance(captured_method, ast.FunctionDef)\n                        and not _function_is_generator(captured_method)\n                        and _python_call_signature_binds(\n                            captured_method, node, implicit_positional=1)):",
     "                if (isinstance(captured_method, ast.FunctionDef)\n                        and _python_call_signature_binds(\n                            captured_method, node, implicit_positional=1)):",
     "tests.test_g11_census.ToolChecks.test_uniterated_generator_method_body_is_not_executed",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("comprehension target shadow ignored", "census",
     "            comprehension_scope = id(node)",
     "            comprehension_scope = scope",
     "tests.test_g11_census.ToolChecks.test_comprehension_target_shadows_outer_static_binding",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("forwarding wrapper Boolean reachability ignored", "census",
     "            if isinstance(node, ast.BoolOp):\n                for index, value in enumerate(node.values):",
     "            if False:\n                for index, value in enumerate(node.values):",
     "tests.test_g11_census.ToolChecks.test_forwarding_wrapper_respects_boolean_short_circuiting",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("forwarding wrapper conditional reachability ignored", "census",
     "            if isinstance(node, ast.IfExp):\n                visit(node.test)\n                truth = _literal_bool(node.test)",
     "            if False:\n                visit(node.test)\n                truth = _literal_bool(node.test)",
     "tests.test_g11_census.ToolChecks.test_forwarding_wrapper_respects_conditional_expression_reachability",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("global wrapper reassignment kept in helper scope", "census",
     "                destination_scope = wrapper_destination_scope(scope, name)",
     "                destination_scope = scope",
     "tests.test_g11_census.ToolChecks.test_global_helper_assignment_invalidates_forwarding_wrapper",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("wrapper reassignment uses lexical line instead of execution order", "census",
     "                    wrapper_reassignments.setdefault(key, []).append(\n                        (execution_line, node.lineno))",
     "                    wrapper_reassignments.setdefault(key, []).append(\n                        (node.lineno, node.lineno))",
     "tests.test_g11_census.ToolChecks.test_global_helper_assignment_invalidates_forwarding_wrapper",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("lambda body traversed at construction", "census",
     "        if isinstance(node, ast.Lambda):\n            # Lambda construction evaluates defaults, but the body remains",
     "        if False:\n            # Lambda construction evaluates defaults, but the body remains",
     "tests.test_g11_census.ToolChecks.test_lambda_creation_executes_defaults_not_body",
     'self.assertEqual(\n            {path: item["referenced"] for path, item in self.tools(tree).items()},'),
    ("nonlocal active function destination ignored", "census",
     "        if name in function_scope_nonlocals.get(scope, set()):\n            parent = function_scope_parents.get(scope)",
     "        if False:\n            parent = function_scope_parents.get(scope)",
     "tests.test_g11_census.ToolChecks.test_nonlocal_assignment_invalidates_enclosing_helper_identity",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("nonlocal wrapper destination ignored", "census",
     "        if name in scope_nonlocals.get(scope, set()):\n            parent = scope_parents.get(scope)",
     "        if False:\n            parent = scope_parents.get(scope)",
     "tests.test_g11_census.ToolChecks.test_nonlocal_helper_assignment_invalidates_enclosing_forwarding_wrapper",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("wrapper keyword argument effects deferred", "census",
     "                for keyword in node.keywords:\n                    visit(keyword.value)\n                mutated_root = _python_mutated_root_name(node)",
     "                for keyword in ():\n                    visit(keyword.value)\n                mutated_root = _python_mutated_root_name(node)",
     "tests.test_g11_census.ToolChecks.test_forwarding_wrapper_applies_argument_effects_before_forwarding",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("forwarding wrapper chained-comparison reachability ignored", "census",
     "            if isinstance(node, ast.Compare):\n                visit(node.left)\n                left = node.left",
     "            if False:\n                visit(node.left)\n                left = node.left",
     "tests.test_g11_census.ToolChecks.test_forwarding_wrapper_respects_chained_comparison_short_circuiting",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("exception handler reachability ignored", "census",
     "        if isinstance(node, (ast.Try, getattr(ast, \"TryStar\", ast.Try))):\n            # The try body executes, and a finally body executes regardless of",
     "        if False:\n            # The try body executes, and a finally body executes regardless of",
     "tests.test_g11_census.ToolChecks.test_try_handlers_require_proven_reachability_but_finally_executes",
     'self.assertEqual(\n            {\n                "tools/body-live.py": True,'),
    ("skipped exception branch effects ignored", "census",
     "            possible_effects = set().union(\n                *(branch_possible_effects(handler.body, scope)\n                  for handler in node.handlers),\n                branch_possible_effects(node.orelse, scope),\n                {handler.name for handler in node.handlers if handler.name})\n            if possible_effects:\n                marker = ast.Delete(",
     "            possible_effects = set().union(\n                *(branch_possible_effects(handler.body, scope)\n                  for handler in node.handlers),\n                branch_possible_effects(node.orelse, scope),\n                {handler.name for handler in node.handlers if handler.name})\n            if False:\n                marker = ast.Delete(",
     "tests.test_g11_census.ToolChecks.test_unproven_exception_branches_invalidate_downstream_bindings",
     '{"tools/else-unused.py": False, "tools/handler-unused.py": False},'),
    ("wrapper exception-handler reachability ignored", "census",
     "            if isinstance(node, (ast.Try, getattr(ast, \"TryStar\", ast.Try))):\n                body_terminates = visit_body(node.body)",
     "            if False:\n                body_terminates = visit_body(node.body)",
     "tests.test_g11_census.ToolChecks.test_forwarding_wrapper_try_boundary_and_finally_execution",
     '{"tools/finally-live.py": True, "tools/handler-unused.py": False},'),
    ("forwarding wrapper comprehension reachability ignored", "census",
     "            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,\n                                 ast.GeneratorExp)):\n                if not node.generators:",
     "            if False:\n                if not node.generators:",
     "tests.test_g11_census.ToolChecks.test_forwarding_wrapper_requires_established_comprehension_iteration",
     '"tools/generator-unused.py": False,'),
    ("finally termination not propagated", "census",
     "            return (statement_sequence_terminates(statement.finalbody)\n                    or (not statement.handlers",
     "            return (False\n                    or (not statement.handlers",
     "tests.test_g11_census.ToolChecks.test_finally_termination_stops_following_helper_statements",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript URL destructured lexical shadow ignored", "census",
     "            if _js_local_declaration_shadows(\n                    visible, call_start, name, declaration_end):\n                continue",
     "            if False:\n                continue",
     "tests.test_g11_census.ToolChecks.test_js_destructured_lexical_binding_shadows_outer_url_argument",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("nonlocal static assignment kept in helper scope", "census",
     "        destination_for_name = lambda name: binding_destination(scope, name)",
     "        destination_for_name = lambda name: (\n            module_bindings if scope is None or name in global_names else local)",
     "tests.test_g11_census.ToolChecks.test_nonlocal_assignment_updates_enclosing_static_command_binding",
     '{"tools/live.py": True, "tools/unused.py": False},'),
    ("unknown branch effects ignored", "census",
     "                    if not _is_none_guard(node.test):\n                        if names:",
     "                    if not _is_none_guard(node.test):\n                        if False:",
     "tests.test_g11_census.ToolChecks.test_unknown_branch_effects_invalidate_downstream_bindings",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("handler-free try body termination not propagated", "census",
     "                    or (not statement.handlers\n                        and statement_sequence_terminates(statement.body)))",
     "                    or (False and not statement.handlers\n                        and statement_sequence_terminates(statement.body)))",
     "tests.test_g11_census.ToolChecks.test_handler_free_try_body_termination_stops_following_helper_statements",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("decorated Python helper retains original body", "census",
     "                if not node.decorator_list:\n                    active_functions[key] = node",
     "                if True:\n                    active_functions[key] = node",
     "tests.test_g11_census.ToolChecks.test_decorated_python_helper_does_not_retain_original_body",
     'self.assertFalse(report["tools/unused.py"]["referenced"])'),
    ("non-string executable reaches interpreter membership", "census",
     "    if not isinstance(first, str):\n        return set()\n    if first not in interpreters:",
     "    if not isinstance(first, str):\n        return _tool_values(first, tools)\n    if first not in interpreters:",
     "tests.test_g11_census.ToolChecks.test_non_string_executable_value_is_left_unresolved",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript destructured helper shadow ignored", "census",
     "    if _js_local_declaration_shadows(\n            text, position, name, permitted_declaration=definition_start):\n        return True",
     "    if False:\n        return True",
     "tests.test_g11_census.ToolChecks.test_js_destructured_helper_binding_shadows_module_helper",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("JavaScript composite argument identifiers credited", "census",
     "        identifiers = _js_direct_child_argument_identifiers(arguments)",
     "        identifiers = _js_expression_identifiers(arguments)",
     "tests.test_g11_census.ToolChecks.test_js_url_binding_must_reach_child_process_arguments",
     '{"tools/live.mjs": True, "tools/unused.mjs": False},'),
    ("JavaScript file URL converter identity ignored", "census",
     "        if converter:\n            import_end = url_converters.get(converter)",
     "        if False:\n            import_end = url_converters.get(converter)",
     "tests.test_g11_census.ToolChecks.test_js_file_url_to_path_requires_unshadowed_node_url_import",
     '{"tools/live.mjs": True, "tools/unused.mjs": False},'),
    ("possible-effect marker bypasses active helper invalidation", "census",
     "                visit(marker, owner, scope, activation_line)\n            visit_statement_sequence(node.finalbody, owner, scope, activation_line)",
     "                execution_line = (marker.lineno if activation_line is None else activation_line)\n                nodes.append((marker, scope, execution_line))\n            visit_statement_sequence(node.finalbody, owner, scope, activation_line)",
     "tests.test_g11_census.ToolChecks.test_exception_boundaries_invalidate_callable_identities",
     '{"tools/helper-unused.py": False, "tools/wrapper-unused.py": False},'),
    ("possible deletion ignored for wrapper identity", "census",
     "        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign,\n                                 ast.Delete, ast.NamedExpr)):",
     "        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign,\n                                 ast.NamedExpr)):",
     "tests.test_g11_census.ToolChecks.test_exception_boundaries_invalidate_callable_identities",
     '{"tools/helper-unused.py": False, "tools/wrapper-unused.py": False},'),
    ("exception target binding effect ignored", "census",
     "                branch_possible_effects(node.orelse, scope),\n                {handler.name for handler in node.handlers if handler.name})",
     "                branch_possible_effects(node.orelse, scope),\n                set())",
     "tests.test_g11_census.ToolChecks.test_exception_target_names_invalidate_static_bindings",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("noniterable literal accepted as comprehension source", "census",
     "    if type(value) not in {list, tuple, set, dict, str, bytes}:\n        return False",
     "    if False:\n        return False",
     "tests.test_g11_census.ToolChecks.test_comprehension_bodies_require_supported_iterables",
     '{"tools/direct-unused.py": False, "tools/wrapper-unused.py": False},'),
    ("unsupported interpreter option skipped", "census",
     "        if parsing_options and token.startswith((\"-\", \"+\")):\n            return set()",
     "        if parsing_options and token.startswith((\"-\", \"+\")):\n            continue",
     "tests.test_g11_census.ToolChecks.test_interpreter_attached_and_exit_only_options_do_not_select_scripts",
     '{"tools/attached-unused.py": False, "tools/live.py": True,'),
    ("assignment expression write ignored", "census",
     "        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):\n            augmented_prior = None",
     "        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):\n            augmented_prior = None",
     "tests.test_g11_census.ToolChecks.test_assignment_expressions_update_static_bindings",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("nested aggregate mutation retains reachable alias provenance", "census",
     "                    destination_for_name, recursive=recursive)",
     "                    destination_for_name, recursive=False)",
     "tests.test_g11_census.ToolChecks.test_nested_aggregate_writes_invalidate_reachable_aliases",
     '{"tools/live.py": True, "tools/unused.py": False},'),
    ("decorated Python forwarding wrapper registered", "census",
     "                if _function_is_generator(statement) or statement.decorator_list:",
     "                if _function_is_generator(statement) or False:",
     "tests.test_g11_census.ToolChecks.test_decorated_python_wrapper_is_not_registered",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript URL options argument credited", "census",
     "            identifiers.update(part for part in parts if identifier.fullmatch(part))\n    return identifiers",
     "            identifiers.update(part for part in parts if identifier.fullmatch(part))\n    identifiers.update(argument for argument in arguments[2:] if identifier.fullmatch(argument))\n    return identifiers",
     "tests.test_g11_census.ToolChecks.test_js_url_binding_requires_command_argument_position",
     '{"tools/live.mjs": True, "tools/unused.mjs": False},'),
    ("reachable events resorted by source line", "census",
     "    reachable_nodes = _reachable_python_nodes(tree, mode)",
     "    reachable_nodes = sorted(_reachable_python_nodes(tree, mode),\n"
     "                             key=lambda item: item[2])",
     "tests.test_g11_census.ToolChecks.test_multiline_calls_preserve_argument_effect_order",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("direct loop effects ignored", "census",
     "            possible_effects = branch_possible_effects([node], scope)",
     "            possible_effects = set()",
     "tests.test_g11_census.ToolChecks.test_possible_loop_writes_invalidate_static_bindings",
     '{"tools/body-unused.py": False, "tools/else-unused.py": False,'),
    ("forwarding wrapper loop effects ignored", "census",
     "            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):\n                visit(node.iter if isinstance(node, (ast.For, ast.AsyncFor)) else node.test)\n                for name in _branch_assigned_names([node]):",
     "            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):\n                visit(node.iter if isinstance(node, (ast.For, ast.AsyncFor)) else node.test)\n                for name in ():",
     "tests.test_g11_census.ToolChecks.test_possible_loop_writes_invalidate_static_bindings",
     '{"tools/body-unused.py": False, "tools/else-unused.py": False,'),
    ("subprocess executable override ignored", "census",
     "    for keyword in node.keywords:\n        if keyword.arg is None:\n            return False",
     "    for keyword in ():\n        if keyword.arg is None:\n            return False",
     "tests.test_g11_census.ToolChecks.test_subprocess_executable_override_blocks_command_credit",
     '{"tools/direct-unused.py": False, "tools/dynamic-unused.py": False,'),
    ("class body bindings leak into enclosing scope", "census",
     "            class_scope = id(node)",
     "            class_scope = scope",
     "tests.test_g11_census.ToolChecks.test_class_body_bindings_do_not_replace_module_bindings",
     '{"tools/class-live.py": True, "tools/method-live.py": True,'),
    ("class runtime shadows ignored", "census",
     "        static_shadowed = ((visible_shadowed | set(local_imports[\"shadowed\"]))\n"
     "                           - locally_imported - global_names\n"
     "                           - scope_nonlocals.get(scope, set()))",
     "        static_shadowed = (visible_shadowed\n"
     "                           - locally_imported - global_names\n"
     "                           - scope_nonlocals.get(scope, set()))",
     "tests.test_g11_census.ToolChecks.test_class_body_bindings_do_not_replace_module_bindings",
     '{"tools/class-live.py": True, "tools/method-live.py": True,'),
    ("subprocess shell mode ignored", "census",
     "        if (keyword.arg == \"shell\"\n                and not (isinstance(keyword.value, ast.Constant)",
     "        if (False\n                and not (isinstance(keyword.value, ast.Constant)",
     "tests.test_g11_census.ToolChecks.test_subprocess_shell_mode_blocks_command_credit",
     '{"tools/dynamic-unused.py": False, "tools/kwargs-unused.py": False,'),
    ("subprocess positional options accepted", "census",
     "    if len(node.args) != 1:\n        return False",
     "    if False:\n        return False",
     "tests.test_g11_census.ToolChecks.test_subprocess_shell_mode_blocks_command_credit",
     '{"tools/dynamic-unused.py": False, "tools/kwargs-unused.py": False,'),
    ("JavaScript URL constructor identity ignored", "census",
     "        if not _js_builtin_name_unshadowed(\n                visible, match.start(\"constructor\"), match.group(\"constructor\")):\n            continue",
     "        if False:\n            continue",
     "tests.test_g11_census.ToolChecks.test_js_url_constructor_requires_unshadowed_builtin_identity",
     '{"tools/block-unused.mjs": False, "tools/import-unused.mjs": False,'),
    ("JavaScript URL property write retains stale provenance", "census",
     "    property_root = (root +\n                     r\"\\s*(?:(?:\\.\\s*[A-Za-z_$][\\w$]*)|(?:\\[[^\\]\\n]+\\]))+\")",
     "    return False\n    property_root = (root +\n                     r\"\\s*(?:(?:\\.\\s*[A-Za-z_$][\\w$]*)|(?:\\[[^\\]\\n]+\\]))+\")",
     "tests.test_g11_census.ToolChecks.test_js_url_property_mutation_invalidates_path_provenance",
     '{"tools/live.mjs": True, "tools/unused.mjs": False},'),
    ("JavaScript expression short-circuit reachability ignored", "census",
     "        if not _js_short_circuit_call_reachable(searchable, candidate):\n            return set()",
     "        if False:\n            return set()",
     "tests.test_g11_census.ToolChecks.test_js_unreachable_child_process_calls_are_not_references",
     '{\n            "tools/branch.mjs": False, "tools/falseor-live.mjs": True,'),
    ("helper invocation binding frame reused", "census",
     "            scope_bindings[scope] = {}",
     "            scope_bindings.setdefault(scope, {})",
     "tests.test_g11_census.ToolChecks.test_helper_invocations_use_fresh_local_binding_environments",
     '{"tools/helper-unused.py": False, "tools/import-unused.py": False,'),
    ("helper invocation import frame reused", "census",
     "            scope_imports[scope] = _empty_import_bindings()",
     "            scope_imports.setdefault(scope, _empty_import_bindings())",
     "tests.test_g11_census.ToolChecks.test_helper_invocations_use_fresh_local_binding_environments",
     '{"tools/helper-unused.py": False, "tools/import-unused.py": False,'),
    ("helper invocation nested definitions reused", "census",
     "            for key in [key for key in active_functions if key[0] == id(function)]:",
     "            for key in []:",
     "tests.test_g11_census.ToolChecks.test_helper_invocations_use_fresh_local_binding_environments",
     '{"tools/helper-unused.py": False, "tools/import-unused.py": False,'),
    ("aggregate str conversion retains argv semantics", "census",
     "        return value if isinstance(value, str) else None",
     "        return value",
     "tests.test_g11_census.ToolChecks.test_str_conversion_does_not_preserve_aggregate_command_semantics",
     '{\n            "tools/live.py": True, "tools/unused.py": False,'),
    ("forwarding wrapper unknown branch effects ignored", "census",
     "                        for name in _branch_assigned_names([node]):\n                            invalidate(name)",
     "                        for name in ():\n                            invalidate(name)",
     "tests.test_g11_census.ToolChecks.test_unknown_wrapper_branch_effects_invalidate_forwarding",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript URL initializer suffix accepted", "census",
     "        if not _js_supported_initializer_complete(visible, match.end()):\n            continue",
     "        if False:\n            continue",
     "tests.test_g11_census.ToolChecks.test_js_url_binding_requires_complete_supported_initializer",
     '{"tools/continued-unused.mjs": False, "tools/live.mjs": True,'),
    ("JavaScript ternary arm reachability ignored", "census",
     "        if not _js_ternary_call_reachable(searchable, candidate):\n            return set()",
     "        if False:\n            return set()",
     "tests.test_g11_census.ToolChecks.test_js_conditional_expression_arms_require_proven_selection",
     '{"tools/false-else-live.mjs": True,'),
    ("skipped Boolean operand effects ignored", "census",
     "                    possible_effects = branch_possible_effects(\n                        node.values[index + 1:], scope)",
     "                    possible_effects = set()",
     "tests.test_g11_census.ToolChecks.test_unknown_boolean_operand_effects_invalidate_downstream_bindings",
     '{"tools/direct-unused.py": False, "tools/live.py": True,'),
    ("forwarding wrapper skipped Boolean operand effects ignored", "census",
     "                        for name in _branch_assigned_names(node.values[index + 1:]):\n                            invalidate(name)",
     "                        for name in ():\n                            invalidate(name)",
     "tests.test_g11_census.ToolChecks.test_unknown_boolean_operand_effects_invalidate_downstream_bindings",
     '"tools/wrapper-unused.py": False},'),
    ("nested helper free name falls back to module binding", "census",
     "                for parent_scope in reversed(enclosing_scopes):",
     "                for parent_scope in ():",
     "tests.test_g11_census.ToolChecks.test_nested_helper_reads_nearest_enclosing_static_binding",
     '{"tools/live.py": True, "tools/unused.py": False},'),
    ("Python helper signature binding ignored", "census",
     "    return required_positional <= bound and required_keyword_only <= bound",
     "    return True",
     "tests.test_g11_census.ToolChecks.test_python_helper_requires_bindable_arguments",
     '{"tools/direct-unused.py": False, "tools/live.py": True,'),
    ("unknown conditional expression effects ignored", "census",
     "                possible_effects = branch_possible_effects(\n                    (node.body, node.orelse), scope)",
     "                possible_effects = set()",
     "tests.test_g11_census.ToolChecks.test_unknown_conditional_expression_effects_invalidate_provenance",
     '{"tools/unused.py": False, "tools/wrapper-unused.py": False}'),
    ("forwarding wrapper unknown conditional expression effects ignored", "census",
     "                    for name in _branch_assigned_names((node.body, node.orelse)):\n                        invalidate(name)",
     "                    for name in ():\n                        invalidate(name)",
     "tests.test_g11_census.ToolChecks.test_unknown_conditional_expression_effects_invalidate_provenance",
     '"tools/wrapper-unused.py": False}'),
    ("forwarding wrapper call signature ignored", "census",
     "                             and _python_call_signature_binds(wrapper[2], node) else ())",
     "                             else ())",
     "tests.test_g11_census.ToolChecks.test_forwarding_wrapper_requires_valid_call_binding",
     '{"tools/live.py": True, "tools/unused.py": False}'),
    ("assignment expression leaves import identity stale", "census",
     "    elif isinstance(node, (ast.AugAssign, ast.NamedExpr)):\n        targets = [node.target]",
     "    elif isinstance(node, ast.AugAssign):\n        targets = [node.target]",
     "tests.test_g11_census.ToolChecks.test_named_expression_invalidates_local_import_identity",
     '{"tools/live.py": True, "tools/unused.py": False}'),
    ("tuple augmented assignment mutates aliases", "census",
     "    return _StaticTuple(values) if isinstance(node, ast.Tuple) else values\n\n\n_MUTATING_LIST_METHODS",
     "    return values\n\n\n_MUTATING_LIST_METHODS",
     "tests.test_g11_census.ToolChecks.test_tuple_augmented_assignment_does_not_mutate_alias",
     '{"tools/live.py": True, "tools/unused.py": False}'),
    ("JavaScript unbraced literal-if reachability ignored", "census",
     "            if consequent_start <= candidate < consequent_end:\n"
     "                if truth is not True:\n"
     "                    return True\n"
     "                continue",
     "            if consequent_start <= candidate < consequent_end:\n                return False",
     "tests.test_g11_census.ToolChecks.test_js_unbraced_literal_false_branch_is_not_activated",
     '{"tools/p1.mjs": False, "tools/p2.mjs": False,'),
    ("JavaScript newline call continuation treated as ASI", "census",
     '    if searchable[right] in "([.?+-*/%&|^:,":\n        return True',
     '    if searchable[right] in "[.?+-*/%&|^:,":\n        return True',
     "tests.test_g11_census.ToolChecks.test_js_unbraced_literal_false_branch_is_not_activated",
     '{"tools/p1.mjs": False, "tools/p2.mjs": False,'),
    ("JavaScript nested-if else not consumed", "census",
     '        if re.match(r"else\\b", searchable[alternate_start:]):\n            return _js_bounded_statement_end(searchable, alternate_start + 4)',
     '        if False:\n            return _js_bounded_statement_end(searchable, alternate_start + 4)',
     "tests.test_g11_census.ToolChecks.test_js_unbraced_literal_false_branch_is_not_activated",
     '"tools/p3.mjs": False, "tools/p4.mjs": False,'),
    ("JavaScript concise arrow body treated as module execution", "census",
     "            if body_start < body_end:\n                concise_scopes.append(\n                    (param_start, param_end, body_start - 1, body_end))",
     "            if False:\n                concise_scopes.append(\n                    (param_start, param_end, body_start - 1, body_end))",
     "tests.test_g11_census.ToolChecks.test_js_concise_arrow_body_is_deferred_until_invocation",
     '"tools/helper-unused.mjs": False,'),
    ("JavaScript concise node-test callback not activated", "census",
     '                    r"(?![\\w$])\\s*\\([^{};]*=>\\s*(?:\\{)?$"',
     '                    r"(?![\\w$])\\s*\\([^{};]*=>\\s*\\{$"',
     "tests.test_g11_census.ToolChecks.test_js_concise_arrow_body_is_deferred_until_invocation",
     '"tools/callback-live.mjs": True,'),
    ("JavaScript spaced block arrow registered as concise", "census",
     '        if body_start < len(searchable) and searchable[body_start] == "{":',
     '        if (body_start < len(searchable) and searchable[body_start] == "{"\n                and arrow.end() == body_start):',
     "tests.test_g11_census.ToolChecks.test_js_spaced_block_arrow_has_one_block_scope",
     "self.assertEqual(scopes, [(16, 16, 20, 35)])"),
    ("annotation-only assignment invalidates import identity", "census",
     "            elif isinstance(node, ast.AnnAssign):\n                return",
     "            elif isinstance(node, ast.AnnAssign):\n                node.value = ast.copy_location(ast.Constant(value=None), node)",
     "tests.test_g11_census.ToolChecks.test_annotation_only_statement_preserves_local_import_identity",
     '{"tools/live.py": True, "tools/unused.py": False}'),
    ("helper argument alias binding ignored", "census",
     '            for formal, call_id, argument_key in getattr(\n                    node, "_g11_argument_bindings", ()):',
     "            for formal, call_id, argument_key in ():",
     "tests.test_g11_census.ToolChecks.test_helper_argument_alias_mutation_updates_caller_aggregate",
     '"tools/keyword-unused.py": False,'),
    ("unittest setup ordering ignored", "census",
     '            if roles["setUp"] is not None:\n                activation_line += 1\n                execute(roles["setUp"], owner, activation_line)\n            activation_line += 1\n            execute(test, owner, activation_line)',
     '            activation_line += 1\n            execute(test, owner, activation_line)\n            if roles["setUp"] is not None:\n                activation_line += 1\n                execute(roles["setUp"], owner, activation_line)',
     "tests.test_g11_census.ToolChecks.test_unittest_setup_effects_precede_each_test_method",
     '"tools/pre.py": False,'),
    ("quoted JavaScript URL declaration accepted", "census",
     '        if (searchable[match.start("kind"):match.end("kind")]\n                != match.group("kind")):\n            continue',
     "        if False:\n            continue",
     "tests.test_g11_census.ToolChecks.test_js_quoted_url_declaration_is_not_a_binding",
     '{"tools/live.mjs": True, "tools/unused.mjs": False}'),
    ("JavaScript URL alias escape retains stale provenance", "census",
     "        if re.search(alias_assignment, segment):\n            return True",
     "        if False:\n            return True",
     "tests.test_g11_census.ToolChecks.test_js_url_alias_mutation_invalidates_original_provenance",
     '{"tools/live.mjs": True, "tools/unused.mjs": False}'),
    ("default aggregate argument binding ignored", "census",
     '            for formal, function_id, default_name in getattr(\n                    node, "_g11_default_bindings", ()):',
     "            for formal, function_id, default_name in ():",
     "tests.test_g11_census.ToolChecks.test_helper_argument_alias_mutation_updates_caller_aggregate",
     '"tools/default-unused.py": False,'),
    ("variadic positional aggregate binding ignored", "census",
     '            if vararg is not None:\n                formal, call_id, argument_keys = vararg',
     '            if False:\n                formal, call_id, argument_keys = vararg',
     "tests.test_g11_census.ToolChecks.test_helper_argument_alias_mutation_updates_caller_aggregate",
     '"tools/vararg-unused.py": False}'),
    ("variadic keyword aggregate binding ignored", "census",
     '            if kwarg is not None:\n                formal, call_id, entries = kwarg',
     '            if False:\n                formal, call_id, entries = kwarg',
     "tests.test_g11_census.ToolChecks.test_helper_argument_alias_mutation_updates_caller_aggregate",
     '"tools/kwarg-unused.py": False,'),
    ("assignment target effects ignored", "census",
     "                visit_assignment_target_effects(\n"
     "                    target, owner, scope, activation_line)\n"
     "                for name in _target_names(target):",
     "                if False:\n"
     "                    visit_assignment_target_effects(\n"
     "                        target, owner, scope, activation_line)\n"
     "                for name in _target_names(target):",
     "tests.test_g11_census.ToolChecks.test_assignment_target_effects_follow_rhs_and_precede_later_statements",
     '{"tools/live.py": True, "tools/unused.py": False},'),
    ("unknown chained-comparison suffix effects ignored", "census",
     "                    possible_effects = branch_possible_effects(\n                        node.comparators[index + 1:], scope)",
     "                    possible_effects = set()",
     "tests.test_g11_census.ToolChecks.test_unknown_chained_comparison_suffix_effect_invalidates_provenance",
     '{"tools/direct-unused.py": False, "tools/wrapper-unused.py": False},'),
    ("forwarding wrapper unknown chained-comparison suffix effects ignored", "census",
     "                        for name in _branch_assigned_names(\n                                node.comparators[index + 1:]):\n                            invalidate(name)",
     "                        for name in ():\n                            invalidate(name)",
     "tests.test_g11_census.ToolChecks.test_unknown_chained_comparison_suffix_effect_invalidates_provenance",
     '{"tools/direct-unused.py": False, "tools/wrapper-unused.py": False},'),
    ("unresolved eager-comprehension effects ignored", "census",
     "            possible_effects = branch_possible_effects(\n                _comprehension_possible_effect_expressions(node), scope)",
     "            possible_effects = set()",
     "tests.test_g11_census.ToolChecks.test_unresolved_eager_comprehension_effects_invalidate_provenance",
     '{"tools/direct-unused.py": False, "tools/wrapper-unused.py": False},'),
    ("forwarding wrapper unresolved eager-comprehension effects ignored", "census",
     "                        for name in _branch_assigned_names(\n                                _comprehension_possible_effect_expressions(node)):\n                            invalidate(name)",
     "                        for name in ():\n                            invalidate(name)",
     "tests.test_g11_census.ToolChecks.test_unresolved_eager_comprehension_effects_invalidate_provenance",
     '{"tools/direct-unused.py": False, "tools/wrapper-unused.py": False},'),
    ("uninvoked JavaScript methods treated as module execution", "census",
     "    for match in method.finditer(method_searchable):\n        if match.group(\"name\") in control_names:",
     "    for match in ():\n        if match.group(\"name\") in control_names:",
     "tests.test_g11_census.ToolChecks.test_js_uninvoked_method_bodies_are_not_module_execution",
     '{"tools/class-unused.mjs": False, "tools/live.mjs": True,'),
    ("literal-true JavaScript else arm treated as reachable", "census",
     '            if (alternate_end is not None\n'
     '                    and alternate_body_start <= candidate < alternate_end):\n'
     '                if truth is not False:\n'
     '                    return True\n'
     '                continue',
     '            if (alternate_end is not None\n                    and alternate_body_start <= candidate < alternate_end):\n                return False',
     "tests.test_g11_census.ToolChecks.test_js_literal_if_excludes_both_unselected_braced_arms",
     '"tools/true-else-unused.mjs": False,'),
    ("unknown branch helper effects ignored", "census",
     "            if call_effects is not None:\n                names.update(call_effects(node))\n            self.generic_visit(node)\n\n        def _target_root",
     "            if False:\n                names.update(call_effects(node))\n            self.generic_visit(node)\n\n        def _target_root",
     "tests.test_g11_census.ToolChecks.test_unknown_branch_helper_effects_invalidate_downstream_bindings",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("quoted JavaScript child-process import trusted", "census",
     "    for match in named_import.finditer(visible):\n        if searchable[match.start():match.start() + 6] != \"import\":\n            continue",
     "    for match in named_import.finditer(visible):\n        if False:\n            continue",
     "tests.test_g11_census.ToolChecks.test_js_quoted_child_process_import_is_not_trusted",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("dead JavaScript URL initializer trusted", "census",
     '        if not _js_call_activation_points(visible, match.start("kind")):\n            continue',
     '        if False:\n            continue',
     "tests.test_g11_census.ToolChecks.test_js_dead_url_initializer_does_not_create_provenance",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("unknown branch helper parameter mutation ignored", "census",
     "            for mutated_name in mutations & local_only:",
     "            for mutated_name in ():",
     "tests.test_g11_census.ToolChecks.test_unknown_branch_helper_parameter_effects_invalidate_caller_binding",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("nested selected JavaScript branch stops reachability checks", "census",
     "            if consequent_start <= candidate < consequent_end:\n"
     "                if truth is not True:\n"
     "                    return True\n"
     "                continue",
     "            if consequent_start <= candidate < consequent_end:\n"
     "                if truth is not True:\n"
     "                    return True\n"
     "                return False",
     "tests.test_g11_census.ToolChecks.test_js_nested_literal_if_checks_every_enclosing_branch",
     '"tools/nested-unused.mjs": False'),
    ("unproven JavaScript loop body activated", "census",
     "        if in_unproven_loop(candidate):\n            return set()",
     "        if False:\n            return set()",
     "tests.test_g11_census.ToolChecks.test_js_loop_body_requires_proven_execution",
     '"tools/false-unused.mjs": False'),
    ("terminated JavaScript helper tail activated", "census",
     "        if after_unconditional_termination(candidate):\n            return set()",
     "        if False:\n            return set()",
     "tests.test_g11_census.ToolChecks.test_js_helper_termination_precedes_process_activation",
     '"tools/return-unused.mjs": False'),
    ("augmented assignment target reloaded after RHS effects", "census",
     "                augmented_prior = evaluated_augassign_targets.get(id(node))",
     "                augmented_prior = visible_module_bindings.get(node.target.id)",
     "tests.test_g11_census.ToolChecks.test_augmented_assignment_captures_target_before_rhs_effects",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("JavaScript default initializer treated as module execution", "census",
     "            (scope for scope in scopes\n"
     "             if (scope[0] <= candidate < scope[1]\n"
     "                 or scope[2] < candidate < scope[3])),",
     "            (scope for scope in scopes if scope[2] < candidate < scope[3]),",
     "tests.test_g11_census.ToolChecks.test_js_default_initializer_executes_only_for_invoked_helper",
     '{"tools/live.mjs": True, "tools/unused.mjs": False}'),
    ("possible branch rebind retains import identity", "census",
     "    elif isinstance(node, ast.Delete):\n"
     "        # Synthetic Delete nodes carry fail-closed invalidations for names\n"
     "        # that a possible control-flow path may rebind.  They must invalidate\n"
     "        # trusted import identities as well as static values and wrappers.\n"
     "        targets = node.targets",
     "    elif isinstance(node, ast.Delete):\n"
     "        targets = []",
     "tests.test_g11_census.ToolChecks.test_unknown_branch_rebinding_invalidates_local_import_identity",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("loop termination tail activated", "census",
     "        if after_unconditional_loop_termination(candidate):\n            return set()",
     "        if False:\n            return set()",
     "tests.test_g11_census.ToolChecks.test_js_loop_termination_precedes_process_activation",
     '{"tools/break-unused.mjs": False,'),
    ("unselected JavaScript switch case activated", "census",
     "        if in_unproven_switch(candidate):\n            return set()",
     "        if False:\n            return set()",
     "tests.test_g11_census.ToolChecks.test_js_switch_requires_proven_case_selection",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("Python call target identity not captured before arguments", "census",
     "                captured_function = active_functions.get((scope, target.id))",
     "                captured_function = None",
     "tests.test_g11_census.ToolChecks.test_helper_identity_is_captured_before_call_argument_effects",
     'self.assertTrue(self.tools(tree)["tools/live.py"]["referenced"])'),
    ("supplied JavaScript argument still runs default initializer", "census",
     '                        and arguments[default_parameter].strip() != "undefined"):',
     "                        and False):",
     "tests.test_g11_census.ToolChecks.test_js_default_initializer_requires_missing_or_undefined_argument",
     "self.assertEqual("),
    ("nested JavaScript parameter scopes flattened", "census",
     "    for param_start, param_end, open_brace in open_scopes:",
     "    for param_start, param_end, open_brace in [\n"
     "            scope for scope in open_scopes\n"
     "            if '(' not in searchable[scope[0]:scope[1]]]:",
     "tests.test_g11_census.ToolChecks.test_js_nested_parameter_scopes_are_deferred",
     "self.assertEqual("),
    ("unproven JavaScript catch handler activated", "census",
     "        if in_unproven_catch(candidate):\n            return set()",
     "        if False:\n            return set()",
     "tests.test_g11_census.ToolChecks.test_js_unproven_catch_handler_is_not_activated",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("skipped JavaScript nullish operand activated", "census",
     '    if operator.group() == "??":\n'
     '        return literal.group(1) in {"null", "undefined"}',
     '    if operator.group() == "??":\n'
     '        return True',
     "tests.test_g11_census.ToolChecks.test_js_nullish_coalescing_requires_rhs_evaluation",
     '{"tools/live.mjs": True, "tools/skipped.mjs": False}'),
    ("JavaScript regular-expression literal treated as code", "census",
     "            if can_start:\n"
     "                cursor = index + 1",
     "            if False:\n"
     "                cursor = index + 1",
     "tests.test_g11_census.ToolChecks.test_js_regular_expression_literal_is_not_a_process_call",
     '{"tools/live.mjs": True, "tools/regex-unused.mjs": False}'),
    ("completed JavaScript block termination not propagated", "census",
     "            blocks.append((block_open, block_close))\n"
     "        token = re.compile",
     "            if False:\n"
     "                blocks.append((block_open, block_close))\n"
     "        token = re.compile",
     "tests.test_g11_census.ToolChecks.test_js_completed_standalone_block_propagates_termination",
     '{"tools/live.mjs": True, "tools/unused.mjs": False}'),
    ("unresolved definition-time effects ignored", "census",
     "            names.add(node.name)\n"
     "            for expression in node.decorator_list:\n"
     "                self.visit(expression)\n"
     "            for expression in node.args.defaults:\n"
     "                self.visit(expression)",
     "            names.add(node.name)\n"
     "            for expression in node.decorator_list:\n"
     "                self.visit(expression)\n"
     "            for expression in ():\n"
     "                self.visit(expression)",
     "tests.test_g11_census.ToolChecks.test_unknown_branch_definition_time_effects_invalidate_downstream_bindings",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("helper-local parameter alias mutation ignored", "census",
     "                formal = formal_aliases.get(mutated_name, mutated_name)",
     "                formal = mutated_name",
     "tests.test_g11_census.ToolChecks.test_unknown_branch_helper_parameter_alias_effects_invalidate_caller_binding",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("optional JavaScript call arguments activated", "census",
     "        if not _js_optional_call_argument_reachable(searchable, candidate):\n"
     "            return set()",
     "        if False:\n"
     "            return set()",
     "tests.test_g11_census.ToolChecks.test_js_optional_call_arguments_require_proven_evaluation",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("chained assignment stores delayed past target effects", "census",
     "            for target in node.targets:\n"
     "                store_assignment_target(target, target)",
     "            for target in reversed(node.targets):\n"
     "                store_assignment_target(target, target)",
     "tests.test_g11_census.ToolChecks.test_chained_assignment_interleaves_target_stores_and_effects",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("unresolved lambda default effects ignored", "census",
     "        def visit_Lambda(self, node):\n"
     "            for expression in node.args.defaults:\n"
     "                self.visit(expression)\n"
     "            for expression in node.args.kw_defaults:\n"
     "                if expression is not None:\n"
     "                    self.visit(expression)\n"
     "            return\n\n"
     "        def visit_Name(self, node):",
     "        def visit_Lambda(self, node):\n"
     "            return\n\n"
     "        def visit_Name(self, node):",
     "tests.test_g11_census.ToolChecks.test_unknown_branch_lambda_defaults_invalidate_downstream_bindings",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("optional JavaScript property chains activated", "census",
     '    for optional in re.finditer(r"\\?\\.", searchable):',
     '    for optional in ():',
     "tests.test_g11_census.ToolChecks.test_js_optional_property_chains_require_proven_evaluation",
     '"tools/property-unused.mjs": False'),
    ("JavaScript instance field initializer treated as module execution", "census",
     "        if _js_in_deferred_instance_field(searchable, candidate):\n"
     "            return set()",
     "        if False:\n"
     "            return set()",
     "tests.test_g11_census.ToolChecks.test_js_unconstructed_instance_field_initializer_is_deferred",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("JavaScript regex after control header treated as code", "census",
     '                return bool(re.search(r"\\b(?:if|for|while|with)\\s*$", prefix))',
     "                return False",
     "tests.test_g11_census.ToolChecks.test_js_regular_expression_after_control_header_is_not_a_process_call",
     '"tools/live.mjs": True, "tools/regex-unused.mjs": False'),
    ("unresolved class-construction effects ignored", "census",
     "            names.add(node.name)\n"
     "            for expression in node.decorator_list:\n"
     "                self.visit(expression)\n"
     "            for expression in node.bases:\n"
     "                self.visit(expression)\n"
     "            for keyword in node.keywords:\n"
     "                self.visit(keyword.value)\n"
     "            for statement in node.body:\n"
     "                self.visit(statement)",
     "            names.add(node.name)\n"
     "            for expression in node.decorator_list:\n"
     "                self.visit(expression)\n"
     "            for expression in node.bases:\n"
     "                self.visit(expression)\n"
     "            for keyword in node.keywords:\n"
     "                self.visit(keyword.value)\n"
     "            for statement in ():\n"
     "                self.visit(statement)",
     "tests.test_g11_census.ToolChecks.test_unknown_branch_class_construction_invalidates_downstream_bindings",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("helper definition-time parameter mutation ignored", "census",
     "        def visit_ClassDef(self, node):\n"
     "            for expression in node.decorator_list:\n"
     "                self.visit(expression)\n"
     "            for expression in node.bases:\n"
     "                self.visit(expression)\n"
     "            for keyword in node.keywords:\n"
     "                self.visit(keyword.value)\n"
     "            for statement in node.body:\n"
     "                self.visit(statement)\n"
     "            return\n\n"
     "        def visit_Lambda(self, node):\n"
     "            for expression in node.args.defaults:\n"
     "                self.visit(expression)",
     "        def visit_ClassDef(self, node):\n"
     "            for expression in node.decorator_list:\n"
     "                self.visit(expression)\n"
     "            for expression in node.bases:\n"
     "                self.visit(expression)\n"
     "            for keyword in node.keywords:\n"
     "                self.visit(keyword.value)\n"
     "            for statement in node.body:\n"
     "                self.visit(statement)\n"
     "            return\n\n"
     "        def visit_Lambda(self, node):\n"
     "            for expression in ():\n"
     "                self.visit(expression)",
     "tests.test_g11_census.ToolChecks.test_unknown_branch_helper_definition_time_parameter_mutation_invalidates_caller",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("continuous optional JavaScript property chain activated", "census",
     '            elif index < len(searchable) and searchable[index] == ".":',
     '            elif False:',
     "tests.test_g11_census.ToolChecks.test_js_continuous_optional_property_chain_requires_proven_evaluation",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("computed JavaScript instance field initializer treated as module execution", "census",
     '            r"(?:[#A-Za-z_$][\\w$#]*|\\[[^\\]\\r\\n]*\\])\\s*=")',
     '            r"[#A-Za-z_$][\\w$#]*\\s*=")',
     "tests.test_g11_census.ToolChecks.test_js_unconstructed_computed_instance_field_initializer_is_deferred",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("optional JavaScript computed or call continuation activated", "census",
     '        while True:\n            index = chain_end\n',
     '        while False:\n            index = chain_end\n',
     "tests.test_g11_census.ToolChecks.test_js_continuous_optional_computed_and_call_chains_require_evaluation",
     '"tools/computed-unused.mjs": False'),
    ("computed JavaScript method body treated as module execution", "census",
     '        r"(?<![\\w$])(?P<name>[A-Za-z_$][\\w$]*|\\[[^\\]\\r\\n]*\\]|"\n'
     '        r"\'\\s*\'|\\\"\\s*\\\")\\s*\\(")',
     '        r"(?<![\\w$])(?P<name>[A-Za-z_$][\\w$]*|\'\\s*\'|"\n'
     '        r"\\\"\\s*\\\")\\s*\\(")',
     "tests.test_g11_census.ToolChecks.test_js_uninvoked_computed_method_bodies_are_not_module_execution",
     '"tools/object-unused.mjs": False'),
    ("semicolonless JavaScript instance field boundary ignored", "census",
     '            r"(?:^|[;{}\\r\\n])\\s*(?!static\\b)(?:accessor\\s+)?"',
     '            r"(?:^|[;{}])\\s*(?!static\\b)(?:accessor\\s+)?"',
     "tests.test_g11_census.ToolChecks.test_js_semicolonless_instance_field_boundary_is_deferred",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("helper class-construction parameter mutation ignored", "census",
     "        def visit_ClassDef(self, node):\n"
     "            for expression in node.decorator_list:\n"
     "                self.visit(expression)\n"
     "            for expression in node.bases:\n"
     "                self.visit(expression)\n"
     "            for keyword in node.keywords:\n"
     "                self.visit(keyword.value)\n"
     "            for statement in node.body:\n"
     "                self.visit(statement)\n"
     "            return\n\n"
     "        def visit_Lambda(self, node):",
     "        def visit_ClassDef(self, node):\n"
     "            for expression in node.decorator_list:\n"
     "                self.visit(expression)\n"
     "            for expression in node.bases:\n"
     "                self.visit(expression)\n"
     "            for keyword in node.keywords:\n"
     "                self.visit(keyword.value)\n"
     "            for statement in ():\n"
     "                self.visit(statement)\n"
     "            return\n\n"
     "        def visit_Lambda(self, node):",
     "tests.test_g11_census.ToolChecks.test_unknown_branch_helper_class_construction_parameter_mutation_invalidates_caller",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("downstream JavaScript await suspension ignored", "census",
     "        if after_unproven_await(candidate):\n"
     "            return set()",
     "        if False:\n"
     "            return set()",
     "tests.test_g11_census.ToolChecks.test_js_process_call_after_await_is_not_proven_reachable",
     '"tools/unused.mjs": False'),
    ("nested computed JavaScript method name unbalanced", "census",
     "    for computed_start, char in enumerate(searchable):\n"
     "        if char != \"[\":",
     "    for computed_start, char in ():\n"
     "        if char != \"[\":",
     "tests.test_g11_census.ToolChecks.test_js_uninvoked_nested_computed_method_body_is_not_module_execution",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("destructuring assignment element stores delayed", "census",
     "                if isinstance(target, (ast.Tuple, ast.List)):\n"
     "                    for element in target.elts:\n"
     "                        store_assignment_target(element, root_target)\n"
     "                    return",
     "                if False:\n"
     "                    for element in target.elts:\n"
     "                        store_assignment_target(element, root_target)\n"
     "                    return",
     "tests.test_g11_census.ToolChecks.test_destructuring_assignment_interleaves_element_stores_and_effects",
     'self.assertFalse(self.tools(tree)["tools/unused.py"]["referenced"])'),
    ("quoted JavaScript method name not deferred", "census",
     '        r"(?<![\\w$])(?P<name>[A-Za-z_$][\\w$]*|\\[[^\\]\\r\\n]*\\]|"\n'
     '        r"\'\\s*\'|\\\"\\s*\\\")\\s*\\(")',
     '        r"(?<![\\w$])(?P<name>[A-Za-z_$][\\w$]*|\\[[^\\]\\r\\n]*\\])\\s*\\(")',
     "tests.test_g11_census.ToolChecks.test_js_uninvoked_quoted_method_bodies_are_not_module_execution",
     '"tools/object-unused.mjs": False'),
    ("nested computed JavaScript field name unbalanced", "census",
     "        for prefix in computed_member.finditer(\n"
     "                searchable, open_brace, position):",
     "        for prefix in ():",
     "tests.test_g11_census.ToolChecks.test_js_unconstructed_nested_computed_instance_field_is_deferred",
     'self.assertFalse(self.tools(tree)["tools/unused.mjs"]["referenced"])'),
    ("destructuring leaf values recomputed after target effects", "census",
     "                    value = evaluated_assignment_target_values.get(\n"
     "                        node._g11_captured_assignment_id, {}).get(\n"
     "                        node._g11_captured_assignment_target_id,\n"
     "                        _UNRESOLVED_STATIC)",
     "                    value = _python_assignment_target_values(\n"
     "                        node._g11_captured_assignment_root, value).get(\n"
     "                        node._g11_captured_assignment_target_id,\n"
     "                        _UNRESOLVED_STATIC)",
     "tests.test_g11_census.ToolChecks.test_destructuring_snapshots_leaf_values_before_target_effects",
     "self.assertTrue(self.tools("),
    ("JavaScript static-block assignment treated as instance field", "census",
     "            return depth == 0",
     "            return True",
     "tests.test_g11_census.ToolChecks.test_js_static_block_computed_assignment_executes_at_class_definition",
     'self.assertTrue(self.tools(tree)["tools/live.mjs"]["referenced"])'),

]
EXPECTED_MUTATION_NAMES = (
    "pass not detected",
    "ellipsis not detected",
    "NotImplementedError not detected",
    "raise call not unwrapped",
    "docstring-only not reported",
    "docstring not stripped",
    "decorator context ignored",
    "Protocol/ABC context ignored",
    "exception context ignored",
    "log_message context ignored",
    "log_message context too broad",
    "module docstring not recorded",
    "class bodies not visited",
    "nested functions not visited",
    "comments read as data",
    "strings read as comments",
    "marker words case-insensitive",
    "marker word boundary dropped",
    "quotes not tracked",
    "string escapes ignored",
    "hash comment needs no whitespace",
    "block comments not entered",
    "block comment end ignored",
    "mentions dropped",
    "noop detection off",
    "block scalar not read",
    "block scalar ends late",
    "inline comment kept",
    "tests counted as tools",
    "__init__ counted as tool",
    "basename ambiguity ignored",
    "path reference not recognised",
    "import not recognised",
    "from-import list not recognised",
    "basename not recognised",
    "transitive exercise off",
    "site not production",
    "research counted as production",
    "vendor counted as site",
    "undecodable decoded leniently",
    "unparseable not a failure class",
    "legitimate context still fails",
    "markers do not fail",
    "noop steps do not fail",
    "tool findings do not fail",
    "exit status always zero",
    "unreadable revision exits zero",
    "decoy decorator names accepted",
    "decoy abstract-base names accepted",
    "decoy request-handler suffix accepted",
    "comment and string mentions count as exercise",
    "empty workflow run block not detected",
    "YAML comments omitted",
    "mutation attribution accepts wrong assertion",
    "function-local decorator shadow accepted",
    "python -c path mention counts as execution",
    "shell -c path mention counts as execution",
    "template literal import text counts as static import",
    "parser drops method identity",
    "parser repeats qualified method identity",
    "subTest failure header not parsed",
    "wrong test id credited",
    "nonfailure status credited",
    "error outcome credited",
    "skipped outcome credited",
    "quoted workflow command boundary ignored",
    "FAIL header mistaken for exception",
    "outer class exemption leaks",
    "local base shadow accepted",
    "uncalled helper counts as exercise",
    "heredoc body counted as command",
    "mutation inventory shrink accepted",
    "hostile Python environment survives",
    "in-scope tracked symlink silently skipped",
    "decoy TestCase base credited",
    "workflow defaults accepted as steps",
    "static path assignments not bound",
    "spec loader path ignored",
    "runpy path ignored",
    "subprocess path list ignored",
    "environment path default ignored",
    "legacy os.path construction ignored",
    "concatenated command lists ignored",
    "lazy-load none guard ignored",
    "arbitrary unknown branch accepted",
    "process wrapper forwarding ignored",
    "nonprocess wrapper accepted",
    "sibling with_name path ignored",
    "JavaScript URL tool binding ignored",
    "JavaScript URL mention accepted without child process",
    "decoy Python loader APIs accepted",
    "decoy run_path API accepted",
    "decoy JavaScript child process accepted",
    "absolute parent traversal accepts the filesystem root",
    "resolved command list credits command-mode arguments",
    "unreachable wrapper process call accepted",
    "function-local process import ignored",
    "JavaScript same-line mention accepted outside call arguments",
    "destructured path values rebound as aggregate",
    "rebound wrapper parameter still forwarded",
    "local static shadow inherits module value",
    "JavaScript quoted and property names credited",
    "forwarded command-mode arguments credited",
    "augmented assignment retains stale static value",
    "JavaScript property call accepted as imported callee",
    "unresolved command element dropped",
    "boolean or evaluated in reverse order",
    "shadowed callee inherits module wrapper",
    "quoted JavaScript text accepted as child process call",
    "resolved starred command elements ignored",
    "dynamic command suffix discards proven prefix",
    "unittest lifecycle methods ignored",
    "import loader wrapper forwarding ignored",
    "class method registered as module wrapper",
    "conditional command alternatives merged",
    "reassigned JavaScript URL binding retains stale path",
    "path constructor ignores later components",
    "command element assignment retains stale sequence",
    "JavaScript parameter shadow accepted as imported callee",
    "mode switch after script mistaken for interpreter mode",
    "wrapper element write retains caller provenance",
    "helper element write falls back to module binding",
    "JavaScript default initializer treated as a binding",
    "interpreter option operand mistaken for script",
    "command alias survives element write",
    "JavaScript URL binding parameter shadow ignored",
    "list mutator calls retain stale command provenance",
    "JavaScript local declaration shadow ignored",
    "JavaScript URL declaration escapes function scope",
    "wrapper parameter alias mutation retains caller provenance",
    "reassigned wrapper remains registered",
    "JavaScript enclosing declaration shadow ignored",
    "wrapper traversal continues after return",
    "deferred wrapper availability uses source line",
    "later nonforwarding wrapper definition retains registration",
    "list augmented assignment breaks alias identity",
    "JavaScript unreachable child-process call accepted",
    "deferred caller uses final module binding",
    "JavaScript uncalled helper chain accepted",
    "node test callback root ignored",
    "none guard branch states combined",
    "node test registration reachability ignored",
    "node test registrar shadow ignored",
    "none guard branch environments merged",
    "nested helper effects deferred",
    "global assignment stored in helper local binding",
    "command element deletion retains stale sequence",
    "JavaScript helper rebinding ignored",
    "wrapper element deletion retains forwarding",
    "declared global remains a local static shadow",
    "JavaScript lexical helper shadow ignored",
    "keyword argument helper effects deferred",
    "reassigned Python helper identity ignored",
    "JavaScript deferred binding checked at definition",
    "earlier argument value reevaluated after later effects",
    "assignment target invalidated before RHS helper",
    "sequence element value reevaluated after later effects",
    "JavaScript module block shadow ignored",
    "captured sequence values lost through concatenation",
    "Python helper traversal continues after termination",
    "JavaScript destructured block shadow ignored",
    "named binary operand reevaluated after later effects",
    "selected branch traversal continues after termination",
    "JavaScript ASI declaration shadow ignored",
    "repeated helper invocation suppressed",
    "global helper reassignment invalidates local identity",
    "Boolean short-circuit reachability ignored",
    "conditional expression reachability ignored",
    "async helper creation treated as execution",
    "async forwarding wrapper treated as execution",
    "generator helper creation treated as execution",
    "comprehension body treated as unconditional",
    "chained comparison short circuit ignored",
    "JavaScript URL lexical block ignored",
    "generator method creation treated as execution",
    "comprehension target shadow ignored",
    "forwarding wrapper Boolean reachability ignored",
    "forwarding wrapper conditional reachability ignored",
    "global wrapper reassignment kept in helper scope",
    "wrapper reassignment uses lexical line instead of execution order",
    "lambda body traversed at construction",
    "nonlocal active function destination ignored",
    "nonlocal wrapper destination ignored",
    "wrapper keyword argument effects deferred",
    "forwarding wrapper chained-comparison reachability ignored",
    "exception handler reachability ignored",
    "skipped exception branch effects ignored",
    "wrapper exception-handler reachability ignored",
    "forwarding wrapper comprehension reachability ignored",
    "finally termination not propagated",
    "JavaScript URL destructured lexical shadow ignored",
    "nonlocal static assignment kept in helper scope",
    "unknown branch effects ignored",
    "handler-free try body termination not propagated",
    "decorated Python helper retains original body",
    "non-string executable reaches interpreter membership",
    "JavaScript destructured helper shadow ignored",
    "JavaScript composite argument identifiers credited",
    "JavaScript file URL converter identity ignored",
    "possible-effect marker bypasses active helper invalidation",
    "possible deletion ignored for wrapper identity",
    "exception target binding effect ignored",
    "noniterable literal accepted as comprehension source",
    "unsupported interpreter option skipped",
    "assignment expression write ignored",
    "nested aggregate mutation retains reachable alias provenance",
    "decorated Python forwarding wrapper registered",
    "JavaScript URL options argument credited",
    "reachable events resorted by source line",
    "direct loop effects ignored",
    "forwarding wrapper loop effects ignored",
    "subprocess executable override ignored",
    "class body bindings leak into enclosing scope",
    "class runtime shadows ignored",
    "subprocess shell mode ignored",
    "subprocess positional options accepted",
    "JavaScript URL constructor identity ignored",
    "JavaScript URL property write retains stale provenance",
    "JavaScript expression short-circuit reachability ignored",
    "helper invocation binding frame reused",
    "helper invocation import frame reused",
    "helper invocation nested definitions reused",
    "aggregate str conversion retains argv semantics",
    "forwarding wrapper unknown branch effects ignored",
    "JavaScript URL initializer suffix accepted",
    "JavaScript ternary arm reachability ignored",
    "skipped Boolean operand effects ignored",
    "forwarding wrapper skipped Boolean operand effects ignored",
    "nested helper free name falls back to module binding",
    "Python helper signature binding ignored",
    "unknown conditional expression effects ignored",
    "forwarding wrapper unknown conditional expression effects ignored",
    "forwarding wrapper call signature ignored",
    "assignment expression leaves import identity stale",
    "tuple augmented assignment mutates aliases",
    "JavaScript unbraced literal-if reachability ignored",
    "JavaScript newline call continuation treated as ASI",
    "JavaScript nested-if else not consumed",
    "JavaScript concise arrow body treated as module execution",
    "JavaScript concise node-test callback not activated",
    "JavaScript spaced block arrow registered as concise",
    "annotation-only assignment invalidates import identity",
    "helper argument alias binding ignored",
    "unittest setup ordering ignored",
    "quoted JavaScript URL declaration accepted",
    "JavaScript URL alias escape retains stale provenance",
    "default aggregate argument binding ignored",
    "variadic positional aggregate binding ignored",
    "variadic keyword aggregate binding ignored",
    "assignment target effects ignored",
    "unknown chained-comparison suffix effects ignored",
    "forwarding wrapper unknown chained-comparison suffix effects ignored",
    "unresolved eager-comprehension effects ignored",
    "forwarding wrapper unresolved eager-comprehension effects ignored",
    "uninvoked JavaScript methods treated as module execution",
    "literal-true JavaScript else arm treated as reachable",
    "unknown branch helper effects ignored",
    "quoted JavaScript child-process import trusted",
    "dead JavaScript URL initializer trusted",
    "unknown branch helper parameter mutation ignored",
    "nested selected JavaScript branch stops reachability checks",
    "unproven JavaScript loop body activated",
    "terminated JavaScript helper tail activated",
    "augmented assignment target reloaded after RHS effects",
    "JavaScript default initializer treated as module execution",
    "possible branch rebind retains import identity",
    "loop termination tail activated",
    "unselected JavaScript switch case activated",
    "Python call target identity not captured before arguments",
    "supplied JavaScript argument still runs default initializer",
    "nested JavaScript parameter scopes flattened",
    "unproven JavaScript catch handler activated",
    "skipped JavaScript nullish operand activated",
    "JavaScript regular-expression literal treated as code",
    "completed JavaScript block termination not propagated",
    "unresolved definition-time effects ignored",
    "helper-local parameter alias mutation ignored",
    "optional JavaScript call arguments activated",
    "chained assignment stores delayed past target effects",
    "unresolved lambda default effects ignored",
    "optional JavaScript property chains activated",
    "JavaScript instance field initializer treated as module execution",
    "JavaScript regex after control header treated as code",
    "unresolved class-construction effects ignored",
    "helper definition-time parameter mutation ignored",
    "continuous optional JavaScript property chain activated",
    "computed JavaScript instance field initializer treated as module execution",
    "optional JavaScript computed or call continuation activated",
    "computed JavaScript method body treated as module execution",
    "semicolonless JavaScript instance field boundary ignored",
    "helper class-construction parameter mutation ignored",
    "downstream JavaScript await suspension ignored",
    "nested computed JavaScript method name unbalanced",
    "destructuring assignment element stores delayed",
    "quoted JavaScript method name not deferred",
    "nested computed JavaScript field name unbalanced",
    "destructuring leaf values recomputed after target effects",
    "JavaScript static-block assignment treated as instance field",
)


def _build_cases():
    cases = []
    for name, old, new in MUTATIONS:
        target_test, fragment = TARGETS[name]
        cases.append((name, "census", old, new, target_test, fragment))
    cases.extend(NEW_MUTATIONS)
    return cases


def _declared_mutation_cases(source):
    tree = ast.parse(source)
    required = {"MUTATIONS", "TARGETS", "NEW_MUTATIONS", "EXPECTED_MUTATION_NAMES"}
    values = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in required):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    if set(values) != required:
        raise HarnessError("mutation inventory declarations are incomplete")
    cases = []
    for item in values["MUTATIONS"]:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise HarnessError("mutation inventory declaration has an invalid census tuple")
        name, old, new = item
        try:
            target_test, fragment = values["TARGETS"][name]
        except (KeyError, TypeError, ValueError) as exc:
            raise HarnessError("mutation inventory target missing for %r: %s" % (name, exc))
        cases.append((name, "census", old, new, target_test, fragment))
    cases.extend(values["NEW_MUTATIONS"])
    expected_names = tuple(values["EXPECTED_MUTATION_NAMES"])
    if tuple(case[0] for case in cases) != expected_names:
        raise HarnessError("mutation inventory names disagree with literal declarations")
    return tuple(cases)


def _validate_mutation_inventory(cases):
    try:
        declared = _declared_mutation_cases(Path(__file__).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError, TypeError, KeyError) as exc:
        raise HarnessError("mutation inventory cannot be parsed: %s" % exc)
    if tuple(cases) != declared:
        raise HarnessError("mutation inventory mismatch: complete case tuples differ")



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
        method, test_id = header.group(1), header.group(2)
        # Python 3.11+ prints "method (module.Class.method)"; earlier versions omit the method.
        if not test_id.endswith("." + method):
            test_id += "." + method
        result["test_id"] = test_id
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


PYTHON_ENVIRONMENT_KEYS = ("PYTHONPATH", "PYTHONHOME", "PYTHONPYCACHEPREFIX",
                           "PYTHONSTARTUP", "PYTHONUSERBASE")


def _clean_child_env(source=None):
    env = dict(os.environ if source is None else source)
    for name in PYTHON_ENVIRONMENT_KEYS:
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _run(workdir, test_id=None, env=None):
    command = [sys.executable, "-B", "-S"]
    if sys.flags.optimize:
        command.append("-O")
    command.extend(["-m", "unittest"])
    command.append(test_id if test_id else "tests.test_g11_census")
    command.append("-v")
    proc = subprocess.run(command, cwd=workdir, env=_clean_child_env(env),
                          capture_output=True, text=True, timeout=300)
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
    cases = _build_cases()
    _validate_mutation_inventory(cases)
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
        baseline_ok = (baseline.get("parsed") and baseline.get("exit") == 0 and baseline.get("ran") == 231
                       and baseline.get("failures") == 0 and baseline.get("errors") == 0
                       and baseline.get("skipped") == 0 and baseline.get("status") == "OK")
        if baseline_ok:
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
    killed = sum(r["killed"] for r in results)
    return {"baseline": baseline, "mutations": results, "killed": killed, "total": len(cases),
            "executed": len(results),
            "status": "PASS" if baseline_ok and len(results) == len(cases) and killed == len(results) else "FAIL"}


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
        print("G11_SELFTEST_%s: baseline %d tests, %d/%d mutations killed (%d executed)" %
              (report["status"], report["baseline"].get("ran", -1), report["killed"],
               report["total"], report["executed"]))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
