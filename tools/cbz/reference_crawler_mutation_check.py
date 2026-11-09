#!/usr/bin/env python3
"""Adverse mutation check for tools/cbz/reference_crawler.py (governance A8 / B-006).

Each mutant disables or corrupts one detection, policy rule, extractor or topology
behaviour. A mutant counts as killed only when tests/test_cbz_reference_crawler.py ran
at least one test and one of the mutant's designated tests (DESIGNATED, governance B-011)
failed from an authorized assertion in that selected test body's actual code. A child
result captures traceback frame/code provenance before unittest cleanup. The selected
test may raise a plain assertion itself or directly invoke unittest's assertion machinery;
any intervening subject, helper or nested-callback frame makes the run invalid. An
error-only or unauthorized-assertion run is invalid, and a run in which only other tests
fail is a survivor, never a kill. Each kill is printed with its failure and error counts
and the designated test that failed. Every mutant must have designated tests, and each
designated test must exist in the suite.

The suite's population is frozen: EXPECTED_TESTS lists every test method and TEST_SHA256
is the SHA-256 of the test file's raw bytes (read without newline translation). Before
anything runs, the live file must match both, DESIGNATED must cover every frozen test,
and the unmutated baseline must run exactly len(EXPECTED_TESTS) tests. Controls prove
that an added test method, a deleted (undesignated) test method, a CRLF-only rewrite of
the file and a designation map that leaves a test uncovered are all refused.
With --failmap FILE, the evidence fail-map must record the same population and mutant
set, list every designated test among its mutant's assertion failures, and name only
frozen tests, as lists of strings, in every fail and error field. With
--write-failmap FILE, this run records which tests fail and error under each mutant plus
the pre-cleanup traceback/code provenance for every outcome, writes that v2 fail-map with
the frozen population, and checks it the same way. Corrupted
copies of it (one designated failure removed, an unknown test id added to a fail or an
error list, or an error field that is not a list) must be refused. Two controls must come out invalid, never killed: a syntax-broken copy,
and an error-only copy whose crawl() raises. The unmutated suite
must pass both before and after the mutants. If the environment changes mid-run, so
that errors could pass for kills, the run is invalid (exit 2). Exit 0 only if every
mutant is killed.
Run from the repository root:
  python3 -B -S tools/cbz/reference_crawler_mutation_check.py [--failmap FILE] [--write-failmap FILE]
  python3 -B -S tools/cbz/reference_crawler_mutation_check.py --shard 1/3 --write-failmap shard-1.json
  python3 -B -S tools/cbz/reference_crawler_mutation_check.py --merge-failmaps shard-1.json shard-2.json shard-3.json --write-failmap complete.json
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "cbz" / "reference_crawler.py"
ACKNOWLEDGEMENTS = TOOL.with_name("reference_crawler_acknowledged.json")
TEST = ROOT / "tests" / "test_cbz_reference_crawler.py"
GRADER_TEST = ROOT / "tests" / "test_reference_crawler_mutation_check.py"
RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
FAILED = re.compile(r"^FAILED \(([^)]*)\)$", re.M)
FAILURES = re.compile(r"\bfailures=(\d+)")
FAIL_NAME = re.compile(r"^FAIL: (\w+) ", re.M)
ERROR_NAME = re.compile(r"^ERROR: (\w+) ", re.M)
TEST_NAME = re.compile(r"^    def (test_\w+)\(", re.M)
CRAWL_START = "    findings = []\n    overrides = overrides or {}\n"
ORIGIN_SENTINEL = "ORIGIN_JSON="
LAST_ORIGIN_REPORT = None
ORIGIN_RUNNER = r'''
import inspect
import json
import sys
import unittest
import unittest.case
from pathlib import Path

def code_of(value):
    return getattr(getattr(value, "__func__", value), "__code__", None)

oracle_codes = tuple(
    code for cls in (unittest.TestCase, unittest.case._AssertRaisesContext,
                     unittest.case._AssertRaisesBaseContext, unittest.case._BaseTestCaseContext)
    for name, value in vars(cls).items()
    if name.startswith("assert") or name in
       ("fail", "_baseAssertEqual", "_formatMessage", "__exit__", "_raiseFailure")
    for code in [code_of(value)] if code is not None
)

def is_oracle(code):
    return any(code is allowed for allowed in oracle_codes)

def leaves(node):
    if isinstance(node, unittest.TestSuite):
        for child in node:
            yield from leaves(child)
    else:
        yield node

suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])
cases = list(leaves(suite))
ids = [case.id() for case in cases]
names = [getattr(case, "_testMethodName", "") for case in cases]
if not cases or len(set(ids)) != len(ids) or len(set(names)) != len(names):
    raise ValueError("test population must be nonempty and unique")
states = {}
for case in cases:
    selected = getattr(case, case._testMethodName, None)
    code = code_of(selected)
    if (code is None or inspect.iscoroutinefunction(selected) or
            inspect.isgeneratorfunction(selected) or inspect.isasyncgenfunction(selected)):
        raise ValueError("only synchronous selected test bodies are supported")
    state = {"code": code, "entered": 0, "phase": "initial"}
    states[case.id()] = state
    for attr, phase in (("_callSetUp", "setup"), ("_callTearDown", "teardown"),
                        ("_callCleanup", "cleanup")):
        original = getattr(case, attr)
        def wrap(*args, _original=original, _state=state, _phase=phase, **kwargs):
            _state["phase"] = _phase
            return _original(*args, **kwargs)
        setattr(case, attr, wrap)
    def call_body(method, _state=state):
        if code_of(method) is not _state["code"]:
            raise RuntimeError("selected test body changed")
        _state["entered"] += 1
        _state["phase"] = "body"
        try:
            returned = method()
            if returned is not None:
                if inspect.iscoroutine(returned) or inspect.isgenerator(returned):
                    returned.close()
                raise RuntimeError("unsupported test-body return value")
        finally:
            _state["phase"] = "body-exited"
    case._callTestMethod = call_body


class ProvenanceResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.events = []
        self.started = []
        self.stopped = []

    def startTest(self, test):
        self.started.append(test.id())
        super().startTest(test)

    def stopTest(self, test):
        self.stopped.append(test.id())
        super().stopTest(test)

    @staticmethod
    def selected_case(test):
        return getattr(test, "test_case", test)

    def record(self, test, err, kind):
        case = self.selected_case(test)
        method_name = getattr(case, "_testMethodName", "")
        state = states.get(case.id(), {})
        selected_code = state.get("code")
        frames = []
        codes = []
        tb = err[2]
        while tb is not None:
            code = tb.tb_frame.f_code
            codes.append(code)
            frames.append({
                "filename": str(Path(code.co_filename).resolve()),
                "firstlineno": code.co_firstlineno,
                "name": code.co_name,
                "qualname": getattr(code, "co_qualname", code.co_name),
                "lineno": tb.tb_lineno,
                "selected": code is selected_code,
                "oracle": is_oracle(code),
            })
            tb = tb.tb_next
        positions = [i for i, code in enumerate(codes) if code is selected_code]
        authorized = (kind == "failure" and type(err[1]) is AssertionError and
                      state.get("entered") == 1 and
                      state.get("phase") in ("body", "body-exited") and
                      len(positions) == 1 and all(is_oracle(code) for code in codes[positions[0] + 1:]))
        self.events.append({
            "test": method_name or test.id(),
            "kind": kind,
            "authorized": authorized,
            "exception": err[0].__name__,
            "selected": {
                "filename": str(Path(selected_code.co_filename).resolve()) if selected_code else None,
                "firstlineno": selected_code.co_firstlineno if selected_code else None,
                "name": selected_code.co_name if selected_code else None,
            },
            "frames": frames,
        })

    def addFailure(self, test, err):
        self.record(test, err, "failure")
        super().addFailure(test, err)

    def addError(self, test, err):
        self.record(test, err, "error")
        super().addError(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            kind = "failure" if issubclass(err[0], test.failureException) else "error"
            self.record(subtest, err, kind)
        super().addSubTest(test, subtest, err)


result = ProvenanceResult()
suite.run(result)
authorized = sorted({event["test"] for event in result.events if event["kind"] == "failure" and event["authorized"]})
unauthorized = sorted({event["test"] for event in result.events if event["kind"] == "failure" and not event["authorized"]})
errors = sorted({event["test"] for event in result.events if event["kind"] == "error"})
print("ORIGIN_JSON=" + json.dumps({
    "schema": "reference-crawler-origin-v3",
    "optimize": sys.flags.optimize,
    "tests_run": result.testsRun,
    "ids": ids,
    "names": names,
    "started": result.started,
    "stopped": result.stopped,
    "failure_count": len(result.failures),
    "error_count": len(result.errors),
    "skip_count": len(result.skipped),
    "expected_failure_count": len(result.expectedFailures),
    "unexpected_success_count": len(result.unexpectedSuccesses),
    "authorized_failures": authorized,
    "unauthorized_failures": unauthorized,
    "errors": errors,
    "events": result.events,
}, sort_keys=True))
'''
MUTANTS = [
    ("blob identity ignored", "for b in ref.blobs if b != oid]", "for b in ref.blobs if False]"),
    ("sha256 ignored", "for d in ref.sha256s if d != digest]", "for d in ref.sha256s if False]"),
    ("bytes ignored", "for n in ref.sizes if n != len(data)]", "for n in ref.sizes if False]"),
    ("missing path silent", 'findings.append(finding(cls, target.name, path, f"path absent at {commit}", ref))', "pass"),
    ("missing commit misclassified",
     'findings.append(finding("MISSING_COMMIT", target.name, ref.path, f"commit {commit} not found", ref))',
     'findings.append(finding("FROZEN_REMOVED", target.name, ref.path, f"commit {commit} not found", ref))'),
    ("live drift misclassified", 'cls = "LIVE_DRIFT" if live else "IDENTITY_MISMATCH"', 'cls = "IDENTITY_MISMATCH"'),
    ("unknown repository ignored", 'findings.append(finding("UNKNOWN_REPOSITORY",', '(finding("UNKNOWN_REPOSITORY",'),
    ("foreign reference silent", 'findings.append(finding("FOREIGN_REPOSITORY",', '(finding("FOREIGN_REPOSITORY",'),
    ("same-name foreign repository conflated", "        target = None  # another owner's repository with the same short name",
     "        pass"),
    ("strict ignored", '(strict and interface_drifted(row) and "acknowledged" not in row)', "False"),
    ("lean statements blind", "**diff_maps(lean_statements(old), lean_statements(new))", "**diff_maps({}, {})"),
    ("python signatures drop arguments", 'return f"{prefix} {fn.name}({ast.unparse(fn.args)}){ret}"',
     'return f"{prefix} {fn.name}(){ret}"'),
    ("json shape shallow", "SHAPE_DEPTH = 6", "SHAPE_DEPTH = 0"),
    ("foreign owners included", "        if owner.lower() not in owners:\n            continue", "        pass"),
    ("unparseable silent", 'findings.append(finding("UNPARSEABLE", source, path,', "(finding(\"UNPARSEABLE\", source, path,"),
    ("unpinned not reported", 'findings.append(finding("TOPOLOGY_UNPINNED", node.name, "", "registered but not pinned"))',
     "pass"),
    ("blocking ignored at exit", 'return 1 if report["blocking"] else 0', "return 0"),
    ("nested comments not tracked", "depth, i = depth + 1, i + 2", "depth, i = depth, i + 2"),
    ("line comments kept", 'elif text.startswith("--", i):', "elif False:"),
    ("pin skew silent", 'findings.append(finding("PIN_SKEW", node.name, "", f"pinned {node.commit}, tip {tip}"))', "pass"),
    ("require-current-pins ignored", '(require_current_pins and row["class"] == "PIN_SKEW")', "False"),
    ("unreachable repository silent",
     'findings.append(finding("UNREACHABLE_REPOSITORY", node.name, "", str(exc)))', "pass"),
    ("unreachable target not blocking", '"TARGET_UNREACHABLE", target.name', '"FROZEN_REMOVED", target.name'),
    ("missing pin silent",
     'findings.append(finding("MISSING_PIN", node.name, "", f"pinned commit {node.commit} not found"))', "pass"),
    ("frozen change unreported", 'elif current[1] != oid:', "elif False:"),
    ("url path not decoded", 'path = urllib.parse.unquote(m.group("path").rstrip(".:;"))',
     'path = m.group("path").rstrip(".:;")'),
    ("excerpt lines never read", 'lines = (obj["line_start"], obj["line_end"]) if', "lines = None if"),
    ("excerpt checked as whole file", 'if ref.lines and kind == "blob":', "if False:"),
    ("excerpt off by one", 'excerpt = "".join(file_lines[start - 1:end])', 'excerpt = "".join(file_lines[start:end])'),
    ("excerpt identity unchecked", 'problems += identity_problems(excerpt, ref, f" in lines {start}-{end}")', "pass"),
    ("excerpt range unchecked", "if not 1 <= start <= end <= len(file_lines):", "if False:"),
    ("excerpt text ignored", 'if ref.text is not None and ref.text.encode("utf-8") != excerpt:', "if False:"),
    ("fixtures not skipped", "if not include_fixtures and is_fixture(path):", "if False:"),
    ("include-fixtures ignored", "references.extend(collect(node, owners, findings, include_fixtures, skipped, binary))",
     "references.extend(collect(node, owners, findings, False, skipped, binary))"),
    ("skipped fixtures not counted", "skipped.setdefault(node.name, []).append(path)", "pass"),
    ("fixture match includes file names", "for part in parts[:-1])", "for part in parts)"),
    ("skip count hidden in markdown", 'skipped = sum(report["skipped_fixture_files"].values())', "skipped = 0"),
    ("json strings not decoded",
     "for string in json_strings(doc):\n            text_references(string, source, path, owners, refs)",
     "text_references(text, source, path, owners, refs)"),
    ("backslash allowed in url paths", 'URL_PATH_CHAR = r"[^\\s()\\]>\\"\'`#?|,\\\\]"',
     'URL_PATH_CHAR = r"[^\\s()\\]>\\"\'`#?|,]"'),
    ("balanced parentheses dropped", "|\\({URL_PATH_CHAR}*\\))+)", ")+)"),
    ("extra blob keys ignored",
     'BLOB_KEYS = ("blob", "git_blob", "observed_blob", "blob_sha", "blob_sha1", "git_blob_sha", "git_blob_sha1")',
     'BLOB_KEYS = ("blob", "git_blob", "observed_blob")'),
    ("extra commit keys ignored", 'COMMIT_KEYS = ("commit", "ref", "source_commit", "cut", "head", "at_commit")',
     'COMMIT_KEYS = ("commit", "ref", "source_commit")'),
    ("raw_sha256 ignored", 'SHA256_KEYS = ("sha256", "raw_sha256")', 'SHA256_KEYS = ("sha256",)'),
    ("byte_length ignored", 'BYTES_KEYS = ("bytes", "byte_length")', 'BYTES_KEYS = ("bytes",)'),
    ("only the first recorded value checked", "if valid(obj.get(k)) and obj[k] not in out:",
     "if valid(obj.get(k)) and not out:"),
    ("byte-identical claims ignored", "    for key in also:", "    for key in ():"),
    ("malformed byte-identical values silent",
     "    malformed += [k for k in also if obj[k] is not None and not hex40(obj[k])]", "    pass"),
    ("byte-identical claims without identity accepted",
     '        report("MALFORMED_IDENTITY", f"{also[0]} claims byte identity, but there is no blob, sha256 or bytes to compare")\n        also = []',
     "        pass"),
    ("unterminated comment accepted",
     '            if end < 0:\n                raise ValueError("unterminated block comment")\n            i = end + 2',
     "            i = len(text) if end < 0 else end + 2"),
    ("topology incomplete not blocking", '"UNREAD", "TOPOLOGY_INCOMPLETE"}', '"UNREAD"}'),
    ("missing workspace registry silent", 'incomplete("workspace", ws_registry, "the workspace registry is missing")', "pass"),
    ("workspace registry shape silent", 'incomplete("workspace", ws_registry, "the workspace registry is not a JSON object")',
     "pass"),
    ("repositories list silent", 'incomplete("workspace", ws_registry, "the workspace registry has no repositories list")',
     "pass"),
    ("registry entry silent", 'incomplete("workspace", ws_registry, f"repositories[{i}] has no full_name")', "pass"),
    ("canonical input silent",
     'incomplete("workspace", ws_registry, "canonical_inputs.repository_registry does not name a repository and a path")',
     "pass"),
    ("registry host silent",
     'incomplete("workspace", ws_registry, f"the repository registry\'s host {reg_input[\'repository\']!r} is not in the topology")',
     "pass"),
    ("registry document missing silent",
     'incomplete(host.name, reg_input["path"], f"the repository registry is missing at {commit}")', "pass"),
    ("registry map silent", 'incomplete(host.name, reg_input["path"], "the repository registry has no repositories map")',
     "pass"),
    ("registry host unreachable silent",
     'incomplete(host.name, reg_input["path"], f"the registry host {host.name} is unreachable")', "pass"),
    ("registry host commit silent",
     'incomplete(host.name, reg_input["path"], f"the registry host\'s commit {commit or \'(none)\'} is not available")',
     "pass"),
    ("suffix filter restored", 'if kind != "blob":\n            continue\n        if not include_fixtures',
     'if kind != "blob" or not path.endswith((".md", ".txt", ".json")):\n            continue\n        if not include_fixtures'),
    ("json lines not parsed", "if path.endswith(JSON_LINES_SUFFIXES):\n        docs, rest = [], []",
     "if False:\n        docs, rest = [], []"),
    ("unparsed json lines dropped", "rest.append(line)", "pass"),
    ("binary files not counted", "binary.setdefault(node.name, []).append(path)", "pass"),
    ("oversize file silent", 'findings.append(finding("UNREAD",', '(finding("UNREAD",'),
    ("size limit ignored", "if size > MAX_TEXT_BYTES:", "if False:"),
    ("binary count hidden in markdown", 'binary = sum(report["binary_files_not_read"].values())', "binary = 0"),
    ("owner case-sensitive", "if owner.lower() not in owners:", "if owner not in owners:"),
    ("unparseable not blocking", '"MALFORMED_IDENTITY", "UNPARSEABLE", "UNREAD", "TOPOLOGY_INCOMPLETE"}',
     '"MALFORMED_IDENTITY", "UNREAD", "TOPOLOGY_INCOMPLETE"}'),
    ("unread not blocking", '"MALFORMED_IDENTITY", "UNPARSEABLE", "UNREAD", "TOPOLOGY_INCOMPLETE"}',
     '"MALFORMED_IDENTITY", "UNPARSEABLE", "TOPOLOGY_INCOMPLETE"}'),
    ("alias conflict ignored", "        report(\"ALIAS_CONFLICT\", f\"the repository fields disagree: {', '.join(repos)}\")\n        return",
     "        pass"),
    ("malformed identity silent", 'report("MALFORMED_IDENTITY", f"{k} = {str(obj[k])[:60]!r} is not a valid value")',
     '(lambda *a: None)("MALFORMED_IDENTITY", f"{k} = {str(obj[k])[:60]!r} is not a valid value")'),
    ("only the first commit checked", "    for commit in commits or [None]:", "    for commit in commits[:1] or [None]:"),
    ("abbreviated commits rejected", 'COMMITISH = re.compile(r"^[0-9a-f]{7,40}$")', 'COMMITISH = re.compile(r"^[0-9a-f]{40}$")'),
    ("named refs malformed", '        elif k == "ref" and isinstance(value, str) and REF_NAME.match(value):', "        elif False:"),
    ("named refs not references", "    if not (commits or named or blobs or sha256s or sizes):",
     "    if not (commits or blobs or sha256s or sizes):"),
    ("path-only records checked", "    if not (commits or named or blobs or sha256s or sizes):\n        return",
     "    if False:\n        return"),
    ("json comments not read", '        return [json.loads(strip_json_comments(text))], "", True', "        raise ValueError"),
    ("comment stripping ignores strings", "        if text[i] == '\"':\n            j = i + 1", "        if False:\n            j = i + 1"),
    ("trailing commas kept", "                continue  # a trailing comma", "                pass  # a trailing comma"),
    ("json shape truncated", "            for child in v:\n                walk(child", "            for child in v[:50]:\n                walk(child"),
    ("file identity on a directory accepted",
     '        problems.append(f"a file identity is recorded, but the path is a {kind}")', "        pass"),
    ("unpinned repositories get no url", '    url = f"https://github.com/{full}.git" if "/" in str(full) else ""', '    url = ""'),
    ("template files read", "or parts[-1].startswith(TEMPLATE_PREFIX)", "or False"),
    ("candidate ignored", "            node.commit, node.candidate = sha, True", "            node.candidate = True"),
    ("candidate missing silent", 'findings.append(finding("MISSING_PIN", node.name, "", f"candidate revision',
     '(finding("MISSING_PIN", node.name, "", f"candidate revision'),
    ("pin skew reported for candidates", "        if node.pinned and not node.candidate:", "        if node.pinned:"),
    ("unknown candidate accepted",
     '            raise CrawlError(f"--at names {name!r}, which is not a crawled repository (use --workspace-rev for the workspace)")',
     "            pass"),
    ("bad pins schema accepted",
     'if not pins or pins.get("schema") != "ulw.pins.v1" or not isinstance(pins.get("repos"), list):',
     'if not pins or not isinstance(pins.get("repos"), list):'),
    ("lean namespace prefix dropped", '        prefix = ".".join(n for kw, n in scopes if kw == "namespace" and n)',
     '        prefix = ""'),
    ("changed entries dropped from diffs", "    changed = sorted(k for k in set(old) & set(new) if old[k] != new[k])",
     "    changed = []"),
    ("unregistered pinned repositories not crawled", "        if node.available:\n            references.extend(collect(",
     "        if node.available and node.registered:\n            references.extend(collect("),
    ("repository name case-sensitive",
     "target = nodes.get(name) or next((n for k, n in nodes.items() if k.lower() == name.lower()), None)",
     "target = nodes.get(name)"),
    ("acknowledgements ignored", ' and "acknowledged" not in row)', ")"),
    ("acknowledgement constraints treated as live references",
     "    if isinstance(obj, dict):\n        repos = distinct(obj, REPO_KEYS,",
     "    if isinstance(obj, dict):\n        if obj.get('cited_repo') and obj.get('cited_path') and obj.get('cited_commit'):\n"
     "            obj = dict(obj, repository=obj['cited_repo'], path=obj['cited_path'], commit=obj['cited_commit'])\n"
     "        repos = distinct(obj, REPO_KEYS,"),
    ("acknowledgement narrowing ignored",
     '        if all(entry[entry_key] == row.get(row_key) for entry_key, row_key in constraints if entry_key in entry):',
     "        if True:"),
    ("acknowledgement commit not compared",
     'for entry_key, row_key in constraints if entry_key in entry)',
     'for entry_key, row_key in constraints if entry_key in entry and entry_key != "cited_commit")'),
    ("directory acknowledgement crosses its boundary", 'cited.startswith(source) if source.endswith("/")',
     'cited.startswith(source.rstrip("/")) if source.endswith("/")'),
    ("every acknowledgement source is a prefix",
     'if not (cited.startswith(source) if source.endswith("/") else cited == source):', "if not cited.startswith(source):"),
    ("unused acknowledgements hidden", '"unused_entries": sorted(set(range(len(entries))) - used)}', '"unused_entries": []}'),
    ("acknowledgement file hash dropped", '"acknowledgements": {"sha256": (acknowledgements or {}).get("sha256"),',
     '"acknowledgements": {"sha256": None,'),
    ("unknown acknowledgement keys accepted", 'and not set(entry) - {*ACK_KEYS, "reason"} and "reason" in entry',
     'and True and "reason" in entry'),
    ("repeated acknowledgement accepted", "        if key in seen:\n", "        if False:\n"),
    ("acknowledgement schema unchecked", 'doc.get("schema") != ACK_SCHEMA or ', ""),
    ("blank acknowledgement reason accepted", "isinstance(v, str) and v.strip() for v", "isinstance(v, str) for v"),
    ("blocks field not recorded", '        row["blocks"] = blocks(row)', '        row["blocks"] = row["blocking"]'),
    ("blocking findings not listed", '        if row["blocks"]:\n            print(f"BLOCKING',
     '        if False:\n            print(f"BLOCKING'),
    ("removed interface files not drift",
     '    return row["class"] == "FROZEN_REMOVED" and row["path"].endswith(INTERFACE_SUFFIXES)', "    return False"),
    ("every removed file is drift", ' and row["path"].endswith(INTERFACE_SUFFIXES)', ""),
]


# The tests written for each mutant's behaviour. A kill needs one of them to fail by assertion.
DESIGNATED = {
    "abbreviated commits rejected": ("test_named_refs_and_abbreviated_commits_are_checked_and_path_only_records_are_not_references",),
    "acknowledgement commit not compared": ("test_acknowledged_drift_does_not_block_under_strict",),
    "acknowledgement constraints treated as live references": ("test_acknowledgement_constraints_are_not_live_reference_records",),
    "acknowledgement file hash dropped": ("test_acknowledged_drift_does_not_block_under_strict",),
    "acknowledgement narrowing ignored": ("test_acknowledged_drift_does_not_block_under_strict",),
    "acknowledgement schema unchecked": ("test_malformed_acknowledgements_stop_the_run_before_crawling",),
    "acknowledgements ignored": ("test_acknowledged_drift_does_not_block_under_strict",),
    "alias conflict ignored": ("test_record_aliases_commits_and_refs",),
    "backslash allowed in url paths": ("test_url_paths_keep_balanced_parentheses_and_stop_at_backslashes",),
    "bad pins schema accepted": ("test_unreadable_workspace_pins_exit_2",),
    "balanced parentheses dropped": ("test_url_paths_keep_balanced_parentheses_and_stop_at_backslashes",),
    "binary count hidden in markdown": ("test_binary_files_are_counted_and_oversize_files_reported",),
    "binary files not counted": ("test_binary_files_are_counted_and_oversize_files_reported",),
    "blank acknowledgement reason accepted": ("test_malformed_acknowledgements_stop_the_run_before_crawling",),
    "blob identity ignored": ("test_identity_failures_block", "test_every_recorded_identity_key_is_checked",),
    "blocking findings not listed": ("test_acknowledged_drift_does_not_block_under_strict",),
    "blocking ignored at exit": ("test_identity_failures_block", "test_missing_path_commit_and_repository_block",),
    "blocks field not recorded": ("test_acknowledged_drift_does_not_block_under_strict",),
    "byte-identical claims ignored": ("test_byte_identical_claims_are_checked_at_their_commit",),
    "byte-identical claims without identity accepted": ("test_byte_identical_values_are_validated",),
    "byte_length ignored": ("test_every_recorded_identity_key_is_checked",),
    "bytes ignored": ("test_identity_failures_block", "test_excerpt_references_are_checked_against_their_lines",),
    "candidate ignored": ("test_candidate_head_is_crawled_in_place_of_the_pin",),
    "candidate missing silent": ("test_candidate_revision_must_exist_and_name_a_crawled_repository",),
    "canonical input silent": ("test_incomplete_topology_inputs_block",),
    "changed entries dropped from diffs": ("test_json_shape_and_diff",),
    "comment stripping ignores strings": ("test_json_with_comments_and_trailing_commas_is_read",),
    "directory acknowledgement crosses its boundary": ("test_a_directory_source_acknowledges_only_files_under_it",),
    "every acknowledgement source is a prefix": ("test_a_directory_source_acknowledges_only_files_under_it",),
    "every removed file is drift": ("test_strict_blocks_on_a_removed_interface_file_unless_acknowledged",),
    "excerpt checked as whole file": ("test_excerpt_references_are_checked_against_their_lines",),
    "excerpt identity unchecked": ("test_excerpt_references_are_checked_against_their_lines",),
    "excerpt lines never read": ("test_excerpt_references_are_checked_against_their_lines",),
    "excerpt off by one": ("test_excerpt_references_are_checked_against_their_lines",),
    "excerpt range unchecked": ("test_excerpt_references_are_checked_against_their_lines",),
    "excerpt text ignored": ("test_excerpt_references_are_checked_against_their_lines",),
    "extra blob keys ignored": ("test_every_recorded_identity_key_is_checked",),
    "extra commit keys ignored": ("test_every_commit_key_pins_the_reference",),
    "file identity on a directory accepted": ("test_file_identity_on_a_directory_is_a_mismatch",),
    "fixture match includes file names": ("test_fixture_paths_are_directory_segments_only",),
    "fixtures not skipped": ("test_fixture_files_are_skipped_and_counted_unless_included",),
    "foreign owners included": ("test_text_references_filter_owner_and_strip_punctuation",),
    "foreign reference silent": ("test_clean_federation_reports_interface_drift_without_blocking",),
    "frozen change unreported": ("test_clean_federation_reports_interface_drift_without_blocking",),
    "include-fixtures ignored": ("test_fixture_files_are_skipped_and_counted_unless_included", "test_template_files_are_synthetic",),
    "json comments not read": ("test_json_with_comments_and_trailing_commas_is_read",),
    "json lines not parsed": ("test_every_text_file_is_read_whatever_its_extension",),
    "json shape shallow": ("test_clean_federation_reports_interface_drift_without_blocking",),
    "json shape truncated": ("test_json_shape_walks_every_array_element",),
    "json strings not decoded": ("test_json_urls_are_matched_in_decoded_strings",),
    "lean namespace prefix dropped": ("test_lean_statements_with_namespace_multiline_and_attributes",),
    "lean statements blind": ("test_clean_federation_reports_interface_drift_without_blocking",),
    "line comments kept": ("test_nested_block_and_line_comments_are_removed_but_strings_kept",),
    "live drift misclassified": ("test_live_reference_drift_blocks",),
    "unknown acknowledgement keys accepted": (
        "test_malformed_acknowledgements_stop_the_run_before_crawling",
        "test_registered_acknowledgements_conform_to_current_schema",
    ),
    "malformed byte-identical values silent": ("test_byte_identical_values_are_validated",),
    "malformed identity silent": ("test_records_with_an_identity_are_references_and_malformed_values_are_reported",),
    "missing commit misclassified": ("test_missing_path_commit_and_repository_block",),
    "missing path silent": ("test_missing_path_commit_and_repository_block",),
    "missing pin silent": ("test_missing_pinned_commit_blocks",),
    "missing workspace registry silent": ("test_incomplete_topology_inputs_block",),
    "named refs malformed": ("test_named_refs_and_abbreviated_commits_are_checked_and_path_only_records_are_not_references",),
    "named refs not references": ("test_named_refs_and_abbreviated_commits_are_checked_and_path_only_records_are_not_references",),
    "nested comments not tracked": ("test_nested_block_and_line_comments_are_removed_but_strings_kept",),
    "only the first commit checked": ("test_record_aliases_commits_and_refs",),
    "only the first recorded value checked": ("test_every_recorded_identity_key_is_checked",),
    "oversize file silent": ("test_binary_files_are_counted_and_oversize_files_reported",),
    "owner case-sensitive": ("test_owner_and_repository_names_match_case_insensitively",),
    "path-only records checked": ("test_named_refs_and_abbreviated_commits_are_checked_and_path_only_records_are_not_references",),
    "pin skew reported for candidates": ("test_candidate_head_is_crawled_in_place_of_the_pin",),
    "pin skew silent": ("test_pin_skew_reported_and_optionally_blocking",),
    "python signatures drop arguments": ("test_python_signatures",),
    "raw_sha256 ignored": ("test_every_recorded_identity_key_is_checked",),
    "registry document missing silent": ("test_incomplete_topology_inputs_block",),
    "registry entry silent": ("test_incomplete_topology_inputs_block",),
    "registry host commit silent": ("test_incomplete_topology_inputs_block",),
    "registry host silent": ("test_incomplete_topology_inputs_block",),
    "registry host unreachable silent": ("test_incomplete_topology_inputs_block",),
    "registry map silent": ("test_incomplete_topology_inputs_block",),
    "removed interface files not drift": ("test_strict_blocks_on_a_removed_interface_file_unless_acknowledged",),
    "repeated acknowledgement accepted": ("test_malformed_acknowledgements_stop_the_run_before_crawling",),
    "repositories list silent": ("test_incomplete_topology_inputs_block",),
    "repository name case-sensitive": ("test_owner_and_repository_names_match_case_insensitively",),
    "require-current-pins ignored": ("test_pin_skew_reported_and_optionally_blocking",),
    "same-name foreign repository conflated": ("test_owner_and_repository_names_match_case_insensitively", "test_clean_federation_reports_interface_drift_without_blocking",),
    "sha256 ignored": ("test_identity_failures_block", "test_excerpt_references_are_checked_against_their_lines",),
    "size limit ignored": ("test_binary_files_are_counted_and_oversize_files_reported",),
    "skip count hidden in markdown": ("test_fixture_files_are_skipped_and_counted_unless_included",),
    "skipped fixtures not counted": ("test_fixture_files_are_skipped_and_counted_unless_included",),
    "strict ignored": ("test_strict_blocks_on_interface_drift",),
    "suffix filter restored": ("test_every_text_file_is_read_whatever_its_extension",),
    "template files read": ("test_template_files_are_synthetic",),
    "topology incomplete not blocking": ("test_incomplete_topology_inputs_block",),
    "trailing commas kept": ("test_json_with_comments_and_trailing_commas_is_read",),
    "unknown candidate accepted": ("test_candidate_revision_must_exist_and_name_a_crawled_repository",),
    "unknown repository ignored": ("test_missing_path_commit_and_repository_block", "test_rename_adapts_traversal_and_reports_stale_names",),
    "unparseable not blocking": ("test_unparseable_json_blocks_as_an_incomplete_census",),
    "unparseable silent": ("test_unparseable_json_blocks_as_an_incomplete_census", "test_json_urls_are_matched_in_decoded_strings",),
    "unparsed json lines dropped": ("test_every_text_file_is_read_whatever_its_extension",),
    "unpinned not reported": ("test_registered_but_unpinned_repository_is_crawled_and_reported",),
    "unpinned repositories get no url": ("test_registered_unpinned_repository_resolves_through_the_cache",),
    "unreachable repository silent": ("test_unreachable_pinned_repository_blocks_but_does_not_halt", "test_stale_registry_entry_is_reported_without_halting",),
    "unreachable target not blocking": ("test_stale_registry_entry_is_reported_without_halting",),
    "unread not blocking": ("test_binary_files_are_counted_and_oversize_files_reported",),
    "unregistered pinned repositories not crawled": ("test_added_repository_is_crawled",),
    "unterminated comment accepted": ("test_unterminated_comment_is_unparseable_and_its_tail_scanned",),
    "unused acknowledgements hidden": ("test_acknowledged_drift_does_not_block_under_strict",),
    "url path not decoded": ("test_percent_encoded_url_paths_are_decoded",),
    "workspace registry shape silent": ("test_incomplete_topology_inputs_block",),
}


# The test population this inventory was designed and run against. Any change to the suite
# (a test added, removed or renamed, or any edit to the file) must update these deliberately,
# together with DESIGNATED and the evidence fail-map.
TEST_SHA256 = "e93af5bbc71ff07ff3a0f35370a0f118521b4c600f634b111e035cf118d3123f"
EXPECTED_TESTS = (
    "test_a_directory_source_acknowledges_only_files_under_it",
    "test_acknowledgement_constraints_are_not_live_reference_records",
    "test_acknowledged_drift_does_not_block_under_strict",
    "test_added_repository_is_crawled",
    "test_binary_files_are_counted_and_oversize_files_reported",
    "test_byte_identical_claims_are_checked_at_their_commit",
    "test_byte_identical_values_are_validated",
    "test_candidate_head_is_crawled_in_place_of_the_pin",
    "test_candidate_revision_must_exist_and_name_a_crawled_repository",
    "test_clean_federation_reports_interface_drift_without_blocking",
    "test_every_commit_key_pins_the_reference",
    "test_every_recorded_identity_key_is_checked",
    "test_every_text_file_is_read_whatever_its_extension",
    "test_excerpt_references_are_checked_against_their_lines",
    "test_file_identity_on_a_directory_is_a_mismatch",
    "test_fixture_files_are_skipped_and_counted_unless_included",
    "test_fixture_paths_are_directory_segments_only",
    "test_identity_failures_block",
    "test_incomplete_topology_inputs_block",
    "test_json_shape_and_diff",
    "test_json_shape_walks_every_array_element",
    "test_json_urls_are_matched_in_decoded_strings",
    "test_json_with_comments_and_trailing_commas_is_read",
    "test_lean_statements_with_namespace_multiline_and_attributes",
    "test_live_reference_drift_blocks",
    "test_malformed_acknowledgements_stop_the_run_before_crawling",
    "test_missing_path_commit_and_repository_block",
    "test_missing_pinned_commit_blocks",
    "test_named_refs_and_abbreviated_commits_are_checked_and_path_only_records_are_not_references",
    "test_nested_block_and_line_comments_are_removed_but_strings_kept",
    "test_owner_and_repository_names_match_case_insensitively",
    "test_percent_encoded_url_paths_are_decoded",
    "test_pin_skew_reported_and_optionally_blocking",
    "test_python_signatures",
    "test_record_aliases_commits_and_refs",
    "test_registered_acknowledgements_conform_to_current_schema",
    "test_records_with_an_identity_are_references_and_malformed_values_are_reported",
    "test_registered_but_unpinned_repository_is_crawled_and_reported",
    "test_registered_unpinned_repository_resolves_through_the_cache",
    "test_rename_adapts_traversal_and_reports_stale_names",
    "test_stale_registry_entry_is_reported_without_halting",
    "test_strict_blocks_on_a_removed_interface_file_unless_acknowledged",
    "test_strict_blocks_on_interface_drift",
    "test_template_files_are_synthetic",
    "test_text_references_filter_owner_and_strip_punctuation",
    "test_unparseable_json_blocks_as_an_incomplete_census",
    "test_unreachable_pinned_repository_blocks_but_does_not_halt",
    "test_unreadable_workspace_pins_exit_2",
    "test_unterminated_comment_is_unparseable_and_its_tail_scanned",
    "test_url_paths_keep_balanced_parentheses_and_stop_at_backslashes",
)

GRADER_MUTANTS = (
    (
        "authorize an assertion through any intervening frame",
        'and all(is_oracle(' + 'code) for code in codes[positions[0] + 1:]))',
        "and True)",
        "test_subject_assertion_is_not_an_authorized_kill",
    ),
    (
        "compare oracle code by equality",
        'any(code is ' + 'allowed for allowed in oracle_codes)',
        "any(code == allowed for allowed in oracle_codes)",
        "test_equal_but_distinct_unittest_code_is_not_authorized",
    ),
    (
        "accept skipped tests",
        'report["skip_' + 'count"] or',
        "False or",
        "test_skip_and_expected_failure_are_not_a_pass",
    ),
    (
        "accept expected failures",
        'report["expected_failure_' + 'count"] or',
        "False or",
        "test_skip_and_expected_failure_are_not_a_pass",
    ),
    (
        "accept unexpected successes",
        'report["unexpected_success_' + 'count"]',
        "False",
        "test_skip_and_expected_failure_are_not_a_pass",
    ),
    (
        "drop child optimization",
        'optimize = ["-" + "O" * ' + 'sys.flags.optimize] if sys.flags.optimize else []',
        "optimize = []",
        "test_optimized_nested_parents_pass_mode_to_children",
    ),
    (
        "ignore the declared population",
        'set(names) != ' + 'set(expected_names)',
        "False",
        "test_declared_population_mismatch_is_invalid",
    ),
    (
        "ignore the fail-map provenance schema",
        'if doc.get(' + '"schema") != schema:',
        "if False:",
        "test_failmap_requires_origin_schema_and_provenance",
    ),
    (
        "ignore the fail-map execution binding",
        'execution != ' + 'execution_binding()',
        "False",
        "test_failmap_requires_exact_execution_binding",
    ),
    (
        "accept fail-map errors",
        'elif field == ' + '"error" and ids:',
        "elif False:",
        "test_failmap_rejects_errors_duplicates_and_bad_origin_shape",
    ),
    (
        "accept duplicate fail-map ids",
        'elif len(ids) != ' + 'len(set(ids)):',
        "elif False:",
        "test_failmap_rejects_errors_duplicates_and_bad_origin_shape",
    ),
    (
        "accept boolean source lines",
        'type(selected.get("firstlineno")) is ' + 'not int',
        "False",
        "test_failmap_rejects_errors_duplicates_and_bad_origin_shape",
    ),
    (
        "ignore selected-frame line identity",
        'frames[positions[0]]["firstlineno"] != ' + 'selected.get("firstlineno")',
        "False",
        "test_failmap_rejects_errors_duplicates_and_bad_origin_shape",
    ),
    (
        "ignore exception type",
        'event.get("exception") != ' + '"AssertionError"',
        "False",
        "test_failmap_rejects_errors_duplicates_and_bad_origin_shape",
    ),
    (
        "ignore extra error events",
        '            else:\n                problems.append(f"{name}: provenance[{index}] records a ' +
        'non-failure event")',
        "            else:\n                pass",
        "test_failmap_rejects_errors_duplicates_and_bad_origin_shape",
    ),
    (
        "accept explicitly unauthorized assertion provenance",
        '            elif event.get("kind") == ' + '"failure":\n                problems.append(f"{name}: provenance[{index}] records an unauthorized assertion failure")',
        "            elif False:\n                problems.append(f\"{name}: provenance[{index}] records an unauthorized assertion failure\")",
        "test_failmap_rejects_unauthorized_provenance",
    ),
)


def run_suite(tool_dir, expected_names=None, *, selector=None, copy_test=True):
    """Run a bounded test population against the tool in tool_dir."""
    global LAST_ORIGIN_REPORT
    LAST_ORIGIN_REPORT = None
    work = Path(tool_dir)
    (work / "tests").mkdir(exist_ok=True)
    if copy_test:
        shutil.copy(TEST, work / "tests" / TEST.name)
    optimize = ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
    env = dict(os.environ)
    env.pop("PYTHONOPTIMIZE", None)
    proc = subprocess.run([sys.executable, "-B", *optimize, "-S", "-c", ORIGIN_RUNNER,
                           selector or f"tests.{TEST.stem}"], cwd=work, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    encoded = [line[len(ORIGIN_SENTINEL):] for line in proc.stdout.splitlines()
               if line.startswith(ORIGIN_SENTINEL)]
    if proc.returncode != 0 or len(encoded) != 1:
        return "invalid", f"runner_exit={proc.returncode}", set(), set()
    try:
        report = json.loads(encoded[0])
    except (TypeError, ValueError):
        return "invalid", "invalid origin report", set(), set()
    LAST_ORIGIN_REPORT = report
    if type(report) is not dict:
        return "invalid", "invalid origin report", set(), set()
    counts = ("optimize", "tests_run", "failure_count", "error_count", "skip_count",
              "expected_failure_count", "unexpected_success_count")
    if (report.get("schema") != "reference-crawler-origin-v3" or
            any(type(report.get(key)) is not int for key in counts) or
            report["optimize"] != sys.flags.optimize):
        return "invalid", "invalid origin counts or mode", set(), set()
    ran = report["tests_run"]
    ids, names, events = report.get("ids"), report.get("names"), report.get("events")
    if (expected_names is not None and
            (type(expected_names) not in (list, tuple) or
             any(type(name) is not str for name in expected_names) or
             len(set(expected_names)) != len(expected_names))):
        return "invalid", "invalid expected test population", set(), set()
    if (type(ids) is not list or type(names) is not list or type(events) is not list or
            not ids or any(type(x) is not str for x in ids + names) or
            len(ids) != ran or len(names) != ran or len(set(ids)) != ran or
            len(set(names)) != ran or report.get("started") != ids or report.get("stopped") != ids or
            (expected_names is not None and
             (len(names) != len(expected_names) or set(names) != set(expected_names))) or
            report["skip_count"] or report["expected_failure_count"] or report["unexpected_success_count"]):
        return "invalid", "incomplete, skipped or unexpected test population", set(), set()
    if (any(type(e) is not dict or e.get("kind") not in ("failure", "error") or
            e.get("test") not in names or type(e.get("authorized")) is not bool for e in events) or
            sum(e["kind"] == "failure" for e in events) != report["failure_count"] or
            sum(e["kind"] == "error" for e in events) != report["error_count"]):
        return "invalid", "invalid origin events", set(), set()
    failed = {e["test"] for e in events if e["kind"] == "failure" and e["authorized"]}
    unauthorized = {e["test"] for e in events if e["kind"] == "failure" and not e["authorized"]}
    errored = {e["test"] for e in events if e["kind"] == "error"}
    if (report.get("authorized_failures") != sorted(failed) or
            report.get("unauthorized_failures") != sorted(unauthorized) or
            report.get("errors") != sorted(errored)):
        return "invalid", "inconsistent origin events", set(), set()
    detail = f"failures={len(failed)}, errors={len(errored)}, unauthorized={len(unauthorized)}"
    if errored or unauthorized:
        return "invalid", detail, set(), errored
    if failed:
        return "fail", detail, failed, errored
    return "pass", f"ran={ran}", set(), errored


def with_tool(source):
    tmp = tempfile.TemporaryDirectory()
    (Path(tmp.name) / "tools" / "cbz").mkdir(parents=True)
    (Path(tmp.name) / "tools" / "cbz" / "reference_crawler.py").write_text(source)
    shutil.copy(ACKNOWLEDGEMENTS, Path(tmp.name) / "tools" / "cbz" / ACKNOWLEDGEMENTS.name)
    return tmp


def population_problems(test_bytes):
    """Every way the test file's raw bytes differ from the frozen population; empty when they match."""
    names = TEST_NAME.findall(test_bytes.decode("utf-8"))  # bytes.decode keeps CRLF; read_text would not
    problems = [f"duplicate test name {n}" for n in sorted({n for n in names if names.count(n) > 1})]
    problems += [f"added test {n}" for n in sorted(set(names) - set(EXPECTED_TESTS))]
    problems += [f"removed test {n}" for n in sorted(set(EXPECTED_TESTS) - set(names))]
    if hashlib.sha256(test_bytes).hexdigest() != TEST_SHA256:
        problems.append("test file bytes differ from TEST_SHA256")
    return problems


def designation_problems(designated, mutant_names):
    """Every way the designation map fails to bind each mutant, and to cover each frozen test."""
    problems = []
    if sorted(mutant_names) != sorted(designated) or len(set(mutant_names)) != len(mutant_names):
        problems.append(f"mutants without designations: {sorted(set(mutant_names) - set(designated))}, "
                        f"designations without mutants: {sorted(set(designated) - set(mutant_names))}")
    problems += [f"{n}: no designated test" for n in sorted(n for n, tests in designated.items() if not tests)]
    named = {t for tests in designated.values() for t in tests}
    problems += [f"unknown designated test {t}" for t in sorted(named - set(EXPECTED_TESTS))]
    problems += [f"test {t} is designated by no mutant" for t in sorted(set(EXPECTED_TESTS) - named)]
    return problems


def execution_binding():
    """Identify the local source and Python mode that produced or consumes a v3 map."""
    return {"optimize": sys.flags.optimize,
            "python_version": sys.version,
            "crawler_sha256": hashlib.sha256(TOOL.read_bytes()).hexdigest(),
            "grader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def failmap_problems(doc, expected_names=None, schema="reference-crawler-mutant-failmap/v3"):
    """Every way an evidence fail-map disagrees with the frozen population and DESIGNATED."""
    if type(doc) is not dict:
        return ["the fail-map is not an object"]
    expected_names = set(DESIGNATED) if expected_names is None else set(expected_names)
    population = doc.get("population")
    problems = []
    if doc.get("schema") != schema:
        problems.append(f"the fail-map schema is not {schema}")
    execution = doc.get("execution")
    if (type(execution) is not dict or type(execution.get("optimize")) is not int or
            execution != execution_binding()):
        problems.append("the fail-map execution binding differs from this source or mode")
    if (type(population) is not dict or population.get("test_file_sha256") != TEST_SHA256 or
            type(population.get("tests")) is not list or tuple(population["tests"]) != EXPECTED_TESTS):
        problems.append("the fail-map records a different test population")
    mutants = doc.get("mutants")
    if type(mutants) is not dict:
        return problems + ["the fail-map mutants field is not an object"]
    if set(mutants) != expected_names:
        problems.append("the fail-map covers a different mutant set")
    for name, entry in sorted(mutants.items()):
        if not isinstance(entry, dict):
            problems.append(f"{name}: the fail-map entry is not an object")
            continue
        for field in ("fail", "error"):
            ids = entry.get(field)
            if type(ids) is not list or not all(type(t) is str for t in ids):
                problems.append(f"{name}: {field} is not a list of test names")
            elif len(ids) != len(set(ids)):
                problems.append(f"{name}: duplicate test ids in {field}")
            elif set(ids) - set(EXPECTED_TESTS):
                problems.append(f"{name}: unknown test ids in {field}: {sorted(set(ids) - set(EXPECTED_TESTS))}")
            elif field == "error" and ids:
                problems.append(f"{name}: execution reported errors")
        provenance = entry.get("provenance")
        if not isinstance(provenance, list):
            problems.append(f"{name}: provenance is not a list")
            provenance = []
        authorized = set()
        for index, event in enumerate(provenance):
            if not isinstance(event, dict):
                problems.append(f"{name}: provenance[{index}] is not an object")
                continue
            test = event.get("test")
            if test not in EXPECTED_TESTS:
                problems.append(f"{name}: provenance[{index}] names unknown test {test!r}")
            if event.get("kind") == "failure" and event.get("authorized") is True:
                selected = event.get("selected")
                frames = event.get("frames")
                if (type(selected) is not dict or type(selected.get("filename")) is not str
                        or not selected.get("filename") or type(selected.get("firstlineno")) is not int
                        or selected["firstlineno"] < 1 or event.get("exception") != "AssertionError"
                        or selected.get("name") != test):
                    problems.append(f"{name}: provenance[{index}] lacks the selected test code identity")
                if (type(frames) is not list or not frames or any(
                        type(frame) is not dict or type(frame.get("filename")) is not str or
                        not frame.get("filename") or type(frame.get("name")) is not str or
                        type(frame.get("firstlineno")) is not int or frame["firstlineno"] < 1 or
                        type(frame.get("lineno")) is not int or frame["lineno"] < 1 or
                        type(frame.get("selected")) is not bool or type(frame.get("oracle")) is not bool
                        for frame in frames)):
                    problems.append(f"{name}: provenance[{index}] lacks selected-frame traceback provenance")
                else:
                    positions = [i for i, frame in enumerate(frames) if frame["selected"]]
                    if (len(positions) != 1 or type(selected) is not dict or
                            frames[positions[0]]["filename"] != selected.get("filename") or
                            frames[positions[0]]["firstlineno"] != selected.get("firstlineno") or
                            frames[positions[0]]["name"] != test or
                            not all(frame["oracle"] for frame in frames[positions[0] + 1:])):
                        problems.append(f"{name}: provenance[{index}] has inconsistent selected-frame identity")
                if type(test) is str:
                    authorized.add(test)
            elif event.get("kind") == "failure":
                problems.append(f"{name}: provenance[{index}] records an unauthorized assertion failure")
            else:
                problems.append(f"{name}: provenance[{index}] records a non-failure event")
        fail_ids = entry.get("fail")
        if type(fail_ids) is list and all(type(t) is str for t in fail_ids):
            if set(fail_ids) != authorized:
                problems.append(f"{name}: assertion failures and authorized provenance differ")
    for name, tests in DESIGNATED.items():
        if name not in expected_names:
            continue
        entry = mutants.get(name)
        if isinstance(entry, dict):
            ids = entry.get("fail")
            if (type(ids) is not list or any(type(t) is not str for t in ids) or
                    not set(tests) <= set(ids)):
                problems.append(f"{name}: a designated test is not among the fail-map's assertion failures")
    return problems


def mutants_for_shard(mutants, index, count):
    """Return one deterministic, balanced zero-based shard of the mutant sequence."""
    if count < 1 or index < 0 or index >= count:
        raise ValueError(f"invalid shard {index}/{count}")
    return list(mutants[index::count])


def parse_shard(value):
    """Parse the user-facing one-based INDEX/COUNT form into a zero-based pair."""
    try:
        index_text, count_text = value.split("/", 1)
        index, count = int(index_text), int(count_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shard must be INDEX/COUNT, for example 1/3") from exc
    if count < 1 or index < 1 or index > count:
        raise argparse.ArgumentTypeError("shard INDEX must be between 1 and COUNT")
    return index - 1, count


def shard_failmap_problems(doc):
    """Validate one independently executable fail-map shard and its exact partition."""
    shard = doc.get("shard") or {}
    index, count = shard.get("index"), shard.get("count")
    if not isinstance(index, int) or not isinstance(count, int) or count < 1 or index < 0 or index >= count:
        return ["the fail-map has invalid shard metadata"]
    expected = [name for name, _, _ in mutants_for_shard(MUTANTS, index, count)]
    return failmap_problems(doc, expected, "reference-crawler-mutant-failmap-shard/v3")


def merge_shard_failmaps(paths):
    """Merge a complete, non-overlapping shard partition into one fully validated fail-map."""
    docs, problems = [], []
    for path in paths:
        try:
            doc = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"cannot read shard {path}: {exc}")
            continue
        docs.append(doc)
        problems += [f"{path}: {problem}" for problem in shard_failmap_problems(doc)]
    counts = {doc.get("shard", {}).get("count") for doc in docs}
    if len(counts) != 1 or not counts or not isinstance(next(iter(counts)), int):
        problems.append("shards do not declare one common count")
        count = 0
    else:
        count = next(iter(counts))
    indices = [doc.get("shard", {}).get("index") for doc in docs]
    if len(indices) != len(set(indices)):
        problems.append("duplicate shard index")
    missing = sorted(set(range(count)) - set(i for i in indices if isinstance(i, int)))
    if missing:
        problems.append(f"missing shard indices {missing}")
    merged_mutants = {}
    for doc in docs:
        for name, entry in (doc.get("mutants") or {}).items():
            if name in merged_mutants:
                problems.append(f"duplicate mutant {name!r} across shards")
            merged_mutants[name] = entry
    merged = {
        "schema": "reference-crawler-mutant-failmap/v3",
        "execution": execution_binding(),
        "population": {"test_file_sha256": TEST_SHA256, "tests": list(EXPECTED_TESTS)},
        "mutants": merged_mutants,
    }
    problems += failmap_problems(merged)
    return merged, problems


def grader_mutation_selftest():
    """Kill bounded mutations of this grader's assertion-origin and provenance policy."""
    source = Path(__file__).read_text()
    killed = 0
    for name, old, new, expected_test in GRADER_MUTANTS:
        if source.count(old) != 1 or old == new:
            print(f"GRADER NOT APPLICABLE {name} ({source.count(old)} matches)")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "tools" / "cbz" / Path(__file__).name
            target.parent.mkdir(parents=True)
            target.write_text(source)
            shutil.copy(TOOL, target.parent / TOOL.name)
            shutil.copy(ACKNOWLEDGEMENTS, target.parent / ACKNOWLEDGEMENTS.name)
            test_target = root / "tests" / GRADER_TEST.name
            test_target.parent.mkdir()
            shutil.copy(GRADER_TEST, test_target)
            selector = f"tests.{GRADER_TEST.stem}.AssertionOrigin.{expected_test}"
            baseline = run_suite(root, [expected_test], selector=selector, copy_test=False)
            target.write_text(source.replace(old, new, 1))
            mutated = run_suite(root, [expected_test], selector=selector, copy_test=False)
        if baseline[0] == "pass" and mutated[0] == "fail" and mutated[2] == {expected_test}:
            killed += 1
            print(f"GRADER KILLED {name} ({expected_test})")
        else:
            print(f"GRADER SURVIVED {name} (baseline={baseline[:2]}, mutant={mutated[:2]})")
    return killed, len(GRADER_MUTANTS)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Adverse mutation check for tools/cbz/reference_crawler.py")
    parser.add_argument("--failmap", type=Path, help="evidence fail-map to check against the frozen population")
    parser.add_argument("--write-failmap", type=Path, help="write this run's fail-map (population and per-mutant failures)")
    parser.add_argument("--shard", type=parse_shard, help="run one bounded one-based INDEX/COUNT mutant shard")
    parser.add_argument("--merge-failmaps", type=Path, nargs="+", metavar="SHARD",
                        help="merge a complete set of fail-map shards instead of running mutants")
    args = parser.parse_args(argv)
    if args.merge_failmaps is not None:
        if args.write_failmap is None or args.shard is not None or args.failmap is not None:
            parser.error("--merge-failmaps requires --write-failmap and cannot be combined with --shard or --failmap")
        merged, problems = merge_shard_failmaps(args.merge_failmaps)
        if problems:
            print(f"FAILMAP MERGE FAIL: {'; '.join(problems)}")
            return 2
        args.write_failmap.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
        print(f"FAILMAP MERGE OK: {len(merged['mutants'])}/{len(MUTANTS)} mutants in {args.write_failmap.name}")
        return 0
    source = TOOL.read_text()
    binding = execution_binding()
    test_bytes = TEST.read_bytes()
    problems = population_problems(test_bytes)
    if problems:
        print(f"POPULATION FAIL: {'; '.join(problems)}")
        return 2
    controls = {"an added test method": (test_bytes + b"\n    def test_population_control_extra(self):\n        pass\n",
                                         "added test test_population_control_extra"),
                "a deleted undesignated test method": (
                    test_bytes.replace(b"    def test_added_repository_is_crawled(", b"    def _deleted_control(", 1),
                    "removed test test_added_repository_is_crawled"),
                "a CRLF-only rewrite": (test_bytes.replace(b"\n", b"\r\n"), "test file bytes differ from TEST_SHA256")}
    for label, (data, expected) in controls.items():
        if expected not in population_problems(data):
            print(f"CONTROL FAIL: {label} was not refused by the population check")
            return 2
    grader_killed, grader_total = grader_mutation_selftest()
    if grader_killed != grader_total:
        print(f"GRADER MUTATION FAIL: {grader_killed}/{grader_total} killed")
        return 2
    print(f"GRADER MUTATIONS OK: {grader_killed}/{grader_total} killed")
    if args.failmap is not None:
        problems = failmap_problems(json.loads(args.failmap.read_text()))
        if problems:
            print(f"FAILMAP FAIL: {'; '.join(problems)}")
            return 2

    def baseline(when):
        with with_tool(source) as tmp:
            outcome, detail, _, _ = run_suite(tmp, EXPECTED_TESTS)
        if outcome != "pass" or detail != f"ran={len(EXPECTED_TESTS)}":
            print(f"BASELINE FAIL ({when}): the unmutated suite did not pass all {len(EXPECTED_TESTS)} frozen tests "
                  f"({outcome}: {detail}), so no kill can be trusted")
            return False
        return True

    if not baseline("before"):
        return 2
    with with_tool(source + "\ndef (:\n") as tmp:
        if run_suite(tmp, EXPECTED_TESTS)[0] != "invalid":
            print("CONTROL FAIL: a syntax-broken copy was not reported invalid")
            return 2
    if source.count(CRAWL_START) != 1:
        print("CONTROL FAIL: the error-only control's anchor is not unique")
        return 2
    with with_tool(source.replace(CRAWL_START, CRAWL_START.replace("    findings = []\n",
                                  "    raise RuntimeError(\"error-only control\")\n    findings = []\n"))) as tmp:
        outcome, detail, _, _ = run_suite(tmp, EXPECTED_TESTS)
        if outcome != "invalid" or "errors=" not in detail:
            print(f"CONTROL FAIL: an error-only copy was not reported invalid ({outcome}: {detail})")
            return 2
    names = [name for name, _, _ in MUTANTS]
    problems = designation_problems(DESIGNATED, names)
    if problems:
        print(f"DESIGNATION FAIL: {'; '.join(problems)}")
        return 2
    sole = next(t for t in EXPECTED_TESTS if sum(t in ts for ts in DESIGNATED.values()) == 1)
    owner = next(n for n, ts in DESIGNATED.items() if sole in ts)
    other = next(t for t in EXPECTED_TESTS if t != sole)
    uncovering = dict(DESIGNATED, **{owner: tuple(t for t in DESIGNATED[owner] if t != sole) or (other,)})
    if f"test {sole} is designated by no mutant" not in designation_problems(uncovering, names):
        print("CONTROL FAIL: a designation map that leaves a test uncovered was not refused")
        return 2
    selected_mutants = MUTANTS if args.shard is None else mutants_for_shard(MUTANTS, *args.shard)
    killed, observed = 0, {}
    for name, old, new in selected_mutants:
        if source.count(old) != 1:
            print(f"NOT APPLICABLE {name} ({source.count(old)} matches)")
            continue
        with with_tool(source.replace(old, new)) as tmp:
            outcome, detail, failed, errored = run_suite(tmp, EXPECTED_TESTS)
        observed[name] = {"fail": sorted(failed), "error": sorted(errored),
                          "provenance": (LAST_ORIGIN_REPORT or {}).get("events", [])}
        hit = sorted(failed & set(DESIGNATED[name]))
        if outcome == "fail" and hit:
            killed += 1
            print(f"KILLED   {name} ({detail}; designated {hit[0]} failed)")
        elif outcome == "fail":
            print(f"SURVIVED {name} ({detail}; no designated test failed, only {sorted(failed)[:3]})")
        else:
            print(f"{'INVALID ' if outcome == 'invalid' else 'SURVIVED'} {name}" + (f" ({detail})" if detail else ""))
    if not baseline("after"):
        return 2
    if execution_binding() != binding:
        print("SOURCE DRIFT: crawler, grader or Python mode changed during the campaign")
        return 2
    if args.write_failmap is not None:
        shard_mode = args.shard is not None
        doc = {"schema": ("reference-crawler-mutant-failmap-shard/v3" if shard_mode
                          else "reference-crawler-mutant-failmap/v3"),
               "execution": binding,
               "population": {"test_file_sha256": TEST_SHA256, "tests": list(EXPECTED_TESTS)},
               "mutants": observed}
        if shard_mode:
            doc["shard"] = {"index": args.shard[0], "count": args.shard[1]}
        problems = shard_failmap_problems(doc) if shard_mode else failmap_problems(doc)
        name = selected_mutants[0][0]
        for label, field, change in (("missing a designated failure", "fail", lambda ids: [t for t in ids if t not in DESIGNATED[name]]),
                                     ("with an unknown fail id", "fail", lambda ids: ids + ["test_not_in_the_suite"]),
                                     ("with an unknown error id", "error", lambda ids: ids + ["test_not_in_the_suite"]),
                                     ("whose error field is not a list", "error", lambda ids: "not a list")):
            corrupt = json.loads(json.dumps(doc))
            if name in corrupt["mutants"]:
                corrupt["mutants"][name][field] = change(corrupt["mutants"][name][field])
            if not failmap_problems(corrupt):
                print(f"CONTROL FAIL: a fail-map {label} was not refused")
                return 2
        args.write_failmap.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"FAILMAP {'OK' if not problems else 'FAIL: ' + '; '.join(problems)}: {args.write_failmap.name}")
    suffix = "" if args.shard is None else f" (shard {args.shard[0] + 1}/{args.shard[1]})"
    print(f"{killed}/{len(selected_mutants)} selected mutants killed{suffix}")
    return 0 if killed == len(selected_mutants) else 1


if __name__ == "__main__":
    sys.exit(main())
