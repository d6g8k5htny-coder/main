#!/usr/bin/env python3
"""Adverse mutation check for tools/cbz/reference_crawler.py (governance A8 / B-006).

Each mutant disables or corrupts one detection, policy rule, extractor or topology
behaviour. A mutant counts as killed only when tests/test_cbz_reference_crawler.py ran
at least one test and one of the mutant's designated tests (DESIGNATED, governance B-011)
failed by assertion. An error-only run is invalid, and a run in which only other tests
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
--write-failmap FILE, this run records which tests fail and error under each mutant,
writes that fail-map with the frozen population, and checks it the same way. Corrupted
copies of it (one designated failure removed, an unknown test id added to a fail or an
error list, or an error field that is not a list) must be refused. Two controls must come out invalid, never killed: a syntax-broken copy,
and an error-only copy whose crawl() raises. The unmutated suite
must pass both before and after the mutants. If the environment changes mid-run, so
that errors could pass for kills, the run is invalid (exit 2). Exit 0 only if every
mutant is killed.
Run from the repository root:
  python3 -B -S tools/cbz/reference_crawler_mutation_check.py [--failmap FILE] [--write-failmap FILE]
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "cbz" / "reference_crawler.py"
TEST = ROOT / "tests" / "test_cbz_reference_crawler.py"
RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
FAILED = re.compile(r"^FAILED \(([^)]*)\)$", re.M)
FAILURES = re.compile(r"\bfailures=(\d+)")
FAIL_NAME = re.compile(r"^FAIL: (\w+) ", re.M)
ERROR_NAME = re.compile(r"^ERROR: (\w+) ", re.M)
TEST_NAME = re.compile(r"^    def (test_\w+)\(", re.M)
CRAWL_START = "    findings = []\n    overrides = overrides or {}\n"
MUTANTS = [
    ("blob identity ignored", "for b in ref.blobs if b != oid]", "for b in ref.blobs if False]"),
    ("sha256 ignored", "for d in ref.sha256s if d != digest]", "for d in ref.sha256s if False]"),
    ("bytes ignored", "for n in ref.sizes if n != len(data)]", "for n in ref.sizes if False]"),
    ("missing path silent", 'findings.append(finding(cls, target.name, path, f"path absent at {commit}", ref))', "pass"),
    ("missing commit assumed present", "if not live and not mirror.ensure_commit(commit):", "if False:"),
    ("live drift misclassified", 'cls = "LIVE_DRIFT" if live else "IDENTITY_MISMATCH"', 'cls = "IDENTITY_MISMATCH"'),
    ("unknown repository ignored", 'findings.append(finding("UNKNOWN_REPOSITORY",', '(finding("UNKNOWN_REPOSITORY",'),
    ("foreign reference silent", 'findings.append(finding("FOREIGN_REPOSITORY",', '(finding("FOREIGN_REPOSITORY",'),
    ("same-name foreign repository conflated", "        target = None  # another owner's repository with the same short name",
     "        pass"),
    ("strict ignored", '(strict and row["class"] == "FROZEN_CHANGED" and interface_changed(row))', "False"),
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
    ("unreachable halts the crawl",
     'findings.append(finding("UNREACHABLE_REPOSITORY", node.name, "", str(exc)))', "raise"),
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
]


# The tests written for each mutant's behaviour. A kill needs one of them to fail by assertion.
DESIGNATED = {
    "abbreviated commits rejected": ("test_named_refs_and_abbreviated_commits_are_checked_and_path_only_records_are_not_references",),
    "alias conflict ignored": ("test_record_aliases_commits_and_refs",),
    "backslash allowed in url paths": ("test_url_paths_keep_balanced_parentheses_and_stop_at_backslashes",),
    "bad pins schema accepted": ("test_unreadable_workspace_pins_exit_2",),
    "balanced parentheses dropped": ("test_url_paths_keep_balanced_parentheses_and_stop_at_backslashes",),
    "binary count hidden in markdown": ("test_binary_files_are_counted_and_oversize_files_reported",),
    "binary files not counted": ("test_binary_files_are_counted_and_oversize_files_reported",),
    "blob identity ignored": ("test_identity_failures_block", "test_every_recorded_identity_key_is_checked",),
    "blocking ignored at exit": ("test_identity_failures_block", "test_missing_path_commit_and_repository_block",),
    "byte-identical claims ignored": ("test_byte_identical_claims_are_checked_at_their_commit",),
    "byte-identical claims without identity accepted": ("test_byte_identical_values_are_validated",),
    "byte_length ignored": ("test_every_recorded_identity_key_is_checked",),
    "bytes ignored": ("test_identity_failures_block", "test_excerpt_references_are_checked_against_their_lines",),
    "candidate ignored": ("test_candidate_head_is_crawled_in_place_of_the_pin",),
    "candidate missing silent": ("test_candidate_revision_must_exist_and_name_a_crawled_repository",),
    "canonical input silent": ("test_incomplete_topology_inputs_block",),
    "changed entries dropped from diffs": ("test_json_shape_and_diff",),
    "comment stripping ignores strings": ("test_json_with_comments_and_trailing_commas_is_read",),
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
    "malformed byte-identical values silent": ("test_byte_identical_values_are_validated",),
    "malformed identity silent": ("test_records_with_an_identity_are_references_and_malformed_values_are_reported",),
    "missing commit assumed present": ("test_missing_path_commit_and_repository_block",),
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
    "unreachable halts the crawl": ("test_unreachable_pinned_repository_blocks_but_does_not_halt", "test_stale_registry_entry_is_reported_without_halting",),
    "unreachable target not blocking": ("test_stale_registry_entry_is_reported_without_halting",),
    "unread not blocking": ("test_binary_files_are_counted_and_oversize_files_reported",),
    "unregistered pinned repositories not crawled": ("test_added_repository_is_crawled",),
    "unterminated comment accepted": ("test_unterminated_comment_is_unparseable_and_its_tail_scanned",),
    "url path not decoded": ("test_percent_encoded_url_paths_are_decoded",),
    "workspace registry shape silent": ("test_incomplete_topology_inputs_block",),
}


# The test population this inventory was designed and run against. Any change to the suite
# (a test added, removed or renamed, or any edit to the file) must update these deliberately,
# together with DESIGNATED and the evidence fail-map.
TEST_SHA256 = "94dbcab58c3160f709e7d8a0e820811b136609984cef84e68565e90a2d023668"
EXPECTED_TESTS = (
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
    "test_missing_path_commit_and_repository_block",
    "test_missing_pinned_commit_blocks",
    "test_named_refs_and_abbreviated_commits_are_checked_and_path_only_records_are_not_references",
    "test_nested_block_and_line_comments_are_removed_but_strings_kept",
    "test_owner_and_repository_names_match_case_insensitively",
    "test_percent_encoded_url_paths_are_decoded",
    "test_pin_skew_reported_and_optionally_blocking",
    "test_python_signatures",
    "test_record_aliases_commits_and_refs",
    "test_records_with_an_identity_are_references_and_malformed_values_are_reported",
    "test_registered_but_unpinned_repository_is_crawled_and_reported",
    "test_registered_unpinned_repository_resolves_through_the_cache",
    "test_rename_adapts_traversal_and_reports_stale_names",
    "test_stale_registry_entry_is_reported_without_halting",
    "test_strict_blocks_on_interface_drift",
    "test_template_files_are_synthetic",
    "test_text_references_filter_owner_and_strip_punctuation",
    "test_unparseable_json_blocks_as_an_incomplete_census",
    "test_unreachable_pinned_repository_blocks_but_does_not_halt",
    "test_unreadable_workspace_pins_exit_2",
    "test_unterminated_comment_is_unparseable_and_its_tail_scanned",
    "test_url_paths_keep_balanced_parentheses_and_stop_at_backslashes",
)


def run_suite(tool_dir):
    """Run the crawler tests against the copy of the tool in tool_dir."""
    work = Path(tool_dir)
    (work / "tests").mkdir(exist_ok=True)
    shutil.copy(TEST, work / "tests" / TEST.name)
    proc = subprocess.run([sys.executable, "-B", "-S", "-m", "unittest", f"tests.{TEST.stem}"],
                          cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ran = RAN.search(proc.stdout)
    errored = set(ERROR_NAME.findall(proc.stdout))
    if not ran or int(ran.group(1)) == 0:
        return "invalid", "", set(), errored
    if proc.returncode == 0 and "\nOK" in proc.stdout:
        return "pass", f"ran={ran.group(1)}", set(), errored
    failed = FAILED.search(proc.stdout)
    if not failed:
        return "invalid", "", set(), errored
    failures = FAILURES.search(failed.group(1))
    if not failures or int(failures.group(1)) == 0:
        return "invalid", failed.group(1), set(), errored  # errors alone are not evidence that a check caught the mutant
    return "fail", failed.group(1), set(FAIL_NAME.findall(proc.stdout)), errored


def with_tool(source):
    tmp = tempfile.TemporaryDirectory()
    (Path(tmp.name) / "tools" / "cbz").mkdir(parents=True)
    (Path(tmp.name) / "tools" / "cbz" / "reference_crawler.py").write_text(source)
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


def failmap_problems(doc):
    """Every way an evidence fail-map disagrees with the frozen population and DESIGNATED."""
    population = doc.get("population") or {}
    problems = []
    if population.get("test_file_sha256") != TEST_SHA256 or tuple(population.get("tests") or ()) != EXPECTED_TESTS:
        problems.append("the fail-map records a different test population")
    mutants = doc.get("mutants") or {}
    if sorted(mutants) != sorted(DESIGNATED):
        problems.append("the fail-map covers a different mutant set")
    for name, entry in sorted(mutants.items()):
        if not isinstance(entry, dict):
            problems.append(f"{name}: the fail-map entry is not an object")
            continue
        for field in ("fail", "error"):
            ids = entry.get(field)
            if not isinstance(ids, list) or not all(isinstance(t, str) for t in ids):
                problems.append(f"{name}: {field} is not a list of test names")
            elif set(ids) - set(EXPECTED_TESTS):
                problems.append(f"{name}: unknown test ids in {field}: {sorted(set(ids) - set(EXPECTED_TESTS))}")
    for name, tests in DESIGNATED.items():
        entry = mutants.get(name)
        if isinstance(entry, dict) and not set(tests) <= set(entry.get("fail") or ()):
            problems.append(f"{name}: a designated test is not among the fail-map's assertion failures")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description="Adverse mutation check for tools/cbz/reference_crawler.py")
    parser.add_argument("--failmap", type=Path, help="evidence fail-map to check against the frozen population")
    parser.add_argument("--write-failmap", type=Path, help="write this run's fail-map (population and per-mutant failures)")
    args = parser.parse_args(argv)
    source = TOOL.read_text()
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
    if args.failmap is not None:
        problems = failmap_problems(json.loads(args.failmap.read_text()))
        if problems:
            print(f"FAILMAP FAIL: {'; '.join(problems)}")
            return 2

    def baseline(when):
        with with_tool(source) as tmp:
            outcome, detail, _, _ = run_suite(tmp)
        if outcome != "pass" or detail != f"ran={len(EXPECTED_TESTS)}":
            print(f"BASELINE FAIL ({when}): the unmutated suite did not pass all {len(EXPECTED_TESTS)} frozen tests "
                  f"({outcome}: {detail}), so no kill can be trusted")
            return False
        return True

    if not baseline("before"):
        return 2
    with with_tool(source + "\ndef (:\n") as tmp:
        if run_suite(tmp)[0] != "invalid":
            print("CONTROL FAIL: a syntax-broken copy was not reported invalid")
            return 2
    if source.count(CRAWL_START) != 1:
        print("CONTROL FAIL: the error-only control's anchor is not unique")
        return 2
    with with_tool(source.replace(CRAWL_START, CRAWL_START.replace("    findings = []\n",
                                  "    raise RuntimeError(\"error-only control\")\n    findings = []\n"))) as tmp:
        outcome, detail, _, _ = run_suite(tmp)
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
    killed, observed = 0, {}
    for name, old, new in MUTANTS:
        if source.count(old) != 1:
            print(f"NOT APPLICABLE {name} ({source.count(old)} matches)")
            continue
        with with_tool(source.replace(old, new)) as tmp:
            outcome, detail, failed, errored = run_suite(tmp)
        observed[name] = {"fail": sorted(failed), "error": sorted(errored)}
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
    if args.write_failmap is not None:
        doc = {"population": {"test_file_sha256": TEST_SHA256, "tests": list(EXPECTED_TESTS)}, "mutants": observed}
        problems = failmap_problems(doc)
        name = next(iter(DESIGNATED))
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
    print(f"{killed}/{len(MUTANTS)} mutants killed")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
