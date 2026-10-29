"""Mutation self-test of the Layer A scanner (CBZ-0 gate G0, directive §2.3/A8).

Each mutant is a copy of tools/cbz/layer_a_scan.py with one deliberate defect:
a hazard token dropped, a lexer rule broken, or a fail-closed exit weakened.
The scanner's own suite (tests/test_cbz_layer_a_scan.py) is run against every
mutant through $CBZ_LAYER_A_SCAN and must FAIL. A surviving mutant means the
suite cannot tell a broken auditor from a sound one, so this test fails.

Every replacement must match the scanner source exactly once, so the harness
cannot silently drift out of sync with the code it mutates. A mutant counts as
killed only when the suite loaded and ran, and at least one test failed or
errored; a suite that could not even be imported proves nothing.
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)
SCANNER = os.path.join(WORKSPACE, "tools", "cbz", "layer_a_scan.py")

MUTANTS = [
    ("drop sorry", '    "sorry": "A1",\n', ""),
    ("drop admit", '    "admit": "A1",\n', ""),
    ("drop axiom", '    "axiom": "A2",\n', ""),
    ("drop native_decide", '    "native_decide": "A2",\n', ""),
    ("drop sorryAx", '    "sorryAx": "A1",\n', ""),
    ("drop admitGoal", '    "admitGoal": "A1",\n', ""),
    ("drop mkSorry", '    "mkSorry": "A1",\n', ""),
    ("drop ofReduceBool", '    "ofReduceBool": "A2",\n', ""),
    ("drop ofReduceNat", '    "ofReduceNat": "A2",\n', ""),
    ("drop trustCompiler", '    "trustCompiler": "A2",\n', ""),
    ("drop skipKernelTC", '    "skipKernelTC": "A2",\n', ""),
    ("drop Float review", '    "Float": "A3",\n', ""),
    ("violation demoted to review", 'return name, VIOLATION_EXACT[name], "violation"',
     'return name, VIOLATION_EXACT[name], "review"'),
    ("no line comments", 'if c == "-" and text.startswith("--", i):', 'if False:'),
    ("flat block comments", 'if text.startswith("/-", i):\n                    depth += 1\n', 'if text.startswith("/-", i):\n                    depth = 1\n'),
    ("plain strings not skipped", "        if c == '\"':\n", "        if False:\n"),
    ("interpolation regions not lexed", 'stack.append(["code", 0])', "pass"),
    ("char literals not recognised", "m = _CHAR_LIT.match(text, i)", "m = None"),
    ("raw strings not recognised", 'if word == "r" and i < n', 'if False and i < n'),
    ("dotted names not split", 'last = name.split(".")[-1]', "last = name"),
    ("stop flagged anywhere", "if line_start and name in LINE_START_REVIEW:",
     "if name in LINE_START_REVIEW:"),
    ("evidence treated as production", 'return "evidence_copy"', 'return "production"'),
    ("git mode lists no blobs", 'if typ == b"blob" and p.endswith(".lean"):', "if False:"),
    ("violations never fail", "return 1 if failing else 0", "return 0"),
    ("undecodable files pass", "    if undecodable:\n        return 1\n",
     "    if False:\n        return 1\n"),
    ("fail on every context by default", 'ap.add_argument("--fail-on", default="production")',
     'ap.add_argument("--fail-on", default="all")'),
    # added after 014K's specification (main#684/6503596060)
    ("drop addDeclWithoutChecking", '    "addDeclWithoutChecking": "A2",\n', ""),
    ("+native ignored", 'if name == "native" and (plus_before or _NATIVE_TRUE.match(follow) or (field_before and _native_assignment(follow))):',
     'if name == "native" and (_NATIVE_TRUE.match(follow) or (field_before and _native_assignment(follow))):'),
    ("native := true ignored", 'if name == "native" and (plus_before or _NATIVE_TRUE.match(follow) or (field_before and _native_assignment(follow))):',
     'if name == "native" and plus_before:'),
    ("escaped constants never match", "        # escaped component: the same constant name, never a keyword\n        last = last[1:]",
     "        return None"),
    ("escaped keywords flagged",
     '        if last in REVIEW_COMPONENT:\n            return last, REVIEW_COMPONENT[last], "review"\n        return None',
     '        if last in REVIEW_COMPONENT:\n            return last, REVIEW_COMPONENT[last], "review"\n'
     '        return (last, VIOLATION_EXACT[last], "violation") if last in VIOLATION_EXACT else None'),
    ("dots inside escaped names split", 'parts = ["\\x00" + inner.replace(".", "\\x01")]', 'parts = ["\\x00" + inner]'),
    ("unterminated string passes", 'errors.append(start + ("unterminated_string",))', "pass"),
    ("unterminated comment passes", 'errors.append(start + ("unterminated_comment",))', "pass"),
    ("unterminated raw string passes", 'errors.append((start_line, start_col, "unterminated_raw_string"))', "pass"),
    ("unterminated interpolation passes",
     'errors.append((frame[1], frame[2], "unterminated_interpolated_string"))', "pass"),
    ("unterminated escaped name passes",
     '                errors.append((line, col, "unterminated_escaped_name"))\n                advance(n - i)\n                break',
     '                advance(n - i)\n                break'),
    ("unterminated dotted escaped name passes",
     '                    errors.append((line, col, "unterminated_escaped_name"))\n                    advance(n - i)\n'
     '                    return',
     '                    advance(n - i)\n                    return'),
    ("lex errors never fail", 'f["severity"] in ("violation", "error")', 'f["severity"] == "violation"'),
]


# Additional source mutations for the canonical input/lexer repairs.
MUTANTS.extend([('native comment layout ignored', "        elif text.startswith('/-', i):", '        elif False:'), ('native suffix truncated', 'while i < len(text) and len(out) < 8:', 'while i < len(text) and len(out) < 8 and i - start < 40:'), ('plain macro braces ignored', 'yield from _brace_segments(text, content_start, i - 1, start[0], start[1] + 1)', 'pass'), ('unfinished char ignored', "errors.append((line, col, 'unterminated_char'))", 'pass'), ('missing root ignored', 'raise OSError(f"source path does not exist: {p}")', 'pass'), ('source symlink accepted', 'raise OSError(f"source symlink requires explicit target admission: {p}")', 'pass'), ('directory symlink skipped', 'raise OSError(f"source directory symlink requires explicit target admission: {d}")', 'pass'), ('gitlink skipped', 'raise OSError(f"gitlink requires a separately pinned scan: {p}")', 'pass'), ('git source symlink accepted', 'if mode not in (b"100644", b"100755"):', 'if False:'), ('git batch header unchecked', 'if len(header) != 3 or header[0] != sha.encode() or header[1] != b"blob" or not header[2].isdigit():', 'if False:'), ('git batch body unchecked', 'if len(data) != size or trailer != b"\\n":', 'if False:'), ('git batch exit unchecked', 'if code != 0:', 'if False:')])

MUTANTS.extend([('interpolation prefix layout ignored', 'if last_code_char in INTERP_PREFIXES:', 'if False:'), ('drop Float32 review', '    "Float32": "A3",\n', ''), ('drop unsafe review', '    "unsafe": "A2",\n', ''), ('drop implemented_by review', '    "implemented_by": "A2",\n', ''), ('drop extern review', '    "extern": "A2",\n', '')])

MUTANTS.extend([('macro hole truncated at primary quote', 'lex_tokens(text[i + 1:], _until_brace=True)', 'lex_tokens(text[i + 1:end], _until_brace=True)'), ('numeral swallows adjacent keyword', 'match = _NUMERAL.match(text, i)', 'match = re.compile(r"[A-Za-z0-9_]+").match(text, i)')])

MUTANTS.extend([('scientific numeric tail ignored', 'match = _NUMERAL.match(text, i)', 'match = re.compile(r"[0-9_]+").match(text, i)'), ('prime notation called invalid char', 'if m is None and not prime_notation:', 'if m is None:'), ('unicode char escape ignored', 'u[0-9a-fA-F]{4}|', '')])

MUTANTS.extend([('identifier domain broadened to Unicode letters', '    o = ord(ch)\n    # Letter-like symbols', '    if ch.isalpha():\n        return True\n    o = ord(ch)\n    # Letter-like symbols'), ('U2C7C identifier suffix omitted', ' or o == 0x2C7C', ''), ('unsupported char escape accepted', '_CHAR_LIT = re.compile(r"\'(?:\\\\(?:u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|[\\\\\\"\'rnt])|[^\\\\\'\\n])\'")', '_CHAR_LIT = re.compile(r"\'(?:\\\\(?:u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|.)|[^\\\\\'\\n])\'")')])

MUTANTS.extend([('unknown native field expression admitted', '(field_before and _native_assignment(follow))', 'False'), ('false expression mistaken for literal false', "not (len(follow) >= 8 and follow[2:7] == 'false' and follow[7] in '},')", "not follow.startswith(':=false')")])

EXPECTED_FAILURE = {'drop sorry': 'test_each_violation_token_detected', 'drop admit': 'test_each_violation_token_detected', 'drop axiom': 'test_each_violation_token_detected', 'drop native_decide': 'test_each_violation_token_detected', 'drop sorryAx': 'test_each_violation_token_detected', 'drop admitGoal': 'test_each_violation_token_detected', 'drop mkSorry': 'test_each_violation_token_detected', 'drop ofReduceBool': 'test_each_violation_token_detected', 'drop ofReduceNat': 'test_each_violation_token_detected', 'drop trustCompiler': 'test_each_violation_token_detected', 'drop skipKernelTC': 'test_each_violation_token_detected', 'drop Float review': 'test_review_tokens', 'violation demoted to review': 'test_each_violation_token_detected', 'no line comments': 'test_line_comment', 'flat block comments': 'test_block_comments_nested_and_doc', 'plain strings not skipped': 'test_strings', 'interpolation regions not lexed': 'test_interpolated_strings', 'char literals not recognised': 'test_char_literals_do_not_open_strings', 'raw strings not recognised': 'test_raw_strings', 'dotted names not split': 'test_each_violation_token_detected', 'stop flagged anywhere': 'test_stop_only_at_line_start', 'evidence treated as production': 'test_context_classification', 'git mode lists no blobs': 'test_git_tree_mode', 'violations never fail': 'test_production_violation_fails', 'undecodable files pass': 'test_undecodable_fails_closed', 'fail on every context by default': 'test_evidence_violation_reported_not_failing_by_default', 'drop addDeclWithoutChecking': 'test_each_violation_token_detected', '+native ignored': 'test_native_decide_options', 'native := true ignored': 'test_native_decide_options', 'escaped constants never match': 'test_escaped_constant_names_are_the_same_constants', 'escaped keywords flagged': 'test_escaped_keyword_names_are_identifiers', 'dots inside escaped names split': 'test_escaped_constant_names_are_the_same_constants', 'unterminated string passes': 'test_unterminated_constructs_are_errors', 'unterminated comment passes': 'test_unterminated_constructs_are_errors', 'unterminated raw string passes': 'test_unterminated_constructs_are_errors', 'unterminated interpolation passes': 'test_unterminated_constructs_are_errors', 'unterminated escaped name passes': 'test_unterminated_constructs_are_errors', 'unterminated dotted escaped name passes': 'test_unterminated_constructs_are_errors', 'lex errors never fail': 'test_lex_error_fails_the_run'}

EXPECTED_FAILURE.update({'native comment layout ignored': 'test_native_configuration_layout_is_not_a_bypass', 'native suffix truncated': 'test_native_configuration_layout_is_not_a_bypass', 'plain macro braces ignored': 'test_plain_macro_braces_are_conservatively_scanned', 'unfinished char ignored': 'test_unterminated_character_literal_is_error', 'missing root ignored': 'test_missing_root_fails_for_both_tools', 'source symlink accepted': 'test_source_symlink_fails_for_both_tools', 'directory symlink skipped': 'test_symlink_directory_is_not_silently_skipped', 'gitlink skipped': 'test_gitlink_is_explicitly_rejected_by_both_tools', 'git source symlink accepted': 'test_git_source_symlink_fails_for_both_tools', 'git batch header unchecked': 'test_malformed_blob_transport_fails_closed', 'git batch body unchecked': 'test_malformed_blob_transport_fails_closed', 'git batch exit unchecked': 'test_malformed_blob_transport_fails_closed'})

EXPECTED_FAILURE.update({'interpolation prefix layout ignored': 'test_interpolation_prefix_accepts_lean_layout', 'drop Float32 review': 'test_review_tokens', 'drop unsafe review': 'test_review_tokens', 'drop implemented_by review': 'test_review_tokens', 'drop extern review': 'test_review_tokens'})

EXPECTED_FAILURE.update({'macro hole truncated at primary quote': 'test_plain_macro_holes_with_nested_literals', 'numeral swallows adjacent keyword': 'test_adjacent_numeral_never_swallows_code_keyword'})

EXPECTED_FAILURE.update({'scientific numeric tail ignored': 'test_scientific_and_radix_numerals_preserve_keyword_boundary', 'prime notation called invalid char': 'test_mathlib_prime_notations_are_not_unfinished_characters', 'unicode char escape ignored': 'test_mathlib_prime_notations_are_not_unfinished_characters'})

EXPECTED_FAILURE.update({'identifier domain broadened to Unicode letters': 'test_pinned_lean_identifier_character_domain', 'U2C7C identifier suffix omitted': 'test_pinned_lean_identifier_character_domain', 'unsupported char escape accepted': 'test_unsupported_character_escape_is_error'})

EXPECTED_FAILURE.update({'unknown native field expression admitted': 'test_unknown_native_field_expression_fails_closed', 'false expression mistaken for literal false': 'test_unknown_native_field_expression_fails_closed'})

def run_suite(scanner_path):
    env = dict(os.environ, CBZ_LAYER_A_SCAN=scanner_path, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, *(["-O"] if sys.flags.optimize else []), "-B", "-S", "-m", "unittest", "-v", "tests.test_cbz_layer_a_scan", "tests.test_cbz_auditor_hardening"],
                          cwd=WORKSPACE, env=env, capture_output=True, text=True, timeout=600)


def killed(r, expected_test=None):
    """Only a named assertion failure kills; launch/import/errors do not."""
    out = r.stdout + r.stderr
    summary = re.search(r"^FAILED \(([^)]*)\)$", out, re.M)
    if r.returncode == 0 or not summary or "errors=" in summary.group(1):
        return False
    if not re.search(r"^Ran [1-9][0-9]* tests?", out, re.M):
        return False
    failures = re.findall(r"^FAIL: (test_[A-Za-z0-9_]+) \(", out, re.M)
    return bool(failures) and (expected_test is None or expected_test in failures)


class ScannerMutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SCANNER, encoding="utf-8") as fh:
            cls.source = fh.read()

    def test_unmutated_scanner_passes_its_suite(self):
        r = run_suite(SCANNER)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_every_mutant_is_killed(self):
        survivors = []
        with tempfile.TemporaryDirectory() as d:
            for i, (name, old, new) in enumerate(MUTANTS):
                with self.subTest(mutant=name):
                    self.assertEqual(self.source.count(old), 1, f"mutation point drifted: {name}")
                    path = os.path.join(d, f"mutant_{i:02d}.py")
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(self.source.replace(old, new))
                    r = run_suite(path)
                    if not killed(r, EXPECTED_FAILURE[name]):
                        survivors.append(name)
        self.assertEqual(survivors, [], f"surviving mutants: {survivors}")
        self.assertGreaterEqual(len(MUTANTS), 39)


if __name__ == "__main__":
    unittest.main()
