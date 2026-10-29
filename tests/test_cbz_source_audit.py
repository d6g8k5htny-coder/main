"""Black-box contract tests; CBZ_AUDITOR_SCRIPT selects an isolated mutant."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(os.environ.get(
    "CBZ_AUDITOR_SCRIPT", str(Path(__file__).resolve().parents[1] / "tools/cbz_source_audit.py")
))


class SourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "source"
        self.root.mkdir()
        self.manifest = self.base / "manifest.json"

    def pin(self, files=None, **extra):
        if files is None:
            files = sorted(p.relative_to(self.root).as_posix()
                           for p in self.root.rglob("*.lean") if p.is_file())
        document = {
            "version": 1,
            "root": str(self.root),
            "files": [{"path": p, "sha256": hashlib.sha256((self.root / p).read_bytes()).hexdigest()}
                      for p in files],
        }
        document.update(extra)
        self.manifest.write_text(json.dumps(document), encoding="utf-8")
        return document

    def source(self, content, name="Main.lean"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        return path

    def run_audit(self, manifest=None, *arguments):
        self.assertTrue(SCRIPT.is_file(), "bounded auditor implementation is missing")
        run = subprocess.run(
            [sys.executable, str(SCRIPT), "--manifest", str(manifest or self.manifest), *arguments],
            text=True, capture_output=True, timeout=15,
        )
        self.assertIn(run.returncode, (1, 2), (run.stdout, run.stderr))
        self.assertEqual(run.stderr, "")
        try:
            report = json.loads(run.stdout)
        except json.JSONDecodeError:
            self.fail(f"auditor did not return a JSON report: {run.stdout!r} {run.stderr!r}")
        self.assertIn(report["aggregate_status"], ("NOT_ESTABLISHED", "REVIEW_REQUIRED"))
        self.assertEqual(report["gate_claims"]["G1"], "NOT_ESTABLISHED")
        self.assertEqual(report["gate_claims"]["CBZ-0"], "NOT_ESTABLISHED")
        return report, run

    def audit_source(self, source):
        self.source(source)
        self.pin()
        report = self.run_audit()[0]
        self.assertEqual(len(report["files"]), 1, "one pinned source must have one accounted record")
        return report

    def assert_error(self, code, report):
        self.assertEqual(report["aggregate_status"], "NOT_ESTABLISHED")
        self.assertEqual(report["input_integrity"]["status"], "NOT_ESTABLISHED")
        self.assertIn(code, [e["code"] for e in report["errors"]], report)

    def test_lexically_clear_source_cannot_establish_layer_a(self):
        report = self.audit_source("theorem example_ok : True := by\n  trivial\n")
        self.assertEqual(report["aggregate_status"], "NOT_ESTABLISHED")
        self.assertEqual(report["input_integrity"]["status"], "CHECKED")
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["files"][0]["status"], "SCANNED")
        gaps = {entry["invariant"]: entry for entry in report["blocking_coverage"]}
        for invariant in ("A3", "A4", "A5", "A6", "A7"):
            self.assertIn(invariant, gaps, "semantic blockers must not disappear from a report")
            self.assertEqual(gaps[invariant]["status"], "NOT_ESTABLISHED")
            self.assertTrue(gaps[invariant]["required_evidence"])
        self.assertEqual(report["coverage"]["lean_grammar_and_elaboration"], "NOT_ESTABLISHED")

    def test_every_prohibited_token_detector_has_independent_evidence(self):
        for token in ("sorry", "admit", "axiom", "native_decide", "unsafe",
                      "Float", "Float32", "run_tac", "run_elab", "eval_expr",
                      "implemented_by", "extern"):
            with self.subTest(token=token):
                report = self.audit_source("-- preceding line\n" + token + "\n")
                self.assertEqual(report["aggregate_status"], "REVIEW_REQUIRED")
                hit = [f for f in report["findings"] if f["token"] == token]
                self.assertEqual(len(hit), 1, report)
                self.assertEqual(hit[0]["line"], 2)
                self.assertEqual(hit[0]["column"], 1)
                self.assertEqual(hit[0]["source_line"], token)

    def test_extensible_syntax_is_reviewed_instead_of_silently_skipped(self):
        for token in ("macro", "macro_rules", "syntax", "elab", "elab_rules",
                      "declare_syntax_cat", "initialize", "builtin_initialize"):
            with self.subTest(token=token):
                report = self.audit_source(token + " anything\n")
                self.assertEqual(report["aggregate_status"], "REVIEW_REQUIRED")
                self.assertTrue(any(f["token"] == token and f["code"] == "EXTENSIBLE_SYNTAX"
                                    for f in report["findings"]))

    def test_nested_comments_and_literal_strings_do_not_create_token_hits(self):
        text = (
            '-- sorry " unterminated quote inside a comment\n'
            '/- axiom /- admit " -/ unsafe /- Float -/ -/\n'
            'def message := "sorry admit axiom native_decide unsafe Float run_tac"\n'
            r'def escaped := "a \" sorry \\ still a string"' + '\n'
            'def raw := r##"sorry " axiom \\ unsafe"##\n'
            "def character := '\"'\n"
        )
        report = self.audit_source(text)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["findings"], [])

    def test_comment_end_inside_string_and_quote_inside_comment_do_not_leak(self):
        report = self.audit_source('def s := "/- sorry -/ \\\" axiom"\n/- " -/\nadmit\n')
        self.assertEqual([(f["token"], f["line"]) for f in report["findings"]], [("admit", 3)])

    def test_raw_string_hash_delimiters_and_backslashes_are_not_normal_escapes(self):
        report = self.audit_source('def a := r"sorry \\"\ndef b := r#"axiom " unsafe"#\n')
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["findings"], [])

    def test_identifiers_with_keyword_substrings_and_apostrophes_are_not_keywords(self):
        report = self.audit_source("def sorryName := 1\ndef xs' := 2\ndef notFloat := 3\ndef sorry? := 4\ndef sorry! := 5\n")
        self.assertEqual(report["findings"], [])

    def test_unicode_identifier_cannot_be_split_into_a_spurious_keyword(self):
        report = self.audit_source("def sorryα := 1\n")
        self.assertEqual(report["errors"], [])
        self.assertEqual([(f["code"], f["token"]) for f in report["findings"]],
                         [("UNSUPPORTED_SYNTAX", "sorryα")])

    def test_qualified_tokens_are_detected_and_have_exact_utf8_byte_offsets(self):
        text = '-- α\r\n\n  Lean.sorry\n'
        report = self.audit_source(text)
        self.assertEqual(len(report["findings"]), 1)
        hit = report["findings"][0]
        self.assertEqual(hit["token"], "sorry")
        self.assertEqual((hit["line"], hit["column"], hit["byte_offset"], hit["byte_length"]),
                         (3, 8, 15, 5))
        self.assertEqual(hit["source_line"], "  Lean.sorry")

    def test_quoted_identifiers_are_reviewed_even_with_comment_characters(self):
        report = self.audit_source("def «/-sorry-/» := 1\n")
        self.assertEqual(report["aggregate_status"], "REVIEW_REQUIRED")
        self.assertEqual([(f["code"], f["token"]) for f in report["findings"]],
                         [("QUOTED_IDENTIFIER", "«/-sorry-/»")])

    def test_quotations_antiquotations_and_unknown_characters_are_reviewed(self):
        for text, token in (("`(tactic| sorry)", "`"), ("$x", "$"),
                            ("#eval 1", "#"), ("sorr\u2060y", "\u2060"),
                            ("def x := λ", "λ"), ("def x := \\x", "\\")):
            with self.subTest(text=text):
                report = self.audit_source(text)
                self.assertEqual(report["aggregate_status"], "REVIEW_REQUIRED")
                self.assertTrue(any(f["code"] == "UNSUPPORTED_SYNTAX" and f["token"] == token
                                    for f in report["findings"]))

    def test_interpolated_strings_are_reviewed_even_if_the_payload_is_masked(self):
        for prefix in ("s!", "m!", "f!"):
            with self.subTest(prefix=prefix):
                report = self.audit_source('def s := ' + prefix + '"{sorry}"\n')
                self.assertEqual(report["aggregate_status"], "REVIEW_REQUIRED")
                self.assertTrue(any(f["code"] == "STRING_INTERPOLATION"
                                    for f in report["findings"]))

    def test_unknown_string_escape_is_explicit_review(self):
        report = self.audit_source('def s := "\\q sorry"\n')
        self.assertEqual(report["aggregate_status"], "REVIEW_REQUIRED")
        self.assertTrue(any(f["code"] == "UNSUPPORTED_ESCAPE" for f in report["findings"]))
        self.assertFalse(any(f["token"] == "sorry" for f in report["findings"]))

    def test_unterminated_lexical_constructs_fail_closed(self):
        for text in ('/- sorry', '"sorry', 'r##"sorry"#', '«sorry', "'x", "'sorry'"):
            with self.subTest(text=text):
                self.assert_error("LEXICAL_ERROR", self.audit_source(text))

    def test_invalid_utf8_is_not_replaced_or_ignored(self):
        self.source(b"theorem x := \xffsorry\n")
        self.pin()
        report, _ = self.run_audit()
        self.assert_error("INVALID_UTF8", report)
        self.assertEqual(report["files"][0]["status"], "ERROR")

    def test_missing_manifest_file_is_not_skipped(self):
        self.source("theorem a : True := by trivial\n")
        self.pin()
        (self.root / "Main.lean").unlink()
        report, run = self.run_audit()
        self.assertEqual(run.returncode, 2)
        self.assert_error("MISSING_FILE", report)
        self.assertEqual(report["input_integrity"]["expected_files"], 1)
        self.assertEqual(report["files"][0]["status"], "ERROR")

    def test_unavailable_root_still_accounts_for_every_manifest_entry(self):
        self.source("def x := 1\n", "X.lean")
        self.source("def y := 2\n", "Y.lean")
        self.pin()
        for path in self.root.iterdir():
            path.unlink()
        self.root.rmdir()
        report, _ = self.run_audit()
        self.assert_error("MISSING_FILE", report)
        self.assertEqual([(f["path"], f["status"]) for f in report["files"]],
                         [("X.lean", "ERROR"), ("Y.lean", "ERROR")])

    def test_source_resource_limit_is_blocking_before_unbounded_read(self):
        path = self.source(b"")
        with path.open("wb") as output:
            output.truncate(33554433)
        self.pin()
        report, _ = self.run_audit()
        self.assert_error("RESOURCE_LIMIT", report)

    def test_otherwise_valid_manifest_with_duplicate_keys_is_rejected(self):
        self.source("def x := 1\n")
        self.pin()
        original = self.manifest.read_text(encoding="utf-8")
        self.manifest.write_text(original.replace('"version": 1', '"version": 1, "version": 1'),
                                 encoding="utf-8")
        self.assert_error("INVALID_MANIFEST", self.run_audit()[0])

    def test_relative_root_is_anchored_to_manifest_directory(self):
        self.source("def x := 1\n")
        self.pin(root="source")
        report, _ = self.run_audit()
        self.assertEqual(report["input_integrity"]["status"], "CHECKED")
        self.assertEqual(report["input_integrity"]["scanned_files"], 1)

    def test_exact_manifest_hash_mismatch_is_blocking(self):
        self.source("def x := 1\n")
        self.pin()
        self.source("def x := 2\n")
        report, _ = self.run_audit()
        self.assert_error("HASH_MISMATCH", report)

    def test_every_lean_descendant_must_be_explicitly_pinned(self):
        self.source("def x := 1\n")
        self.pin()
        self.source("axiom hidden : False\n", "nested/Hidden.lean")
        report, _ = self.run_audit()
        self.assert_error("UNLISTED_FILE", report)
        self.assertIn("nested/Hidden.lean", report["input_integrity"]["unlisted_files"])

    def test_file_and_directory_symlinks_are_rejected_without_following(self):
        self.source("def x := 1\n")
        document = self.pin()
        outside = self.base / "Outside.lean"
        outside.write_text("sorry\n", encoding="utf-8")
        for kind in ("file", "directory", "unrelated"):
            with self.subTest(kind=kind):
                link = self.root / ("Link.lean" if kind == "file" else "link")
                link.symlink_to(outside if kind in ("file", "unrelated") else self.base,
                                target_is_directory=kind == "directory")
                try:
                    report, _ = self.run_audit()
                    self.assert_error("SYMLINK", report)
                finally:
                    link.unlink()
        self.assertEqual(document["files"][0]["path"], "Main.lean")

    def test_symlinked_root_ancestor_and_manifest_are_rejected(self):
        self.source("def x := 1\n")
        document = self.pin()
        link = self.base / "linked"
        link.symlink_to(self.root, target_is_directory=True)
        document["root"] = str(link)
        self.manifest.write_text(json.dumps(document), encoding="utf-8")
        self.assert_error("SYMLINK", self.run_audit()[0])
        document["root"] = str(link / "sub")
        self.manifest.write_text(json.dumps(document), encoding="utf-8")
        self.assert_error("SYMLINK", self.run_audit()[0])
        manifest_link = self.base / "linked-manifest.json"
        manifest_link.symlink_to(self.manifest)
        self.assert_error("SYMLINK", self.run_audit(manifest_link)[0])

    def test_unreadable_source_and_directories_are_not_root_bypass_success(self):
        path = self.source("def x := 1\n", "nested/Main.lean")
        self.pin()
        path.chmod(0)
        try:
            self.assert_error("UNREADABLE", self.run_audit()[0])
        finally:
            path.chmod(0o600)
        path.parent.chmod(0)
        try:
            self.assert_error("UNREADABLE", self.run_audit()[0])
        finally:
            path.parent.chmod(0o700)

    def test_explicit_permission_guard_does_not_depend_on_os_capabilities(self):
        path = self.source("def x := 1\n")
        spec = importlib.util.spec_from_file_location("auditor_guard_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        path.chmod(0)
        try:
            with self.assertRaises(module.AuditFailure) as caught:
                module.check_entry(path.stat(), path)
            self.assertEqual(caught.exception.error["code"], "UNREADABLE")
        finally:
            path.chmod(0o600)

    def test_output_fifo_refuses_without_waiting_for_a_reader(self):
        self.source("def x := 1\n")
        self.pin()
        output = self.base / "output.fifo"
        os.mkfifo(output)
        try:
            run = subprocess.run(
                [sys.executable, str(SCRIPT), "--manifest", str(self.manifest),
                 "--output", str(output)], text=True, capture_output=True, timeout=2,
            )
        except subprocess.TimeoutExpired:
            self.fail("report output blocked on a FIFO")
        self.assertEqual(run.returncode, 2)
        self.assertIn("OUTPUT_ERROR", [e["code"] for e in json.loads(run.stdout)["errors"]])

    def test_output_cannot_overwrite_manifest_source_or_previous_evidence(self):
        source = self.source("def x := 1\n")
        self.pin()
        previous = self.base / "previous.json"
        previous.write_bytes(b"immutable earlier evidence\n")
        for output in (self.manifest, source, previous):
            with self.subTest(output=output.name):
                before = output.read_bytes()
                report, run = self.run_audit(None, "--output", str(output))
                self.assertEqual(run.returncode, 2)
                self.assertIn("OUTPUT_ERROR", [e["code"] for e in report["errors"]])
                self.assertEqual(output.read_bytes(), before)

    def test_output_is_outside_audited_root_and_has_no_symlinked_ancestor(self):
        self.source("def x := 1\n")
        self.pin()
        output = self.root / "new-output.json"
        report, run = self.run_audit(None, "--output", str(output))
        self.assertEqual(run.returncode, 2)
        self.assertIn("OUTPUT_ERROR", [e["code"] for e in report["errors"]])
        self.assertFalse(output.exists())
        target = self.base / "output-directory"
        target.mkdir()
        alias = self.base / "output-alias"
        alias.symlink_to(target, target_is_directory=True)
        report, run = self.run_audit(None, "--output", str(alias / "receipt.json"))
        self.assertIn("OUTPUT_ERROR", [e["code"] for e in report["errors"]])
        self.assertFalse((target / "receipt.json").exists())

    def test_directory_limit_bounds_consumption_before_sorting_names(self):
        for index in range(8):
            (self.root / f"{index}.txt").write_text("entry\n")
        spec = importlib.util.spec_from_file_location("auditor_limit_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.MAX_TREE_ENTRIES = 3
        real_scandir = os.scandir
        consumed = []
        class TrackedDirectory:
            def __init__(self, fd):
                self.iterator = real_scandir(fd)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.iterator.close()
            def __iter__(self):
                return self
            def __next__(self):
                entry = next(self.iterator)
                consumed.append(entry.name)
                return entry
        fd = module.open_directory(self.root)
        try:
            module.os.scandir = TrackedDirectory
            with self.assertRaises(module.AuditFailure) as caught:
                module.discover_sources(fd)
            self.assertEqual(caught.exception.error["code"], "RESOURCE_LIMIT")
        finally:
            module.os.scandir = real_scandir
            os.close(fd)
        self.assertLessEqual(len(consumed), 4, "enumeration must stop at the first excess entry")

    def test_nonregular_source_is_rejected_without_blocking(self):
        self.source("def x := 1\n")
        document = self.pin()
        (self.root / "Main.lean").unlink()
        os.mkfifo(self.root / "Main.lean")
        self.assert_error("NONREGULAR_FILE", self.run_audit()[0])
        self.assertEqual(len(document["files"]), 1)

    def test_unsafe_manifest_paths_and_duplicates_fail_closed(self):
        self.source("def x := 1\n")
        original = self.pin()
        for path in ("../Outside.lean", "/Main.lean", "nested/../Main.lean", "./Main.lean",
                     "nested//Main.lean", "nested\\Main.lean", "Main.txt", ""):
            with self.subTest(path=path):
                document = json.loads(json.dumps(original))
                document["files"][0]["path"] = path
                self.manifest.write_text(json.dumps(document), encoding="utf-8")
                self.assert_error("INVALID_MANIFEST", self.run_audit()[0])
        original["files"].append(original["files"][0])
        self.manifest.write_text(json.dumps(original), encoding="utf-8")
        self.assert_error("INVALID_MANIFEST", self.run_audit()[0])

    def test_empty_invalid_or_extended_manifest_is_not_a_vacuous_pass(self):
        self.source("def x := 1\n")
        original = self.pin()
        for update in ({"files": []}, {"version": True}, {"version": 2}, {"skip_errors": True},
                       {"files": [{"path": "Main.lean", "sha256": "0" * 63}]},
                       {"files": [{"path": "Main.lean", "sha256": "0" * 64, "skip": True}]}):
            with self.subTest(update=update):
                document = dict(original)
                document.update(update)
                self.manifest.write_text(json.dumps(document), encoding="utf-8")
                self.assert_error("INVALID_MANIFEST", self.run_audit()[0])
        for payload in ('{"version":1,"version":1}', '{"value":NaN}', "[]", "{"):
            with self.subTest(payload=payload):
                self.manifest.write_text(payload, encoding="utf-8")
                self.assert_error("INVALID_MANIFEST", self.run_audit()[0])

    def test_missing_unreadable_and_malformed_utf8_manifest_fail_closed(self):
        self.assert_error("MISSING_FILE", self.run_audit()[0])
        self.manifest.write_bytes(b"\xff")
        self.assert_error("INVALID_UTF8", self.run_audit()[0])
        self.manifest.chmod(0)
        try:
            self.assert_error("UNREADABLE", self.run_audit()[0])
        finally:
            self.manifest.chmod(0o600)

    def test_reports_are_deterministic_and_account_for_every_pinned_file(self):
        self.source("def z := 2\n", "z/Z.lean")
        self.source("def a := 1\n", "A.lean")
        document = self.pin(["z/Z.lean", "A.lean"])
        report, first = self.run_audit()
        _, second = self.run_audit()
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual([f["path"] for f in report["files"]], ["A.lean", "z/Z.lean"])
        self.assertEqual(report["input_integrity"]["expected_files"], 2)
        self.assertEqual(report["input_integrity"]["scanned_files"], 2)
        self.assertEqual(report["input_integrity"]["discovered_files"], ["A.lean", "z/Z.lean"])
        expected = {f["path"]: f["sha256"] for f in document["files"]}
        for entry in report["files"]:
            self.assertEqual(entry["sha256"], expected[entry["path"]])

    def test_output_artifact_matches_stdout_and_does_not_imply_success(self):
        self.source("sorry\n")
        self.pin()
        output = self.base / "report.json"
        report, run = self.run_audit(None, "--output", str(output))
        self.assertEqual(output.read_text(encoding="utf-8"), run.stdout)
        self.assertEqual(report["aggregate_status"], "REVIEW_REQUIRED")


if __name__ == "__main__":
    unittest.main()
