"""Tests for the CBZ-0 Layer A Lean scanner (gate G0 runs these on mutants).

Set CBZ_SCANNER_PATH to test a different copy of the scanner; the mutation
check (tools/cbz_scanner_mutation_check.py) uses this to run every mutant
against the same assertions.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCANNER = Path(os.environ.get('CBZ_SCANNER_PATH', ROOT / 'tools' / 'cbz_layer_a_scan.py'))


def _load():
    spec = importlib.util.spec_from_file_location('cbz_layer_a_scan_under_test', SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scan = _load()


def tokens(text):
    return [tok for tok, _ in scan.scan_text(text)]


class FlagsRealViolations(unittest.TestCase):
    def test_sorry_in_term_and_tactic(self):
        self.assertEqual(tokens('theorem t : True := sorry\n'), ['sorry'])
        self.assertEqual(tokens('theorem t : True := by\n  sorry\n'), ['sorry'])
        self.assertEqual(tokens('example : 1 = 1 := (sorry)\n'), ['sorry'])

    def test_admit(self):
        self.assertEqual(tokens('theorem t : True := by admit\n'), ['admit'])

    def test_sorry_ax_including_dotted(self):
        self.assertEqual(tokens('def x : Nat := sorryAx Nat true\n'), ['sorryAx'])
        self.assertEqual(tokens('def x : Nat := Lean.sorryAx Nat true\n'), ['sorryAx'])

    def test_axiom_declarations(self):
        self.assertEqual(tokens('axiom choice_like : False\n'), ['axiom'])
        self.assertEqual(tokens('private axiom h : 1 = 2\n'), ['axiom'])
        self.assertEqual(tokens('@[simp] axiom h : 1 = 2\n'), ['axiom'])

    def test_native_decide_and_plus_native(self):
        self.assertEqual(tokens('example : 2 + 2 = 4 := by native_decide\n'), ['native_decide'])
        self.assertEqual(tokens('example : 2 + 2 = 4 := by decide +native\n'), ['+native'])
        self.assertEqual(tokens('example : 2 + 2 = 4 := by decide  +native\n'), ['+native'])

    def test_of_reduce_bool(self):
        self.assertEqual(tokens('theorem t := Lean.ofReduceBool a b h\n'), ['ofReduceBool'])

    def test_several_hits_and_line_numbers(self):
        text = ('/- header\n spanning\n lines -/\n'
                'theorem a : True := sorry\n'
                'def s := "multi\nline"\n'
                'theorem b : True := by admit\n')
        self.assertEqual(scan.scan_text(text), [('sorry', 4), ('admit', 7)])


class IgnoresNonCode(unittest.TestCase):
    def test_line_comment(self):
        self.assertEqual(tokens('-- sorry admit axiom native_decide\ntheorem t : True := trivial\n'), [])

    def test_block_and_doc_comments(self):
        self.assertEqual(tokens('/- sorry -/ theorem t : True := trivial\n'), [])
        self.assertEqual(tokens('/-- uses no sorry -/ theorem t : True := trivial\n'), [])
        self.assertEqual(tokens('/-! module doc: axiom-free -/\n'), [])

    def test_nested_block_comment(self):
        self.assertEqual(tokens('/- outer /- inner -/ still comment sorry -/\ndef x := 1\n'), [])
        # After the nested comment closes, real code is scanned again.
        self.assertEqual(tokens('/- a /- b -/ c -/ theorem t : True := sorry\n'), ['sorry'])

    def test_string_literals(self):
        self.assertEqual(tokens('def s := "sorry"\n'), [])
        self.assertEqual(tokens('def s := "quote \\" then sorry"\n'), [])
        self.assertEqual(tokens('def s := "a" ++ "b"\ntheorem t : True := sorry\n'), ['sorry'])

    def test_raw_strings(self):
        self.assertEqual(tokens('def s := r"sorry"\n'), [])
        self.assertEqual(tokens('def s := r#"has "sorry" inside"#\ndef y := 2\n'), [])

    def test_char_literal_quote_does_not_open_a_string(self):
        self.assertEqual(tokens("def c := '\"'\ntheorem t : True := sorry\n"), ['sorry'])
        self.assertEqual(tokens("def c := '\\''\ntheorem t : True := sorry\n"), ['sorry'])

    def test_identifier_boundaries(self):
        self.assertEqual(tokens('def sorryish := 1\ndef my_admit := 2\n'), [])
        self.assertEqual(tokens('#print axioms foo\n'), [])
        self.assertEqual(tokens("theorem h' : True := trivial\n"), [])
        self.assertEqual(tokens('def native := 3\ndef x := native\n'), [])

    def test_name_literals_are_flagged_in_the_safe_direction(self):
        # `sorry is a Name literal (data), but the lexer treats it as code: a false
        # positive, never a false negative.
        self.assertEqual(tokens('def n := `sorry\n'), ['sorry'])


class FailsClosed(unittest.TestCase):
    def test_unterminated_string_raises(self):
        with self.assertRaises(scan.LexError):
            scan.scan_text('def s := "open\ntheorem t : False := sorry\n')

    def test_unterminated_block_comment_raises(self):
        with self.assertRaises(scan.LexError):
            scan.scan_text('/- open\ntheorem t : False := sorry\n')

    def test_unterminated_raw_string_raises(self):
        # Without raw-string handling this would lex as a closed plain string.
        with self.assertRaises(scan.LexError):
            scan.scan_text('def s := r#"has " quote\ntheorem t : False := trivial\n')

    def test_unterminated_interpolated_string_raises(self):
        with self.assertRaises(scan.LexError):
            scan.scan_text('def s := s!"a {f x\n')


class ReviewerFalseNegatives(unittest.TestCase):
    """The six cases 014K reported on #687 (review 5701985906), each now caught."""

    def test_interpolated_hole_is_code(self):
        self.assertEqual(tokens('def s := s!"value {sorry}"\n'), ['sorry'])
        self.assertEqual(tokens('def s := m!"{f (sorry)}"\n'), ['sorry'])
        self.assertEqual(tokens('def s := f!"x {admit} y"\n'), ['admit'])

    def test_interpolated_hole_with_nested_string(self):
        # A plain-string lexer would end at the inner quote and miss this sorry.
        self.assertEqual(tokens('def s := s!"a {g "x" sorry} b"\n'), ['sorry'])

    def test_interpolated_nested_braces_and_escapes(self):
        self.assertEqual(tokens('def s := s!"a {f {x := 1}} \\{sorry} b"\n'), [])
        self.assertEqual(tokens('def s := s!"a {f {x := sorry}} b"\n'), ['sorry'])
        # The hole stays open past an inner closing brace; a non-nesting lexer misses this.
        self.assertEqual(tokens('def s := s!"a {f {x := 1} sorry} b"\n'), ['sorry'])

    def test_plain_string_braces_are_scanned(self):
        # throwError and similar macros interpolate plain-looking literals.
        self.assertEqual(tokens('def e := throwError "bad {sorry}"\n'), ['sorry'])
        self.assertEqual(tokens('def e := throwError "axiom {ax} in {name}"\n'), [])
        self.assertEqual(tokens('def s := "a lone { brace"\ntheorem t : True := trivial\n'), [])

    def test_skip_kernel_tc(self):
        self.assertEqual(tokens('set_option debug.skipKernelTC true\n'), ['skipKernelTC'])

    def test_of_reduce_nat_and_trust_compiler(self):
        self.assertEqual(tokens('theorem t : True := Lean.ofReduceNat f g h\n'), ['ofReduceNat'])
        self.assertEqual(tokens('theorem t : True := Lean.trustCompiler\n'), ['trustCompiler'])

    def test_add_decl_without_checking(self):
        self.assertEqual(tokens('def x := Environment.addDeclWithoutChecking\n'), ['addDeclWithoutChecking'])

    def test_native_config_field(self):
        self.assertEqual(tokens('example : P := by decide (config := { native := true })\n'), ['native:='])
        self.assertEqual(tokens('example : P := by decide (config := {kernel := false, native := true})\n'),
                         ['native:='])


class ScansTreesWithoutSilentSkips(unittest.TestCase):
    def _tree(self, files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for rel, data in files.items():
            path = Path(tmp.name, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return tmp.name

    def test_nested_directories_and_exact_counts(self):
        root = self._tree({
            'A.lean': b'theorem a : True := trivial\n',
            'deep/er/B.lean': b'theorem b : True := sorry\n',
            'deep/notes.md': b'sorry in markdown is not Lean\n',
        })
        result = scan.scan_roots([root])
        self.assertEqual(result['files_scanned'], 2)
        self.assertEqual([(Path(h['path']).name, h['line'], h['token']) for h in result['hits']],
                         [('B.lean', 1, 'sorry')])
        self.assertEqual(result['errors'], [])

    def test_undecodable_file_is_an_error_not_a_skip(self):
        root = self._tree({'Bad.lean': b'\xff\xfe sorry'})
        result = scan.scan_roots([root])
        self.assertEqual(result['files_scanned'], 1)
        self.assertEqual(len(result['errors']), 1)

    def test_lake_build_directory_is_excluded(self):
        # .lake holds fetched dependencies and build outputs, not repository sources.
        root = self._tree({'.lake/packages/x/X.lean': b'axiom x : False\n',
                           'Main.lean': b'def m := 1\n'})
        result = scan.scan_roots([root])
        self.assertEqual(result['files_scanned'], 1)
        self.assertEqual(result['hits'], [])


class CommandLine(unittest.TestCase):
    def _run(self, files, *extra):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for rel, data in files.items():
            path = Path(tmp.name, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        proc = subprocess.run([sys.executable, '-B', '-S', str(SCANNER), tmp.name, '--json', *extra],
                              capture_output=True, text=True)
        return proc.returncode, json.loads(proc.stdout) if proc.stdout.strip().startswith('{') else None

    def test_clean_tree_exits_zero(self):
        code, out = self._run({'A.lean': b'theorem a : True := trivial\n'})
        self.assertEqual(code, 0)
        self.assertEqual(out['blocking_hits'], 0)

    def test_hit_exits_one(self):
        code, out = self._run({'A.lean': b'theorem a : True := sorry\n'})
        self.assertEqual(code, 1)
        self.assertEqual(out['blocking_hits'], 1)

    def test_error_exits_one(self):
        code, _ = self._run({'A.lean': b'\xff'})
        self.assertEqual(code, 1)

    def test_unterminated_construct_is_an_error_exit(self):
        code, out = self._run({'A.lean': b'def s := "open\ntheorem t : False := sorry\n'})
        self.assertEqual(code, 1)
        self.assertEqual(len(out['errors']), 1)

    def test_allowed_axiom_is_reported_but_not_blocking(self):
        code, out = self._run({'A.lean': b'axiom ok : True\n'}, '--allow-axiom', 'A.lean:1')
        self.assertEqual(code, 0)
        self.assertEqual(len(out['hits']), 1)
        self.assertEqual(out['blocking_hits'], 0)

    def test_allow_list_does_not_cover_sorry(self):
        code, out = self._run({'A.lean': b'theorem a : True := sorry\n'}, '--allow-axiom', 'A.lean:1')
        self.assertEqual(code, 1)


if __name__ == '__main__':
    unittest.main()
