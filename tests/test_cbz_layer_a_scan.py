"""Tests for the CBZ-0 Layer A scanner (gate G1); the mutation harness (gate G0) reruns them.

CBZ_LAYER_A_SCANNER may name an alternative scanner file; the mutation harness
uses it to run this suite against each mutant. Scientific effect: NONE.
"""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNER = Path(os.environ.get('CBZ_LAYER_A_SCANNER', ROOT / 'tools/cbz/layer_a_scan.py'))


def load():
    spec = importlib.util.spec_from_file_location('cbz_layer_a_scan_under_test', SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scanner = load()


def tokens(text):
    hits, _, errors = scanner.scan_text(text)
    return [h['token'] for h in hits], errors


# The suite's own copy of the rule table: deleting or recategorizing a scanner rule must fail.
EXPECTED_RULES = {
    'sorry': 'A1.sorry', 'admit': 'A1.admit', 'sorryAx': 'A1.sorry', 'stop': 'A1.stop',
    'mkSorry': 'A1.sorry', 'mkSyntheticSorry': 'A1.sorry', 'mkLabeledSorry': 'A1.sorry',
    'exact?': 'A1.search', 'apply?': 'A1.search', 'rw?': 'A1.search', 'simp?': 'A1.search',
    'simp_all?': 'A1.search', 'aesop?': 'A1.search',
    'axiom': 'A2.axiom', 'native_decide': 'A2.native', 'native': 'A2.native',
    'ofReduceBool': 'A2.native', 'ofReduceNat': 'A2.native', 'reduceBool': 'A2.native',
    'reduceNat': 'A2.native', 'implemented_by': 'A2.trusted-code', 'extern': 'A2.trusted-code',
    'skipKernelTC': 'A2.kernel-bypass', 'addDecl': 'A2.kernel-bypass', 'addAndCompile': 'A2.kernel-bypass',
    'Float': 'A3.float', 'Float32': 'A3.float',
}


class PlantedViolations(unittest.TestCase):
    """Every rule fires in plain code."""

    def test_every_expected_rule_is_detected_with_its_category(self):
        for token, rule in EXPECTED_RULES.items():
            with self.subTest(token=token):
                hits, _, errors = scanner.scan_text(f'theorem t : True := by\n  {token}\n')
                self.assertEqual([(h['token'], h['rule']) for h in hits], [(token, rule)])
                self.assertEqual(errors, [])

    def test_rule_table_has_no_untested_entries(self):
        self.assertEqual(scanner.RULES, EXPECTED_RULES)

    def test_exit_command_is_detected(self):
        found, _ = tokens('theorem a : True := trivial\n#exit\ntheorem b : False := trivial\n')
        self.assertEqual(found, ['#exit'])

    def test_rule_categories(self):
        self.assertEqual(scanner.RULES['sorry'], 'A1.sorry')
        self.assertEqual(scanner.RULES['axiom'], 'A2.axiom')
        self.assertEqual(scanner.RULES['native_decide'], 'A2.native')
        self.assertEqual(scanner.RULES['Float'], 'A3.float')
        self.assertEqual(scanner.COMMANDS['#exit'], 'A1.exit')

    def test_position_is_exact(self):
        hits, _, _ = scanner.scan_text('-- c\ntheorem t : True := by\n  exact sorry\n')
        self.assertEqual([(h['line'], h['col']) for h in hits], [(3, 9)])

    def test_several_hits_are_all_reported(self):
        found, _ = tokens('axiom ax : False\ntheorem t : 1 = 2 := by native_decide\ndef f := sorry\n')
        self.assertEqual(found, ['axiom', 'native_decide', 'sorry'])


class Masking(unittest.TestCase):
    """Comments, string text and char literals never hide or invent hits."""

    def test_line_comment(self):
        self.assertEqual(tokens('theorem t : True := trivial -- sorry\n')[0], [])

    def test_block_comment_and_doc_comment(self):
        self.assertEqual(tokens('/- sorry -/ /-- admit -/ /-! axiom -/ theorem t : True := trivial\n')[0], [])

    def test_nested_block_comment(self):
        self.assertEqual(tokens('/- a /- sorry -/ still comment sorry -/ theorem t : True := trivial\n')[0], [])

    def test_code_after_nested_comment_is_scanned(self):
        self.assertEqual(tokens('/- a /- b -/ c -/ theorem t : True := sorry\n')[0], ['sorry'])

    def test_string_text_is_masked(self):
        self.assertEqual(tokens('def m := "sorry admit axiom native_decide"\n')[0], [])

    def test_escaped_quote_stays_inside_string(self):
        self.assertEqual(tokens('def m := "a \\" sorry"\n')[0], [])

    def test_escaped_backslash_ends_string(self):
        self.assertEqual(tokens('def m := "a \\\\" ++ sorry\n')[0], ['sorry'])

    def test_char_literal_quote_does_not_open_a_string(self):
        self.assertEqual(tokens("def c := '\"'\ntheorem t : True := sorry\n")[0], ['sorry'])

    def test_char_literals_are_masked(self):
        self.assertEqual(tokens("def c := 'a'\ndef d := '\\n'\n")[0], [])

    def test_prime_in_identifier_is_not_a_char_literal(self):
        self.assertEqual(tokens("theorem t' : True := sorry\ntheorem u' : True := trivial\n")[0], ['sorry'])

    def test_raw_strings_are_masked(self):
        self.assertEqual(tokens('def m := r"sorry"\ndef n := r#"admit "inner" axiom"#\n')[0], [])

    def test_raw_string_has_no_escapes(self):
        found, errors = tokens('def m := r"a\\" ++ sorry\n')
        self.assertEqual((found, errors), (['sorry'], []))

    def test_prime_followed_by_char_like_text_is_one_name(self):
        self.assertEqual(tokens("def sorry'x' := 1\n")[0], [])

    def test_code_after_raw_string_is_scanned(self):
        self.assertEqual(tokens('def m := r#"x"# ++ sorry\n')[0], ['sorry'])

    def test_comment_markers_inside_strings_are_text(self):
        self.assertEqual(tokens('def m := "-- /-"\ntheorem t : True := sorry\n')[0], ['sorry'])


class Interpolation(unittest.TestCase):
    """Code inside string interpolation is code."""

    def test_s_bang_interpolant(self):
        self.assertEqual(tokens('def m := s!"value {sorry} done"\n')[0], ['sorry'])

    def test_m_bang_nested_string_in_interpolant(self):
        self.assertEqual(tokens('def m := m!"a {f "sorry text"} {admit} b"\n')[0], ['admit'])

    def test_throw_error_interpolates_plain_literal(self):
        found, _ = tokens('def x := do\n  throwError "forbidden axiom {sorry} in {name}"\n')
        self.assertEqual(found, ['sorry'])

    def test_throw_error_nested_string_stays_inside_interpolant(self):
        found, errors = tokens('def x := do\n  throwError "x {f "sorry"} y"\n')
        self.assertEqual((found, errors), ([], []))

    def test_throw_error_text_is_masked(self):
        self.assertEqual(tokens('def x := do\n  throwError "forbidden transitive axiom {ax} in {name}"\n')[0], [])

    def test_plain_string_braces_are_scanned_conservatively(self):
        self.assertEqual(tokens('def m := "{sorry}"\n')[0], ['sorry'])

    def test_escaped_brace_is_text(self):
        self.assertEqual(tokens('def m := s!"\\{sorry}"\n')[0], [])

    def test_interpolant_followed_by_code(self):
        self.assertEqual(tokens('def m := s!"{x}" ++ toString sorry\n')[0], ['sorry'])


class Identifiers(unittest.TestCase):
    """Name boundaries follow Lean 4."""

    def test_longer_names_are_other_names(self):
        self.assertEqual(tokens('theorem sorry_free : True := not_sorry\ndef admits := sorry₁\n')[0], [])

    def test_prime_and_letter_like_suffixes(self):
        self.assertEqual(tokens("def sorry' := 1\ndef sorryα := 2\ndef native_x := 3\n")[0], [])

    def test_lambda_is_not_an_identifier_character(self):
        self.assertEqual(tokens('def f := sorryλ x => x\n')[0], ['sorry'])

    def test_pi_and_sigma_are_not_identifier_characters(self):
        self.assertEqual(tokens('def f := sorryΠ\ndef g := admitΣ\n')[0], ['sorry', 'admit'])

    def test_dotted_component_matches(self):
        self.assertEqual(tokens('def f := Foo.sorry\ndef g := Lean.ofReduceBool\n')[0], ['sorry', 'ofReduceBool'])

    def test_quoted_identifier_matches_conservatively(self):
        self.assertEqual(tokens('def «sorry» := 1\n')[0], ['sorry'])

    def test_digits_do_not_absorb_a_following_name(self):
        self.assertEqual(tokens('def f := 2sorry\ndef g := 0x1F\n')[0], ['sorry'])

    def test_decide_plus_native(self):
        self.assertEqual(tokens('theorem t : 2 + 2 = 4 := by decide +native\n')[0], ['native'])

    def test_kernel_bypass_option(self):
        self.assertEqual(tokens('set_option debug.skipKernelTC true in\ntheorem t : True := trivial\n')[0], ['skipKernelTC'])

    def test_trusted_code_attributes(self):
        self.assertEqual(tokens('@[implemented_by g] def f := 1\n@[extern "c"] def h := 2\n')[0], ['implemented_by', 'extern'])

    def test_search_tactics(self):
        self.assertEqual(tokens('theorem t : True := by exact?\ntheorem u : True := by apply?\n')[0], ['exact?', 'apply?'])

    def test_print_axioms_is_not_an_axiom(self):
        self.assertEqual(tokens('#print axioms foo\n#check Lean.ConstantInfo.axiomInfo\n')[0], [])

    def test_exit_inside_comment_or_name(self):
        self.assertEqual(tokens('-- #exit\ndef f := 1 #exit_like\n')[0], [])


class FailClosed(unittest.TestCase):
    def test_unterminated_block_comment(self):
        self.assertTrue(tokens('/- open\ntheorem t : True := sorry\n')[1])

    def test_unterminated_string(self):
        self.assertTrue(tokens('def m := "open\n')[1])

    def test_unterminated_interpolation(self):
        self.assertTrue(tokens('def m := s!"{open"\n')[1])

    def test_unterminated_quoted_identifier(self):
        self.assertTrue(tokens('def «open := 1\n')[1])


def git(path, *args):
    subprocess.run(['git', '-C', str(path), *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Repository(unittest.TestCase):
    """End to end through the CLI on a real Git repository."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / 'r'
        self.repo.mkdir()
        git(self.repo, 'init', '-q')
        git(self.repo, 'config', 'user.email', 'cbz@example.invalid')
        git(self.repo, 'config', 'user.name', 'cbz')
        (self.repo / 'Pkg').mkdir()
        (self.repo / 'Pkg/Good.lean').write_text('theorem good : True := trivial\n', encoding='utf-8')
        (self.repo / 'Pkg/Control.lean').write_text('theorem bad : False := sorry\n', encoding='utf-8')
        (self.repo / 'Pkg/notes.md').write_text('sorry in prose is not Lean\n', encoding='utf-8')
        git(self.repo, 'add', '-A')
        git(self.repo, 'commit', '-q', '-m', 'init')
        self.registry = Path(self.tmp.name) / 'exemptions.json'

    def tearDown(self):
        self.tmp.cleanup()

    def write_registry(self, rows):
        self.registry.write_text(json.dumps({'schema': 1, 'exemptions': rows}), encoding='utf-8')

    def run_cli(self, *extra):
        out = Path(self.tmp.name) / 'report.json'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = scanner.main(['--repo', f'R={self.repo}@HEAD', '--exemptions', str(self.registry),
                                 '--json', str(out), *extra])
        return code, json.loads(out.read_text(encoding='utf-8'))

    def run_cli_code(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return scanner.main(list(args))

    def exemption(self, **kw):
        row = {'repository': 'R', 'path': 'Pkg/Control.lean', 'token': 'sorry', 'count': 1,
               'reason': 'negative control that must be refused by the gate', 'isolated': True}
        row.update(kw)
        return row

    def test_unexempted_hit_fails(self):
        self.write_registry([])
        code, report = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertEqual(report['summary']['violations'], 1)
        self.assertEqual(report['repositories'][0]['lean_files'], 2)

    def test_exact_exemption_passes(self):
        self.write_registry([self.exemption()])
        code, report = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual(report['summary'], {'lean_files': 2, 'hits': 1, 'exempt_hits': 1, 'violations': 0,
                                             'errors': 0, 'exemption_errors': 0, 'isolation_errors': 0})

    def test_count_mismatch_exempts_nothing(self):
        self.write_registry([self.exemption(count=2)])
        code, report = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(report['summary']['violations'], 1)
        self.assertEqual(report['summary']['exemption_errors'], 1)

    def test_new_hit_in_exempted_file_is_not_hidden(self):
        self.write_registry([self.exemption()])
        (self.repo / 'Pkg/Control.lean').write_text('theorem bad : False := sorry\ntheorem worse : False := sorry\n',
                                                     encoding='utf-8')
        git(self.repo, 'commit', '-q', '-am', 'second hit')
        code, report = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(report['summary']['violations'], 2)

    def test_missing_exempted_path_is_an_error(self):
        self.write_registry([self.exemption(), self.exemption(path='Pkg/Gone.lean')])
        code, report = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(report['summary']['exemption_errors'], 1)

    def test_duplicate_exemption_is_an_error(self):
        self.write_registry([self.exemption(), self.exemption()])
        code, _ = self.run_cli()
        self.assertEqual(code, 1)

    def test_isolated_fixture_must_not_be_imported(self):
        self.write_registry([self.exemption()])
        (self.repo / 'Pkg/Good.lean').write_text('import Pkg.Control\ntheorem good : True := trivial\n',
                                                  encoding='utf-8')
        git(self.repo, 'commit', '-q', '-am', 'import the fixture')
        code, report = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(report['isolation_errors'][0]['imports'], 'Pkg.Control')

    def test_committed_bytes_not_working_tree(self):
        self.write_registry([self.exemption()])
        (self.repo / 'Pkg/Good.lean').write_text('theorem good : True := sorry\n', encoding='utf-8')
        code, report = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual(report['summary']['violations'], 0)

    def test_non_utf8_file_is_an_error(self):
        self.write_registry([self.exemption()])
        (self.repo / 'Pkg/Bytes.lean').write_bytes(b'theorem t : True := trivial \xff\n')
        git(self.repo, 'add', '-A')
        git(self.repo, 'commit', '-q', '-m', 'bytes')
        code, report = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(report['errors'][0]['error'], 'not UTF-8')

    def test_symlinked_lean_path_is_an_error(self):
        self.write_registry([self.exemption()])
        os.symlink('Good.lean', self.repo / 'Pkg/Link.lean')
        git(self.repo, 'add', '-A')
        git(self.repo, 'commit', '-q', '-m', 'link')
        code, report = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(report['errors'][0]['error'], 'symlinked Lean path not scanned')

    def test_submodule_is_an_error(self):
        self.write_registry([self.exemption()])
        head = subprocess.run(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], check=True,
                              stdout=subprocess.PIPE).stdout.decode().strip()
        git(self.repo, 'update-index', '--add', '--cacheinfo', f'160000,{head},vendor')
        git(self.repo, 'commit', '-q', '-m', 'gitlink')
        code, report = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(report['errors'][0]['error'], 'submodule not scanned')

    def test_invalid_registry_is_refused(self):
        self.registry.write_text(json.dumps({'schema': 1, 'exemptions': [{'repository': 'R'}]}), encoding='utf-8')
        code = self.run_cli_code('--repo', f'R={self.repo}@HEAD', '--exemptions', str(self.registry))
        self.assertEqual(code, 2)

    def test_short_reason_is_refused(self):
        self.write_registry([self.exemption(reason='because')])
        code = self.run_cli_code('--repo', f'R={self.repo}@HEAD', '--exemptions', str(self.registry))
        self.assertEqual(code, 2)


if __name__ == '__main__':
    unittest.main()
