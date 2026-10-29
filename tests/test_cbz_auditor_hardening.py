"""Regression union for canonical lexical/census auditors (CBZ §2.3).

These assert source-level behavior only; kernel and semantic closure are separate.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCAN = Path(os.environ.get('CBZ_LAYER_A_SCAN', ROOT/'tools/cbz/layer_a_scan.py'))
CENSUS = Path(os.environ.get('CBZ_A3_CENSUS', ROOT/'tools/cbz/a3_float_census.py'))
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
S = load(SCAN, 'hardening_scanner')

def kinds(src):
    return [f['kind'] for f in S.scan_text(src, 'Pkg/M.lean') if f['severity']=='violation']
def cli(tool, *args):
    return subprocess.run([sys.executable,'-B','-S',str(tool),*map(str,args)],capture_output=True,text=True)

class LexicalUnion(unittest.TestCase):
    def test_native_configuration_layout_is_not_a_bypass(self):
        for src in ['by decide (config := { native\n := true })',
                    'by decide (config := { native /- a /- b -/ c -/ := true })',
                    'by decide (config := { native := /- x -/ true })',
                    'by decide (config := { native -- comment\n := -- c\n true })',
                    'by decide (config := { native' + ' '*100 + ':=true })']:
            with self.subTest(src=src):
                self.assertEqual(kinds(src), ['native'])
    def test_plus_native_layout_is_not_a_bypass(self):
        for src in ['by decide +\nnative', 'by decide + /- c -/ native', 'by decide + --c\n native']:
            with self.subTest(src=src): self.assertEqual(kinds(src), ['native'])
    def test_native_false_and_longer_identifiers_are_not_true(self):
        for src in ['native := false', 'native := trueX', 'native := true₁', 'native := trueα']:
            with self.subTest(src=src): self.assertEqual(kinds(src), [])
    def test_unknown_native_field_expression_fails_closed(self):
        for expr in ['(true)','false || true','if true then true else false',
                     '(false)','Bool.not false','flag',
                     'false /- comment -/ || true']:
            for prefix in ['{ ', '{ kernel := false, ', '{ cfg with ']:
                src='by decide (config := '+prefix+'native := '+expr+' })'
                with self.subTest(src=src): self.assertEqual(kinds(src),['native'])
        self.assertEqual(kinds('by decide (config := { native := false })'),[])
        self.assertEqual(kinds('by decide (config := { native := false, kernel := true })'),[])
        self.assertEqual(kinds('def native := 1'),[])
    def test_plain_macro_braces_are_conservatively_scanned(self):
        self.assertEqual(kinds('def e := throwError "bad {sorry}"'), ['sorry'])
        self.assertEqual(kinds('def e := throwError "bad {f {x := admit}}"'), ['admit'])
        self.assertEqual(kinds('def e := throwError "axiom {ax} in {name}"'), [])
        self.assertEqual(kinds('def s := "a lone { brace"'), [])
        self.assertEqual(kinds(r'def s := "escaped \{sorry}"'), [])
    def test_plain_macro_holes_with_nested_literals(self):
        for src in ['def f := throwError "bad {let x := "safe"; (sorry : Nat)}"',
                    'def f := throwError "bad {let x := "safe"; /- } -/ (sorry : Nat)}"',
                    'def f := throwError "bad {let «odd}name» := "safe"; (sorry : Nat)}"',
                    'def f := throwError "bad {s!"{("x", (sorry : Nat))}"}"']:
            with self.subTest(src=src): self.assertIn('sorry',kinds(src))
    def test_adjacent_numeral_never_swallows_code_keyword(self):
        self.assertEqual(kinds('def harmless := 0axiom hiddenPremise : False\ntheorem bad : False := hiddenPremise'),['axiom'])
        for src in ['def x := 1sorry', 'def x := 0native_decide', 'def x := 123admit']:
            with self.subTest(src=src):
                self.assertTrue(kinds(src))
    def test_scientific_and_radix_numerals_preserve_keyword_boundary(self):
        for number in ['1e1','1E1','1.0e1','1.0E1','1e+1','1E-1',
                       '1_0e1_0','0b0','0B1_0','0o0','0O1_0']:
            src=f'def harmless := {number}axiom hiddenPremise : False'
            with self.subTest(src=src): self.assertEqual(kinds(src),['axiom'])
        for number in ['0x0','0Xf_F','1.5e-2']:
            with self.subTest(number=number):
                self.assertEqual(kinds(f'def x := {number}sorry'),['sorry'])
    def test_mathlib_prime_notations_are_not_unfinished_characters(self):
        for src in ["def x := f '' S", "def x := f ⁻¹' S", "def x := ∑' n, f n",
                    "theorem t := hb.1.integrable_norm_pow'", "def x := f'' y"]:
            with self.subTest(src=src):
                self.assertEqual([f for f in S.scan_text(src,'M.lean') if f['severity']=='error'],[])
        for src in [r"def c : Char := '\u0041'", r"def c : Char := '\x41'"]:
            with self.subTest(src=src):
                self.assertEqual([f for f in S.scan_text(src,'M.lean') if f['severity']=='error'],[])
    def test_interpolation_prefix_accepts_lean_layout(self):
        # Both were accepted by the actual pinned Lean4.34.1 parser.
        for src in ['#check s! "{("x", (sorry : Nat))}"',
                    '#check s! /- layout -/ "{("x", (sorry : Nat))}"',
                    '#check m! -- layout\n "{("x", (sorry : Nat))}"']:
            with self.subTest(src=src): self.assertEqual(kinds(src), ['sorry'])
        for src in ['def s := s! "{x"', 'def s := s! /- layout -/ "{x"']:
            with self.subTest(src=src):
                self.assertTrue([f for f in S.scan_text(src,'M.lean') if f['severity']=='error'])
    def test_unterminated_character_literal_is_error(self):
        for src in ["def c := '", "def c := '\\", "def c := '\\n", "def c := '\"\n"]:
            with self.subTest(src=src):
                self.assertIn('unterminated_char', [f['kind'] for f in S.scan_text(src,'M.lean')])
    def test_unicode_identifier_prefix_does_not_hide_keyword(self):
        # Greek lambda is a binder token, not an identifier letter in Lean.
        self.assertEqual(kinds('def f := λsorry => sorry'), ['sorry','sorry'])
    def test_pinned_lean_identifier_character_domain(self):
        # Lean4.34.1 Init/Meta/Defs.lean:101-143, not Unicode category L.
        for prefix in ['λ','Π','Σ','Ж','ᵀ','𝛂']:
            with self.subTest(prefix=prefix): self.assertEqual(kinds(prefix+'sorry'),['sorry'])
        for prefix in ['α','é','Ā','ℵ','𝔸']:
            with self.subTest(prefix=prefix): self.assertEqual(kinds(prefix+'sorry'),[])
        self.assertEqual(kinds('native_decideⱼ'),[])
    def test_unsupported_character_escape_is_error(self):
        for src in [r"def c := '\q'",r"def c := '\u{41}'"]:
            with self.subTest(src=src):
                self.assertTrue([f for f in S.scan_text(src,'M.lean') if f['severity']=='error'])

class InputCoverage(unittest.TestCase):
    def test_missing_root_fails_for_both_tools(self):
        with tempfile.TemporaryDirectory() as d:
            for tool in [SCAN,CENSUS]:
                with self.subTest(tool=tool):
                    p=cli(tool,Path(d)/'does-not-exist')
                    self.assertEqual(p.returncode,2,p.stdout+p.stderr)
    def test_source_symlink_fails_for_both_tools(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for tool,suffix in [(SCAN,'.lean'),(CENSUS,'.py')]:
                target=root/('target'+suffix);target.write_text('')
                link=root/('link'+suffix);link.symlink_to(target)
                with self.subTest(tool=tool):
                    p=cli(tool,link)
                    self.assertEqual(p.returncode,2,p.stdout+p.stderr)
    def test_symlink_directory_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);actual=root/'actual';actual.mkdir();(root/'alias').symlink_to(actual,target_is_directory=True)
            for tool in [SCAN,CENSUS]:
                with self.subTest(tool=tool): self.assertEqual(cli(tool,root).returncode,2)
    def test_git_source_symlink_fails_for_both_tools(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def git(*a):
                return subprocess.run(['git','-c','user.name=t','-c','user.email=t@x','-c','commit.gpgsign=false',*a],cwd=d,check=True,capture_output=True)
            git('init','-q');(root/'A.lean').symlink_to('other');(root/'a.py').symlink_to('other')
            git('add','-A');git('commit','-qm','fixture')
            for tool in [SCAN,CENSUS]:
                with self.subTest(tool=tool): self.assertEqual(cli(tool,'--git',root).returncode,2)
    def test_gitlink_is_explicitly_rejected_by_both_tools(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def git(*a): return subprocess.run(['git','-c','user.name=t','-c','user.email=t@x','-c','commit.gpgsign=false',*a],cwd=d,check=True,capture_output=True)
            git('init','-q');(root/'README').write_text('x');git('add','-A');git('commit','-qm','fixture')
            sha=git('rev-parse','HEAD').stdout.decode().strip();git('update-index','--add','--cacheinfo',f'160000,{sha},sub');git('commit','-qm','gitlink')
            for tool in [SCAN,CENSUS]:
                with self.subTest(tool=tool): self.assertEqual(cli(tool,'--git',root).returncode,2)


class GitObjectTransport(unittest.TestCase):
    def test_malformed_blob_transport_fails_closed(self):
        import io
        from unittest.mock import patch
        sha='a'*40
        class Proc:
            def __init__(self, payload, code):
                self.stdin=io.BytesIO();self.stdout=io.BytesIO(payload);self.stderr=io.BytesIO();self.code=code
            def poll(self): return self.code
            def wait(self): return self.code
            def terminate(self): self.code=-15
        for tool,fn,suffix in [(SCAN,'git_lean_blobs','.lean'),(CENSUS,'_git_blobs','.py')]:
            mod=load(tool,'transport_test')
            tree=subprocess.CompletedProcess([],0,stdout=f'100644 blob {sha}\tA{suffix}\0'.encode())
            bad=[(f'{sha} tree 1\nx\n'.encode(),0),
                 (f'{"b"*40} blob 1\nx\n'.encode(),0),
                 (f'{sha} blob 8\nx\n'.encode(),0),
                 (f'{sha} blob 1\nxX'.encode(),0),
                 (f'{sha} blob 1\nx\n'.encode(),7)]
            for payload,code in bad:
                with self.subTest(tool=tool,payload=payload,code=code):
                    with patch.object(mod.subprocess,'run',return_value=tree),patch.object(mod.subprocess,'Popen',return_value=Proc(payload,code)):
                        with self.assertRaises(OSError): list(getattr(mod,fn)('repo','HEAD'))

@unittest.skipIf(os.environ.get("CBZ_LAYER_A_SCAN") or os.environ.get("CBZ_A3_CENSUS"), "mutation subprocess: harness controls run only at top level")
class MutationKillClassification(unittest.TestCase):
    def test_nonbehavioral_failures_do_not_count_as_kills(self):
        modules=[load(ROOT/'tests/test_cbz_layer_a_scan_mutants.py','scan_mutants'),load(ROOT/'tests/test_cbz_a3_float_census.py','a3_mutants')]
        samples=['SyntaxError: invalid syntax',
                 'Ran 1 test in 0.0s\nFAILED (errors=1)\nERROR: test_expected (tests.C)\n',
                 'Ran 3 tests in 0.0s\nFAILED (failures=1, errors=1)\nFAIL: test_expected (tests.C)\n',
                 'Ran 3 tests in 0.0s\nFAILED (failures=1)\nFAIL: test_other (tests.C)\n']
        for mod in modules:
            for out in samples:
                with self.subTest(module=mod.__name__,out=out):
                    self.assertFalse(mod.killed(subprocess.CompletedProcess([],1,stdout='',stderr=out),'test_expected'))
            good='Ran 3 tests in 0.0s\nFAILED (failures=1)\nFAIL: test_expected (tests.C)\n'
            self.assertTrue(mod.killed(subprocess.CompletedProcess([],1,stdout='',stderr=good),'test_expected'))
    def test_actual_syntax_broken_tool_is_not_a_kill(self):
        with tempfile.TemporaryDirectory() as d:
            broken=Path(d)/'broken.py';broken.write_text('def broken(:\n')
            for testfile in ['test_cbz_layer_a_scan_mutants.py','test_cbz_a3_float_census.py']:
                mod=load(ROOT/'tests'/testfile,'broken_harness')
                proc=mod.run_suite(str(broken))
                self.assertNotEqual(proc.returncode,0)
                self.assertFalse(mod.killed(proc),'syntax-broken source must remain an invalid mutant')

if __name__=='__main__': unittest.main()
