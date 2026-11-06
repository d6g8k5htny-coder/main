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
    def test_plain_macro_holes_do_not_recurse_on_nested_string_candidates(self):
        script = (
            "import importlib.util,sys\n"
            f"p={str(SCAN)!r}\n"
            "s=importlib.util.spec_from_file_location('bounded_scanner',p)\n"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
            "sys.setrecursionlimit(80)\n"
            "m.scan_text(('\\\"{ ' * 60) + '\\\"','M.lean')\n"
        )
        p = subprocess.run([sys.executable, '-B', '-S', '-c', script],
                           capture_output=True, text=True, timeout=5)
        last_error = p.stderr.rstrip().rsplit('\n', 1)[-1] if p.stderr else ''
        self.assertEqual(p.returncode, 0,
                         f"scanner subprocess exited {p.returncode}: {last_error}")

    def test_plain_macro_holes_do_not_repeat_overlapping_suffix_work(self):
        script = (
            "import importlib.util\n"
            f"p={str(SCAN)!r}\n"
            "s=importlib.util.spec_from_file_location('bounded_scanner',p)\n"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
            "m.scan_text(('\\\"{ ' * 30) + '\\\"','M.lean')\n"
        )
        try:
            p = subprocess.run([sys.executable, '-B', '-S', '-c', script],
                               capture_output=True, text=True, timeout=2)
        except subprocess.TimeoutExpired as exc:
            self.fail(f"overlapping ordinary-string hole scans exceeded 2 seconds: {exc}")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

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
    def test_native_field_inside_code_holes_keeps_field_context(self):
        for src in ['#check s!"{let _ : 1 = 1 := by decide (config := { native := (true) }); 0}"',
                    'def e := m!"{f { a := 1 } { native := (true) }}"',
                    'def e := throwError "bad {decide (config := { native := (true) })}"']:
            with self.subTest(src=src): self.assertEqual(kinds(src), ['native'])
        self.assertEqual(kinds('#check s!"{decide (config := { native := false })}"'), [])
    def test_native_false_requires_the_false_token(self):
        for expr in ['f alse', 'f /- c -/ alse', 'fa\n  lse', 'false x']:
            src = 'theorem bad : 1 = 1 := by decide (config := { native := ' + expr + ' })'
            with self.subTest(src=src): self.assertEqual(kinds(src), ['native'])
        self.assertEqual(kinds('by decide (config := { native := /- c -/ false })'), [])
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

def fail_block(test):
    return ('='*70+'\nFAIL: '+test+' (tests.C.'+test+')\n'+'-'*70+'\nTraceback (most recent call last):\n'
            '  File "/w/tests/t.py", line 9, in '+test+'\n    self.assertEqual(1, 2)\nAssertionError: 1 != 2\n\n')

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
            good=fail_block('test_expected')+'-'*70+'\nRan 1 test in 0.001s\n\nFAILED (failures=1)\n'
            self.assertTrue(mod.killed(subprocess.CompletedProcess([],1,stdout='',stderr=good),'test_expected'))
            a8=[(2,good),  # wrong nonzero exit
                (1,fail_block('test_expected')+fail_block('test_other')+'-'*70+'\nRan 1 test in 0.001s\n\nFAILED (failures=2)\n'),
                (1,good.replace('Ran 1 test','Ran 3 tests')),  # the designated test must run alone
                (1,good.replace(', in test_expected',', in setUp')),  # raised outside the designated method
                (1,good.replace('FAILED (failures=1)','FAILED (failures=2)')),
                # an AssertionError raised inside the tool under test is a crash, not the test's assertion
                (1,good.replace('    self.assertEqual(1, 2)\n','    self.assertEqual(1, 2)\n'
                                '  File "/tmp/m/mutant_00.py", line 5, in classify_token\n    raise AssertionError\n'))]
            for code,out in a8:
                with self.subTest(module=mod.__name__,code=code,out=out):
                    self.assertFalse(mod.killed(subprocess.CompletedProcess([],code,stdout='',stderr=out),'test_expected'))
    def test_actual_syntax_broken_tool_is_not_a_kill(self):
        with tempfile.TemporaryDirectory() as d:
            broken=Path(d)/'broken.py';broken.write_text('def broken(:\n')
            for testfile in ['test_cbz_layer_a_scan_mutants.py','test_cbz_a3_float_census.py']:
                mod=load(ROOT/'tests'/testfile,'broken_harness')
                proc=mod.run_suite(str(broken))
                self.assertNotEqual(proc.returncode,0)
                self.assertFalse(mod.killed(proc),'syntax-broken source must remain an invalid mutant')

C = load(CENSUS, 'hardening_census')


def census_kinds(src):
    r = C.census_text(src, 'x.py')
    return r['class'], [k['kind'] for k in r['signals']]


def git_init(d):
    subprocess.run(['git','init','-q',str(d)],check=True,capture_output=True,
                   env=dict(os.environ,GIT_CONFIG_GLOBAL=os.devnull,GIT_CONFIG_SYSTEM=os.devnull))


class ReviewParityULW29(unittest.TestCase):
    """Findings first raised by automated review on ULW#29, the consolidated source."""
    def test_fail_on_rejects_unknown_contexts(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'A.lean').write_text('example : False := by sorry\n')
            for bad in ['prodution', '', 'production,', ',control', 'production,nope']:
                with self.subTest(fail_on=bad):
                    p = cli(SCAN, d, '--fail-on', bad)
                    self.assertEqual(p.returncode, 2, p.stdout+p.stderr)
            self.assertEqual(cli(SCAN, d, '--fail-on', 'production,control').returncode, 1)
            self.assertEqual(cli(SCAN, d, '--fail-on', 'evidence_copy').returncode, 0)

    def test_context_does_not_depend_on_path_spelling(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)/'checkout'; (repo/'raw').mkdir(parents=True); git_init(repo)
            (repo/'raw'/'A.lean').write_text('theorem t : True := sorry\n')
            for arg in [repo, repo/'raw', repo/'raw'/'A.lean']:
                with self.subTest(arg=str(arg)):
                    out = Path(d)/'o.json'
                    p = cli(SCAN, arg, '--json', out)
                    self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
                    findings = json.loads(out.read_text())['findings']
                    self.assertEqual([(f['context'], f['context_path']) for f in findings],
                                     [('evidence_copy', 'raw/A.lean')])

    def test_unrelated_ancestor_does_not_change_context(self):
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d)/'archive'/'project'; proj.mkdir(parents=True)
            (proj/'P.lean').write_text('theorem t : True := sorry\n')
            for arg in [proj/'P.lean', proj]:
                with self.subTest(arg=str(arg)):
                    p = cli(SCAN, arg)
                    self.assertEqual(p.returncode, 1, p.stdout+p.stderr)
                    self.assertIn('P.lean:1:21: A1 sorry (production)', p.stdout)

    def test_explicit_context_root(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as other:
            proj = Path(d)/'archive'/'project'; proj.mkdir(parents=True)
            src = proj/'P.lean'; src.write_text('theorem t : True := sorry\n')
            out = Path(d)/'o.json'
            p = cli(SCAN, src, '--context-root', d, '--json', out)
            self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
            f = json.loads(out.read_text())['findings'][0]
            self.assertEqual((f['context'], f['context_path']), ('evidence_copy', 'archive/project/P.lean'))
            self.assertEqual(cli(SCAN, src, '--context-root', other).returncode, 2)

    def test_census_alias_imported_after_use(self):
        self.assertEqual(census_kinds('def f(n):\n    return m.sqrt(n)\nimport math as m\n'),
                         ('FLOAT_SIGNAL', ['math_float']))
        cls, sig = census_kinds('def f():\n    return mp.iv\nimport mpmath as mp\n')
        self.assertIn('exact_module', sig)

    def test_census_interval_module_from_parent_package(self):
        for src in ['from . import ia\nx = ia.up(0.5)\n', 'from pkg import ia\nx = ia.up(0.5)\n',
                    'from pkg import ia as I\nx = I.up(0.5)\n', 'from .. import other, ia\nx = 0.5\n']:
            with self.subTest(src=src):
                cls, sig = census_kinds(src)
                self.assertEqual(cls, 'MIXED')
                self.assertIn('interval_layer', sig)
        self.assertEqual(census_kinds('from pkg import iax\nx = 0.5\n')[0], 'FLOAT_SIGNAL')

    def test_census_declared_encodings(self):
        latin = b"# -*- coding: latin-1 -*-\n\xe9t\xe9 = 1\nx = 0.5\n"
        self.assertEqual(C.census_text(b'\xef\xbb\xbfx = 0.5\n', 'bom.py')['class'], 'FLOAT_SIGNAL')
        self.assertEqual(C.census_text(latin, 'latin.py')['class'], 'FLOAT_SIGNAL')
        self.assertEqual(C.census_text(b"x = '\xff\xfe'\n", 'bad.py')['class'], 'UNPARSEABLE')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'a.py').write_bytes(b'\xef\xbb\xbfx = 0.5\n'); (root/'b.py').write_bytes(latin)
            out = root/'o.json'
            p = cli(CENSUS, root, '--json', out)
            self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
            self.assertEqual(sorted((f['path'], f['class']) for f in json.loads(out.read_text())['files']),
                             [('a.py', 'FLOAT_SIGNAL'), ('b.py', 'FLOAT_SIGNAL')])
            subprocess.run(['git','-c','user.name=t','-c','user.email=t@x','-c','commit.gpgsign=false','init','-q'],cwd=d,check=True,capture_output=True)
            subprocess.run(['git','-c','user.name=t','-c','user.email=t@x','-c','commit.gpgsign=false','add','a.py','b.py'],cwd=d,check=True,capture_output=True)
            subprocess.run(['git','-c','user.name=t','-c','user.email=t@x','-c','commit.gpgsign=false','commit','-qm','f'],cwd=d,check=True,capture_output=True)
            p = cli(CENSUS, '--git', root, '--json', out)
            self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
            self.assertEqual(sorted((f['path'], f['class']) for f in json.loads(out.read_text())['files']),
                             [('a.py', 'FLOAT_SIGNAL'), ('b.py', 'FLOAT_SIGNAL')])

if __name__=='__main__': unittest.main()
