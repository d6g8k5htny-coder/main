"""Behavioral contracts; each test names a scanner weakening it catches."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
try:
    import auditor
    import source_checks
except ImportError:
    auditor = source_checks = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SourceRules(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(source_checks, 'source checker implementation is missing')

    def lean_codes(self, text):
        return {x['code'] for x in source_checks.check_lean(text)['findings']}

    def python_codes(self, text, mathematical=True):
        return {x['code'] for x in source_checks.check_python(text, mathematical)['findings']}

    def test_lean_each_logical_escape_and_qualified_compiler_axiom_is_detected(self):
        for word, code in [('sorry','LEAN_PLACEHOLDER'),('admit','LEAN_PLACEHOLDER'),
                           ('sorryAx','LEAN_PLACEHOLDER'),('axiom fake : False','LEAN_AXIOM'),
                           ('constant fake : False','LEAN_AXIOM'),('native_decide','LEAN_NATIVE'),
                           ('decide +native','LEAN_NATIVE'),('Lean.ofReduceBool','LEAN_NATIVE'),
                           ('Lean.ofReduceNat','LEAN_NATIVE'),('Lean.ofReduceInt','LEAN_NATIVE'),
                           ('Lean.trustCompiler','LEAN_NATIVE')]:
            with self.subTest(word=word):
                self.assertIn(code, self.lean_codes('theorem t : True := by ' + word))

    def test_comment_strings_nested_comments_raw_strings_and_char_are_not_code(self):
        text = '''-- sorry axiom bad : False
/- sorry /- native_decide -/ admit -/
def s := "sorry \\" admit /-"
def r := r##"native_decide \" axiom"##
def c := 'x'
theorem t : True := by trivial
'''
        self.assertEqual(self.lean_codes(text), set())

    def test_unicode_boundaries_and_escaped_identifiers_do_not_hide_exact_escape(self):
        self.assertEqual(self.lean_codes('def sorryα := 1\ndef αsorry := 2\ndef «sorry» := 3'), set())
        self.assertIn('LEAN_PLACEHOLDER', self.lean_codes('theorem λ : True := by\n\u2003sorry'))
        self.assertIn('LEAN_NATIVE', self.lean_codes('def f := Lean.«ofReduceBool»'))

    def test_lexical_incompleteness_and_interpolation_are_fatal_not_silent(self):
        for text in ['/- unclosed', 'def s := "open', 'def r := r##"open"#', 'def «open := 1',
                     'def s := s!"{by sorry}"']:
            with self.subTest(text=text):
                self.assertIn('LEAN_LEXICAL_GAP', self.lean_codes(text))

    def test_macro_expansion_and_unsafe_trust_require_separate_certifier(self):
        for text in ['macro "magic" : term => `(by trivial)', 'elab "magic" : term => pure ()',
                     'unsafe def x := 1', 'syntax "magic" : term', 'run_elab pure ()']:
            with self.subTest(text=text):
                self.assertIn('LEAN_EXPANSION_GAP', self.lean_codes(text))

    def test_adopted_public_lexer_covers_helpers_compiler_options_and_unicode_boundaries(self):
        for text,code in [('def x := Lean.Elab.admitGoal','LEAN_PLACEHOLDER'),
                          ('def x := Lean.Meta.mkSorry','LEAN_PLACEHOLDER'),
                          ('set_option debug.skipKernelTC true','LEAN_NATIVE'),
                          ('def sorryλ := 1','LEAN_PLACEHOLDER'),
                          ('def f (x : Float) := x','LEAN_FLOAT_REVIEW_GAP'),
                          ('@[extern "c"] def f := 1','LEAN_EXPANSION_GAP')]:
            with self.subTest(text=text):
                self.assertIn(code,self.lean_codes(text))

    def test_reported_kernel_escape_names_and_native_config_layouts(self):
        cases = [
            ('def x := Environment.addDeclWithoutChecking','LEAN_KERNEL_BYPASS'),
            ('def x := Lean.Environment.addDeclWithoutChecking','LEAN_KERNEL_BYPASS'),
            ('def x := Environment.«addDeclWithoutChecking»','LEAN_KERNEL_BYPASS'),
            ('def x := «addDeclWithoutChecking»','LEAN_KERNEL_BYPASS'),
            ('theorem t : True := by decide +native','LEAN_NATIVE'),
            ('theorem t : True := by decide + native','LEAN_NATIVE'),
            ('theorem t : True := by decide (config := { native := true })','LEAN_NATIVE'),
            ('theorem t : True := by decide (config := {\n native\n :=\n true\n})','LEAN_NATIVE'),
            ('theorem t : True := by decide (config := { native /- gap /- nested -/ -/ := /- gap -/ true })','LEAN_NATIVE'),
            ('theorem t : True := by decide (config := { «native» /- gap -/ := true })','LEAN_NATIVE'),
            ('def x := Lean.«ofReduceBool»','LEAN_NATIVE'),
            ('def x := «sorryAx»','LEAN_PLACEHOLDER'),
            ('def x := Lean.Elab.«admitGoal»','LEAN_PLACEHOLDER'),
            ('def x := Lean.Meta.«mkSorry»','LEAN_PLACEHOLDER'),
        ]
        for source,code in cases:
            with self.subTest(source=source):
                self.assertIn(code,self.lean_codes(source))

    def test_unterminated_character_literals_are_never_silent(self):
        for source in ["def x := 'a", "def x := '\\n", "def x := '\\", "def x := '"]:
            with self.subTest(source=source):
                self.assertIn('LEAN_LEXICAL_GAP',self.lean_codes(source))

    def test_new_kernel_escape_names_inside_comments_and_strings_are_not_code(self):
        source = '-- Environment.addDeclWithoutChecking\n/- «sorryAx» /- native := true -/ -/\ndef x := "Environment.addDeclWithoutChecking native := true"\n'
        self.assertEqual(self.lean_codes(source),set())

    def test_lean_explicit_imports_are_extracted_across_lines(self):
        result = source_checks.check_lean('import Closed.Base\nimport\n  Closed.A Closed.B\n-- import Q.Bad\n')
        self.assertEqual(result['imports'], ['Closed.Base','Closed.A','Closed.B'])

    def test_lean_vacuity_candidates_have_declaration_location(self):
        text = 'theorem t (h : False) : True := by trivial\nlemma u (h : P) (hn : ¬ P) : True := by trivial\nexample (n : Nat) (h : n < n) : True := by trivial\n'
        ds = source_checks.check_lean(text)['declarations']
        self.assertEqual([d['key'] for d in ds], ['t@1','u@2','example@3'])
        self.assertTrue(all(d['contradictions'] for d in ds))
        for hypothesis in ['(h : 1 = 0)', '(h : x ≠ x)', '(h : Not True)']:
            self.assertTrue(source_checks.check_lean('theorem x ' + hypothesis + ' : True := by trivial')['declarations'][0]['contradictions'])

    def test_axiom_audit_print_command_is_not_a_new_axiom_declaration(self):
        self.assertEqual(self.lean_codes('#print axioms Closed.t'),set())

    def test_canonical_metadata_and_error_channels_fail_closed(self):
        from unittest.mock import patch
        with patch.object(source_checks, 'canonical_identifier_tokens', return_value=iter([('x', 1, 1, False, False, '')])):
            self.assertIn('LEAN_LEXICAL_GAP', self.lean_codes('def x := 1'))
        def failed_lexing(text, errors):
            errors.append((1, 1, 'unsupported canonical lexical region'))
            return iter(())
        with patch.object(source_checks, 'canonical_identifier_tokens', failed_lexing):
            self.assertIn('LEAN_LEXICAL_GAP', self.lean_codes('def x := 1'))

    def test_canonical_numeric_boundaries_preserve_complete_owned_axiom_tokens(self):
        for numeral in ['0', '0b0', '0o0', '1e1', '1.0E+1']:
            with self.subTest(numeral=numeral):
                self.assertIn('LEAN_AXIOM', self.lean_codes('def harmless := ' + numeral + 'axiom hiddenPremise : False\n'))
        self.assertEqual(self.lean_codes('def n := 0x0a\ndef f := 1.0e+1\n'), set())

    def test_nested_string_placeholder_and_benign_literal_boundaries(self):
        self.assertIn('LEAN_PLACEHOLDER', self.lean_codes('def f := throwError "bad {let x := "safe"; (sorry : Nat)}"'))
        self.assertEqual(self.lean_codes('def s := "brace { harmless } and \\"quoted\\""\n'), set())

    def test_lean_no_semantic_sat_claim_from_uninterpreted_hypotheses(self):
        ds = source_checks.check_lean('theorem t (h : complicated P) : P := h')['declarations']
        self.assertEqual(ds[0]['contradictions'], [])

    def test_exact_integer_fraction_program_and_alias_are_accepted(self):
        text = '''from fractions import Fraction as F
# float(1.1) is only a comment
"""A mathematical docstring may mention float, sorry and native_decide."""
def square(n: int):
    return n ** 2
values = [F(square(n), 3) for n in range(5)]
assert sum(values) == F(10, 1)
'''
        self.assertEqual(self.python_codes(text), set())

    def test_every_inexact_python_route_is_rejected(self):
        for text, code in [('x=0.1','PY_INEXACT'),('x=1j','PY_INEXACT'),('x=1/2','PY_INEXACT'),
                           ('x=2**-1','PY_INEXACT'),('x=2**n','PY_INEXACT'),('x=float(1)','PY_CALL_GAP'),
                           ('x=complex(1)','PY_CALL_GAP'),('from math import sqrt\nx=sqrt(2)','PY_IMPORT_GAP'),
                           ('from decimal import Decimal','PY_IMPORT_GAP'),('x=eval("0.1")','PY_CALL_GAP'),
                           ('x=getattr(1,"real")','PY_CALL_GAP'),('x=(1).__class__','PY_ATTRIBUTE_GAP'),
                           ('from fractions import Fraction as F\nF=lambda x:x\nx=F(0.1)','PY_REBIND'),
                           ('def f(Fraction):\n return Fraction(1)','PY_REBIND'),
                           ('def f():\n return 1\ng=f\nx=g()','PY_CALL_GAP')]:
            with self.subTest(text=text):
                self.assertIn(code, self.python_codes(text))

    def test_unhandled_python_constructs_are_explicit_not_skipped(self):
        for text in ['class C: pass', 'async def f(): return 1', '@deco\ndef f(): return 1',
                     'try:\n x=1\nexcept:\n pass', 'x=lambda y:y', 'x=yield_value()', 'import fractions as f\nx=f.Fraction(1,2)']:
            with self.subTest(text=text):
                self.assertTrue(self.python_codes(text))
        self.assertIn('PY_SYNTAX', self.python_codes('def broken('))

    def test_callbacks_and_rebound_local_functions_cannot_smuggle_float_decisions(self):
        for text in ['x=sorted([1,2],key=float)',
                     'def f(x): return x\nf=float\nx=f(1)',
                     'from fractions import Fraction as F\ndef f(): return 1\nf=F.from_float\nx=f(1)',
                     'def f(x): return x\ndef g(f): return f(1)\nx=g(float)',
                     'from fractions import Fraction as int\nx=int(1,2)']:
            with self.subTest(text=text):
                self.assertTrue(self.python_codes(text))

    def test_python_dynamic_code_aliases_and_relative_imports_are_explicit_gaps(self):
        for text in ['loader=__import__\nloader("bad")',
                     'import importlib.util as util\nspec=util.spec_from_file_location("bad","bad.py")',
                     'from . import bad',
                     'from bad import *',
                     'from builtins import __import__ as load\nload("bad")']:
            with self.subTest(text=text):
                self.assertTrue(self.python_codes(text,False))

    def test_lean_block_comment_separator_cannot_join_identifiers_into_escape(self):
        self.assertEqual(self.lean_codes('def sor/- comment -/ry := 1'),set())
        self.assertIn('LEAN_PLACEHOLDER',self.lean_codes('theorem t : True := by /-x-/sorry'))

    def test_defined_reflective_loader_forms_fail_in_engineering_scope(self):
        cases=["getattr(__builtins__, '__dict__', __builtins__)['__import__']('Q.bad')",
               "vars()['__builtins__']['__import__']('Q.bad')",
               "f.__globals__['__builtins__']['__import__']('Q.bad')",
               "().__class__.__base__.__subclasses__()"]
        for source in cases:
            with self.subTest(source=source):
                self.assertIn('PY_DYNAMIC_IMPORT',self.python_codes(source,False))

    def test_import_module_primes_and_identifier_suffixes_are_extracted_completely(self):
        result=source_checks.check_lean("import Good Q.Bad' Q.Other! Q.Last?\n")
        self.assertEqual(result['imports'],['Good',"Q.Bad'",'Q.Other!','Q.Last?'])
        self.assertEqual(result['findings'],[])

    def test_unsupported_import_tail_is_a_gap_after_a_valid_prefix(self):
        self.assertIn('LEAN_IMPORT_GAP',self.lean_codes('import Good ; Q.Bad\n'))
        self.assertIn('LEAN_IMPORT_GAP',self.lean_codes('import Good Q.«Bad»\n'))

    def test_literal_import_operands_are_explicit_gaps_after_valid_prefix(self):
        for operand in ['"Q.Bad"', 'r"Q.Bad"', "'q'"]:
            with self.subTest(operand=operand):
                self.assertIn('LEAN_IMPORT_GAP', self.lean_codes('import Good ' + operand + '\n'))

    def test_engineering_float_is_scanned_but_not_mislabeled_mathematical(self):
        self.assertEqual(self.python_codes('duration=0.25\n', False), set())
        self.assertIn('PY_DYNAMIC_IMPORT', self.python_codes('__import__("Q.bad")', False))
        self.assertIn('PY_DYNAMIC_IMPORT', self.python_codes('import importlib as il\nil.import_module("Q.bad")', False))
        self.assertIn('PY_DYNAMIC_IMPORT', self.python_codes('from importlib import import_module as load\nload("Q.bad")', False))


class ManifestRules(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(auditor, 'auditor implementation is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.m = {'schema_version':1, 'approved_axioms':['propext','Classical.choice','Quot.sound'],
                  'roots':[{'id':'r','path':str(self.root),'tree_id':'b'*40,
                            'tracked_files':[], 'files':{}}], 'module_map':{}}

    def add(self, path, text, kind='documentation', **options):
        data = text if isinstance(text,bytes) else text.encode()
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        self.m['roots'][0]['tracked_files'].append(path)
        entry = {'sha256':sha(data),'kind':kind, **options}
        self.m['roots'][0]['files'][path] = entry
        if 'module' in entry:
            self.m['module_map'][entry['module']] = 'r:' + path
        return entry

    def bind(self, path):
        return {'root':'r','path':path,'sha256':self.m['roots'][0]['files'][path]['sha256']}

    def runaudit(self):
        return auditor.audit(self.m)

    def codes(self):
        return {f['code'] for f in self.runaudit()['findings']}

    def cleanlean(self, path='Closed.lean', text='theorem t : True := by trivial\n', **opts):
        return self.add(path,text,'project_lean',module='lean:'+path[:-5].replace('/','.'),
                        vacuity={'t@1':{'status':'conditional','premises':[],'reason':'source statement only; G2/G4 required'}},**opts)

    def qlean(self):
        return self.add('Q/Bad.lean','theorem t : False := by sorry\n','quarantined_source',language='lean',
                        module='lean:Q.Bad', expected_findings={'LEAN_PLACEHOLDER':1},
                        quarantine={'id':'Q-1','boundary':'Q/Bad.lean is isolated; no closed imports',
                                    'blocking_reason':'missing proof','owner':'none','maintenance':'zero'})

    def test_reported_hazards_fail_through_actual_manifest_auditor_interface(self):
        cases=[('def x := Environment.addDeclWithoutChecking','LEAN_KERNEL_BYPASS'),
               ('def x := Environment.«addDeclWithoutChecking»','LEAN_KERNEL_BYPASS'),
               ('def x := «sorryAx»','LEAN_PLACEHOLDER'),
               ('def x := Lean.«ofReduceBool»','LEAN_NATIVE'),
               ('def x := by decide +native','LEAN_NATIVE'),
               ('def x := by decide (config := { native /- gap -/ := true })','LEAN_NATIVE'),
               ('def x := by decide (config := { native\n := /- nested /- c -/ -/\n true })','LEAN_NATIVE'),
               ("def c := 'a",'LEAN_LEXICAL_GAP'),
               ('/- unterminated','LEAN_LEXICAL_GAP'),
               ('def s := "unterminated','LEAN_LEXICAL_GAP'),
               ('def s := r#"unterminated','LEAN_LEXICAL_GAP'),
               ('def «unterminated','LEAN_LEXICAL_GAP')]
        for source,code in cases:
            with self.subTest(source=source):
                self.m['roots'][0]['tracked_files'].clear();self.m['roots'][0]['files'].clear();self.m['module_map'].clear()
                self.add('Hazard.lean',source,'project_lean',module='lean:Hazard',vacuity={})
                result=self.runaudit()
                self.assertEqual(result['result'],'FAIL',result)
                self.assertIn(code,{f['code'] for f in result['findings']})

    def test_numeral_adjoining_owned_axiom_cannot_receive_static_pass(self):
        # Canonical698's already compiled counterexample; old numeric lexing hides axiom.
        source = 'def harmless := 0axiom hiddenPremise : False\ntheorem bad : False := hiddenPremise\n#print axioms bad\n'
        self.add('Hazard.lean', source, 'project_lean', module='lean:Hazard',
                 vacuity={'bad@2':{'status':'conditional','premises':[],
                                  'reason':'source statement only; not a witness claim'}})
        result = self.runaudit()
        self.assertIn('LEAN_AXIOM', {f['code'] for f in result['findings']}, result)
        self.assertEqual(result['result'], 'FAIL', result)

    def test_nested_quoted_throwerror_placeholder_cannot_receive_static_pass(self):
        # Lean is a benign mapped import fixture; actual compilation evidence is reused.
        self.add('Lean.lean', 'def imported := 1\n', 'project_lean', module='lean:Lean', vacuity={})
        source = 'import Lean\nopen Lean\n\ndef f : CoreM Unit := throwError "bad {let x := "safe"; (sorry : Nat)}"\n#print axioms f\n'
        self.add('Hazard.lean', source, 'project_lean', module='lean:Hazard', vacuity={})
        result = self.runaudit()
        self.assertIn('LEAN_PLACEHOLDER', {f['code'] for f in result['findings']}, result)
        self.assertEqual(result['result'], 'FAIL', result)

    def test_bound_engineering_review_cannot_waive_defined_reflective_loader(self):
        self.add('Q/bad.py','x=0.5','quarantined_source',language='python',module='python:Q.bad',
            expected_findings={'PY_INEXACT':1},quarantine={'id':'Q-r','boundary':'Q/bad.py','blocking_reason':'float','owner':'none','maintenance':'zero'})
        e=self.add('runner.py',"getattr(__builtins__, '__dict__', __builtins__)['__import__']('Q.bad')",'engineering_python',module='python:runner')
        self.add('review.json',json.dumps({'type':'nonmathematical_review','source_sha256':e['sha256'],
            'scope':'nonmathematical_execution_only','reviewer':'fixture reviewer','rationale':'A nonmath review is not an import-isolation waiver'}))
        e['nonmath_review']=self.bind('review.json')
        r=self.runaudit()
        self.assertEqual(r['result'],'FAIL',r)
        self.assertIn('PY_DYNAMIC_IMPORT',{f['code'] for f in r['findings']})

    def test_known_unsupported_source_suffix_cannot_claim_a_supported_language(self):
        for path,language,text in [('bad.sh','lean','echo hello\n'),('bad.js','python','x=1\n')]:
            with self.subTest(path=path):
                self.add(path,text,'external_pinned',language=language,module=language+':Bad'+language,
                    **({'vacuity':{}} if language=='lean' else {}),pin={'revision':'a'*40,'origin':'fixture'})
                self.assertIn('LANGUAGE_GAP',self.codes())

    def test_retained_text_source_requires_language_then_a_registered_module(self):
        e=self.add('Q.txt','theorem t : False := by sorry\n','quarantined_source',expected_findings={},
            quarantine={'id':'Q-t','boundary':'Q.txt','blocking_reason':'unproved','owner':'none','maintenance':'zero'})
        self.assertIn('LANGUAGE_GAP',self.codes())
        e['language']='lean';e['expected_findings']={'LEAN_PLACEHOLDER':1}
        self.assertIn('MODULE_ALIAS',self.codes())
        e['module']='lean:Q';self.m['module_map']['lean:Q']='r:Q.txt'
        self.assertEqual(self.runaudit()['result'],'STATIC_PASS',self.runaudit())

    def test_mapped_external_text_module_cannot_skip_supported_source_analysis(self):
        self.add('fake.txt','Fraction=float','external_pinned',module='python:fractions',
            pin={'revision':'a'*40,'origin':'fixture'})
        self.add('closed.py','from fractions import Fraction\nx=Fraction(1,2)','math_python',module='python:closed')
        self.assertIn('LANGUAGE_GAP',self.codes())

    def test_primed_lean_module_after_valid_import_cannot_hide_quarantine(self):
        self.add('Good.lean','def x := 1\n','project_lean',module='lean:Good',vacuity={})
        self.add("Q/Bad'.lean",'theorem t : False := by sorry\n','quarantined_source',language='lean',module="lean:Q.Bad'",
            expected_findings={'LEAN_PLACEHOLDER':1},quarantine={'id':'Q-p','boundary':"Q/Bad'.lean",'blocking_reason':'unproved','owner':'none','maintenance':'zero'})
        self.add('Closed.lean',"import Good Q.Bad'\n",'project_lean',module='lean:Closed',vacuity={})
        self.assertIn('QUARANTINE_IMPORT',self.codes())

    def test_status_observation_binding_distinguishes_json_booleans_and_integers(self):
        self.add('input.txt','input')
        observed={'schema_version':True,'gate':'G5','observed_status':'PASS','run_id':'run-typed',
            'command':['fixture'],'exit_code':False,'inputs':{'r:input.txt':sha(b'input')}}
        self.add('observation.json',json.dumps(observed),'data')
        status={**observed,'schema_version':1,'exit_code':0,'evidence':self.bind('observation.json')}
        self.add('status.json',json.dumps(status),'data',status_evidence=True)
        self.assertIn('STATUS_EVIDENCE',self.codes())

    def test_complete_manifest_counts_every_file_and_is_deterministic(self):
        self.cleanlean()
        self.add('README.md','documentation')
        r = self.runaudit()
        self.assertEqual(r['result'],'STATIC_PASS',r)
        self.assertEqual(r['classified_count'],2)
        self.assertEqual(r['scanned_count'],2)
        self.assertEqual(r,self.runaudit())
        self.assertIn('G2', [x['gate'] for x in r['remaining_obligations']])
        self.assertIn('G4', [x['gate'] for x in r['remaining_obligations']])

    def test_each_file_is_really_scanned_not_only_first(self):
        self.cleanlean('A.lean')
        self.cleanlean('B.lean','theorem t : True := by sorry\n')
        self.assertIn('LEAN_PLACEHOLDER', self.codes())

    def test_unclassified_unknown_skipped_and_extra_paths_fail_closed(self):
        self.add('a.txt','a')
        self.m['roots'][0]['tracked_files'].append('missing.txt')
        self.assertIn('SCOPE_CLASSIFICATION',self.codes())
        self.m['roots'][0]['tracked_files'].pop()
        self.m['roots'][0]['files']['a.txt']['kind']='ignore'
        self.assertIn('CLASSIFICATION',self.codes())
        self.m['roots'][0]['files']['a.txt']['kind']='documentation'
        self.m['roots'][0]['files']['extra.txt']={'kind':'data','sha256':'0'*64}
        self.assertIn('SCOPE_CLASSIFICATION',self.codes())

    def test_duplicate_scope_and_unknown_schema_keys_are_rejected(self):
        self.add('a.txt','a')
        self.m['roots'][0]['tracked_files'].append('a.txt')
        self.assertIn('MANIFEST',self.codes())
        self.m['roots'][0]['tracked_files'].pop()
        self.m['skip_everything']=True
        self.assertIn('MANIFEST',self.codes())

    def test_empty_scope_cannot_manufacture_green(self):
        self.assertIn('MANIFEST',self.codes())

    def test_missing_unreadable_invalid_utf8_files_are_not_skipped(self):
        self.add('a.txt','a')
        (self.root/'a.txt').unlink()
        self.assertIn('FILE_MISSING',self.codes())
        (self.root/'a.txt').write_text('a')
        (self.root/'a.txt').chmod(0)
        self.assertIn('FILE_UNREADABLE',self.codes())
        (self.root/'a.txt').chmod(0o600)
        e=self.add('Bad.lean',b'\xff','project_lean',module='lean:Bad',vacuity={})
        self.assertIn('FILE_ENCODING',self.codes())

    def test_zero_mode_file_is_rejected_even_when_existing_descriptor_can_read(self):
        # The policy guard matters under privileged readers, not just ordinary PermissionError.
        from unittest.mock import patch
        self.add('mode.txt','a')
        target=self.root/'mode.txt'
        descriptor=os.open(target,os.O_RDONLY)
        self.addCleanup(os.close,descriptor)
        target.chmod(0)
        self.addCleanup(target.chmod,0o600)
        self.assertEqual(os.pread(descriptor,1,0),b'a')
        original=Path.read_bytes
        def privileged_read(path):
            return os.pread(descriptor,1,0) if path==target else original(path)
        with patch.object(Path,'read_bytes',privileged_read):
            self.assertIn('FILE_UNREADABLE',self.codes())

    def test_pin_hash_and_foundation_allowlist_cannot_be_weakened(self):
        self.cleanlean()
        (self.root/'Closed.lean').write_text('theorem t : False := by sorry\n')
        self.assertIn('FILE_HASH',self.codes())
        self.m['approved_axioms'].append('sorryAx')
        self.assertIn('FOUNDATION_PROFILE',self.codes())

    def test_foundation_names_cannot_authorize_new_owned_axioms(self):
        for name in ['propext','Classical.choice','Quot.sound']:
            with self.subTest(name=name):
                self.m['roots'][0]['files'].clear();self.m['roots'][0]['tracked_files'].clear();self.m['module_map'].clear()
                self.add('New.lean','axiom '+name+' : False\n','project_lean',module='lean:New',vacuity={})
                self.assertIn('LEAN_AXIOM',self.codes())

    def test_absolute_parent_normalization_and_symlink_alias_paths_are_rejected(self):
        self.add('a.txt','a')
        for bad in ['../a.txt','/a.txt','./a.txt','a//b','a\\b','é/../a','e\u0301.txt']:
            with self.subTest(path=bad):
                m=copy.deepcopy(self.m)
                m['roots'][0]['tracked_files']=[bad]
                m['roots'][0]['files']={bad:{'kind':'data','sha256':'0'*64}}
                self.assertIn('PATH', {x['code'] for x in auditor.audit(m)['findings']})
        (self.root/'alias.txt').symlink_to(self.root/'a.txt')
        self.m['roots'][0]['tracked_files'].append('alias.txt')
        self.m['roots'][0]['files']['alias.txt']={'kind':'data','sha256':sha(b'a')}
        self.assertIn('PATH_SYMLINK',self.codes())

    def test_source_relabeling_to_docs_or_binary_is_not_silent(self):
        self.add('Bad.lean','theorem t : False := by sorry','documentation')
        self.assertIn('SOURCE_CLASSIFICATION',self.codes())
        self.m['roots'][0]['files']['Bad.lean']['kind']='data'
        self.assertIn('SOURCE_CLASSIFICATION',self.codes())

    def test_engineering_requires_explicit_bound_nonmathematical_review(self):
        e=self.add('timer.py','duration=0.1\n','engineering_python',module='python:timer')
        self.assertIn('NONMATH_REVIEW',self.codes())
        self.add('review.json',json.dumps({'type':'nonmathematical_review','source_sha256':e['sha256'],
                    'scope':'nonmathematical_execution_only','reviewer':'fixture reviewer','rationale':'timing only'}))
        e['nonmath_review']=self.bind('review.json')
        self.assertEqual(self.runaudit()['result'],'STATIC_PASS',self.runaudit())
        review=json.loads((self.root/'review.json').read_text());review['source_sha256']='0'*64
        (self.root/'review.json').write_text(json.dumps(review))
        self.m['roots'][0]['files']['review.json']['sha256']=sha((self.root/'review.json').read_bytes())
        e['nonmath_review']=self.bind('review.json')
        self.assertIn('NONMATH_REVIEW',self.codes())

    def test_negative_fixture_retains_exact_violation_and_never_counts_as_clean(self):
        self.add('Negative.lean','example : False := by sorry\n','negative_fixture',language='lean',
                 module='lean:Negative',expected_findings={'LEAN_PLACEHOLDER':1},fixture_purpose='retain failing proof test')
        r=self.runaudit()
        self.assertEqual(r['result'],'STATIC_PASS',r)
        self.assertEqual(r['retained_findings'][0]['code'],'LEAN_PLACEHOLDER')
        self.assertEqual(r['retained_count'],1)
        e=self.m['roots'][0]['files']['Negative.lean'];e['expected_findings']={}
        self.assertIn('RETAINED_FINDING_MISMATCH',self.codes())

    def test_quarantine_requires_exact_boundary_and_scans_retained_source(self):
        e=self.qlean()
        self.assertEqual(self.runaudit()['result'],'STATIC_PASS',self.runaudit())
        e['quarantine']['boundary']=''
        self.assertIn('QUARANTINE',self.codes())
        e['quarantine']['boundary']='exact boundary'
        e['expected_findings']={'LEAN_PLACEHOLDER':2}
        self.assertIn('RETAINED_FINDING_MISMATCH',self.codes())

    def test_closed_import_cannot_reach_quarantine_directly_or_transitively(self):
        self.qlean()
        self.cleanlean('Mid.lean','import Q.Bad\ntheorem t : True := by trivial\n')['vacuity']={'t@2':{'status':'conditional','premises':[],'reason':'fixture'}}
        self.cleanlean('Top.lean','import Mid\ntheorem t : True := by trivial\n')['vacuity']={'t@2':{'status':'conditional','premises':[],'reason':'fixture'}}
        r=self.runaudit()
        hits=[x for x in r['findings'] if x['code']=='QUARANTINE_IMPORT']
        self.assertEqual({x['path'] for x in hits},{'Mid.lean','Top.lean'})

    def test_unknown_module_alias_and_dynamic_import_fail_closed(self):
        self.cleanlean(text='import Unknown.Module\ntheorem t : True := by trivial\n')['vacuity']={'t@2':{'status':'conditional','premises':[],'reason':'fixture'}}
        self.assertIn('IMPORT_UNRESOLVED',self.codes())
        self.m['module_map']['lean:Unknown.Module']='r:Closed.lean'
        self.assertIn('MODULE_ALIAS',self.codes())
        self.m['module_map'].pop('lean:Unknown.Module')
        self.add('bad.py','__import__("Q.Bad")','math_python',module='python:bad')
        self.assertIn('PY_DYNAMIC_IMPORT',self.codes())

    def test_python_import_alias_cannot_hide_quarantine(self):
        self.add('bad.py','x=0.5','quarantined_source',language='python',module='python:bad',
                 expected_findings={'PY_INEXACT':1},quarantine={'id':'Q-2','boundary':'bad.py','blocking_reason':'float', 'owner':'none','maintenance':'zero'})
        self.add('closed.py','import bad as good\nassert True','math_python',module='python:closed')
        self.assertIn('QUARANTINE_IMPORT',self.codes())

    def test_fraction_stdlib_exception_cannot_hide_a_mapped_local_quarantine(self):
        self.add('fractions.py','x=0.5','quarantined_source',language='python',module='python:fractions',
            expected_findings={'PY_INEXACT':1},quarantine={'id':'Q-f','boundary':'fractions.py','blocking_reason':'fake fractions','owner':'none','maintenance':'zero'})
        self.add('closed.py','from fractions import Fraction\nx=Fraction(1,2)','math_python',module='python:closed')
        self.assertIn('QUARANTINE_IMPORT',self.codes())

    def test_engineering_from_import_alias_checks_a_mapped_submodule_boundary(self):
        self.add('Q/__init__.py','x=1','math_python',module='python:Q')
        self.add('Q/bad.py','x=0.5','quarantined_source',language='python',module='python:Q.bad',
            expected_findings={'PY_INEXACT':1},quarantine={'id':'Q-b','boundary':'Q/bad.py','blocking_reason':'float','owner':'none','maintenance':'zero'})
        e=self.add('runner.py','from Q import bad as harmless','engineering_python',module='python:runner')
        self.add('review.json',json.dumps({'type':'nonmathematical_review','source_sha256':e['sha256'],
            'scope':'nonmathematical_execution_only','reviewer':'fixture reviewer','rationale':'nonmath import runner'}))
        e['nonmath_review']=self.bind('review.json')
        self.assertIn('QUARANTINE_IMPORT',self.codes())

    def test_external_source_needs_immutable_pin_and_is_still_scanned(self):
        e=self.add('Dep.lean','axiom hidden : False','external_pinned',language='lean',module='lean:Dep')
        self.assertIn('EXTERNAL_PIN',self.codes())
        e['pin']={'revision':'a'*40,'origin':'https://example.invalid/dependency'}
        self.assertIn('LEAN_AXIOM',self.codes())

    def test_binary_proof_requires_bound_g2_g4_evidence(self):
        e=self.add('Closed.olean',b'\x00\xffproof','binary_proof')
        self.assertIn('BINARY_BINDING',self.codes())
        bindings={}
        for gate in ['G2','G4']:
            self.add(gate+'.json',json.dumps({'type':'kernel_evidence','gate':gate,'subject_sha256':e['sha256'],
                'observed_status':'PASS','exit_code':0,'command':['fixture','kernel-check'],'run_id':'fixture-run'}))
            bindings[gate]=self.bind(gate+'.json')
        e['bindings']=bindings
        self.assertEqual(self.runaudit()['result'],'STATIC_PASS',self.runaudit())
        e['bindings']['G4']['sha256']='0'*64
        self.assertIn('EVIDENCE_BINDING',self.codes())

    def test_vacuity_missing_stale_and_false_nonvacuity_annotations_are_rejected(self):
        e=self.cleanlean(text='theorem t (h : False) : True := by trivial\n')
        self.assertIn('VACUITY_CONTRADICTION',self.codes())
        e['vacuity']={}
        self.assertIn('VACUITY_COVERAGE',self.codes())
        e['vacuity']={'t@1':{'status':'vacuous','reason':'False hypothesis'}}
        self.assertEqual(self.runaudit()['result'],'STATIC_PASS',self.runaudit())
        e['vacuity']['old@1']={'status':'vacuous','reason':'stale'}
        self.assertIn('VACUITY_COVERAGE',self.codes())

    def test_nonvacuous_claim_requires_content_bound_certificate(self):
        e=self.cleanlean()
        e['vacuity']['t@1']={'status':'nonvacuous','reason':'explicit witness'}
        self.assertIn('VACUITY_EVIDENCE',self.codes())

    def test_status_pass_requires_actual_bound_observation_not_aspiration(self):
        self.add('input.txt','bound input')
        status={'schema_version':1,'gate':'G5','observed_status':'PASS','run_id':'run-1',
            'command':['python','fixture.py'],'exit_code':0,'inputs':{'r:input.txt':sha(b'bound input')}}
        e=self.add('status.json',json.dumps(status),'data',status_evidence=True)
        self.assertIn('STATUS_EVIDENCE',self.codes())
        self.add('observation.json',json.dumps(status),'data')
        status['evidence']=self.bind('observation.json')
        raw=json.dumps(status).encode();(self.root/'status.json').write_bytes(raw);e['sha256']=sha(raw)
        self.assertEqual(self.runaudit()['result'],'STATIC_PASS',self.runaudit())
        status['exit_code']=1
        raw=json.dumps(status).encode();(self.root/'status.json').write_bytes(raw);e['sha256']=sha(raw)
        self.assertIn('STATUS_EVIDENCE',self.codes())

    def test_malformed_manifest_values_fail_with_findings_instead_of_crashing(self):
        self.add('a.txt','a')
        samples=[]
        for key,value in [('kind',[]),('language',[]),('module',[]),('sha256',None)]:
            m=copy.deepcopy(self.m);m['roots'][0]['files']['a.txt'][key]=value;samples.append(m)
        for key,value in [('roots',{}),('module_map',[]),('approved_axioms',[[],[],[]]),('schema_version',True)]:
            m=copy.deepcopy(self.m);m[key]=value;samples.append(m)
        for m in samples:
            with self.subTest(manifest=m):
                r=auditor.audit(m)
                self.assertEqual(r['result'],'FAIL',r)
                self.assertTrue(r['findings'])

    def test_archived_source_suffixes_and_wrong_explicit_language_cannot_hide_code(self):
        self.add('Old.lean.txt','theorem t : False := by sorry','documentation')
        self.assertIn('SOURCE_CLASSIFICATION',self.codes())
        self.m['roots'][0]['files'].clear();self.m['roots'][0]['tracked_files'].clear()
        self.add('Wrong.lean','theorem t : False := by sorry','quarantined_source',language='python',
            module='python:Wrong',expected_findings={'PY_SYNTAX':1},
            quarantine={'id':'Q-x','boundary':'Wrong.lean','blocking_reason':'fixture','owner':'none','maintenance':'zero'})
        self.assertIn('SOURCE_CLASSIFICATION',self.codes())

    def test_modules_cannot_point_to_documentation_in_place_of_source(self):
        self.cleanlean(text='import Fake\ntheorem t : True := by trivial\n')['vacuity']={'t@2':{'status':'conditional','premises':[],'reason':'fixture'}}
        self.add('fake.txt','unscanned','documentation',module='lean:Fake')
        self.assertIn('MODULE_ALIAS',self.codes())

    def test_classification_fields_are_not_silently_ignored_and_tree_pin_is_immutable(self):
        self.add('a.txt','a',vacuity={})
        self.assertIn('CLASSIFICATION',self.codes())
        self.m['roots'][0]['files']['a.txt'].pop('vacuity')
        self.m['roots'][0]['tree_id']='main'
        self.assertIn('MANIFEST',self.codes())

    def test_external_binary_proof_cannot_bypass_kernel_bindings_or_utf8_mode(self):
        self.add('Dep.olean',b'\x00\xffproof','external_pinned',pin={'revision':'a'*40,'origin':'fixture origin'})
        self.assertIn('BINARY_BINDING',self.codes())
        self.assertNotIn('FILE_ENCODING',self.codes())

    def test_file_hash_failure_does_not_prevent_violation_detection(self):
        self.cleanlean()
        (self.root/'Closed.lean').write_text('theorem t : False := by sorry\n')
        self.assertTrue({'FILE_HASH','LEAN_PLACEHOLDER'}<=self.codes())

    def test_nonvacuity_certificate_and_not_run_status_have_positive_binding_cases(self):
        e=self.cleanlean()
        self.add('witness.json',json.dumps({'type':'nonvacuity_certificate','subject_sha256':e['sha256'],
            'declaration':'t@1','witness_description':'No hypotheses; True witness trivial'}))
        e['vacuity']['t@1']={'status':'nonvacuous','reason':'no hypotheses','evidence':self.bind('witness.json')}
        self.add('status.json',json.dumps({'schema_version':1,'gate':'G4','observed_status':'NOT_RUN',
            'run_id':'fixture-not-run','command':['fixture','kernel-check'],'exit_code':None,
            'inputs':{'r:Closed.lean':e['sha256']}}),'data',status_evidence=True)
        self.assertEqual(self.runaudit()['result'],'STATIC_PASS',self.runaudit())

    def test_malformed_vacuity_and_status_types_are_explicit_findings(self):
        e=self.cleanlean();e['vacuity']['t@1']['status']=[]
        self.assertIn('VACUITY_ANNOTATION',self.codes())
        e['vacuity']['t@1']['status']='conditional'
        self.add('bad-status.json',json.dumps({'schema_version':1,'gate':[], 'observed_status':[],
            'run_id':'fixture','command':['fixture'],'exit_code':0,'inputs':{'r:Closed.lean':e['sha256']}}),
            'data',status_evidence=True)
        self.assertIn('STATUS_EVIDENCE',self.codes())

    def test_exact_math_cannot_import_a_nonmathematical_fractions_shadow(self):
        e=self.add('fractions.py','Fraction=float','engineering_python',module='python:fractions')
        self.add('review.json',json.dumps({'type':'nonmathematical_review','source_sha256':e['sha256'],
            'scope':'nonmathematical_execution_only','reviewer':'fixture reviewer','rationale':'nonmath fixture'}))
        e['nonmath_review']=self.bind('review.json')
        self.add('closed.py','from fractions import Fraction\nx=Fraction(1,2)','math_python',module='python:closed')
        self.assertIn('MATH_DEPENDENCY_GAP',self.codes())

    def test_binary_source_language_mismatch_is_a_finding_not_a_crash(self):
        self.add('Bad.olean',b'\x00\xffproof','external_pinned',language='lean',module='lean:Bad',
            pin={'revision':'a'*40,'origin':'fixture origin'})
        self.assertIn('SOURCE_CLASSIFICATION',self.codes())

    def test_cli_output_write_failure_has_distinct_nonzero_code(self):
        self.add('readme.txt','ok')
        manifest=self.root.parent/(self.root.name+'-manifest.json')
        self.addCleanup(lambda:manifest.unlink(missing_ok=True))
        manifest.write_text(json.dumps(self.m))
        p=subprocess.run([sys.executable,'-B',str(HERE/'auditor.py'),'audit','--manifest',str(manifest),
            '--output',str(self.root/'absent'/'out.json')],capture_output=True,text=True)
        self.assertEqual(p.returncode,3,p.stderr+p.stdout)
        self.assertEqual(json.loads(p.stdout)['result'],'OUTPUT_ERROR')

    def test_cli_json_and_nonzero_failures_and_duplicate_json_keys(self):
        self.add('readme.txt','ok')
        manifest=self.root.parent/(self.root.name+'-manifest.json')
        self.addCleanup(lambda: manifest.unlink(missing_ok=True))
        manifest.write_text(json.dumps(self.m))
        cmd=[sys.executable,'-B',str(HERE/'auditor.py'),'audit','--manifest',str(manifest)]
        p=subprocess.run(cmd,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr+p.stdout)
        self.assertEqual(json.loads(p.stdout)['result'],'STATIC_PASS')
        (self.root/'readme.txt').unlink()
        p=subprocess.run(cmd,capture_output=True,text=True)
        self.assertEqual(p.returncode,1,p.stderr+p.stdout)
        self.assertIn('FILE_MISSING',{x['code'] for x in json.loads(p.stdout)['findings']})
        manifest.write_text('{"schema_version":1,"schema_version":1}')
        p=subprocess.run(cmd,capture_output=True,text=True)
        self.assertEqual(p.returncode,2,p.stderr+p.stdout)
        self.assertEqual(json.loads(p.stdout)['result'],'INPUT_ERROR')
        manifest.write_text('{"n":0.1}')
        p=subprocess.run(cmd,capture_output=True,text=True)
        self.assertEqual(p.returncode,2)


if __name__=='__main__':
    unittest.main()
