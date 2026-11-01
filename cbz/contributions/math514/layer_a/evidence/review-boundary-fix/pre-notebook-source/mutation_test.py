#!/usr/bin/env python3
"""G0: real isolated checker-source mutations; 100% declared mutants must be killed."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import functools
import io
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent

# Each change weakens a claimed auditor rule, rather than planting a bad proof fixture.
MUTATIONS=[
 ('restore-augmented-division-gap', 'source_checks.py', 'if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Div):', 'if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):'),
 ('restore-augmented-power-gap', 'source_checks.py', 'if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Pow):', 'if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):'),
 ('restore-python-candidate-nonmath-gap', 'auditor.py', "if kind=='math_python' and records[candidate_target]['entry']['kind']=='engineering_python':", 'if False:'),
 ('restore-python-prefix-nonmath-gap', 'auditor.py', "if kind=='math_python' and records[prefix]['entry']['kind']=='engineering_python':", 'if False:'),
 ('restore-null-status-evidence-skip', 'auditor.py', "if 'status_evidence' in entry:", "if entry.get('status_evidence') is not None:"),
 ('restore-es-module-extension-gap', 'auditor.py', "'.js','.mjs','.ts'", "'.js','.ts'"),
 ('restore-es-module-text-wrapper-gap', 'auditor.py', "'.sh.txt','.js.txt','.mjs.txt'", "'.sh.txt','.js.txt'"),
 ('restore-metadata-generic-numeral-swallowing', 'source_checks.py', "        elif c.isdigit():\n            # Preserve the canonical numeral boundary: valid Lean permits a\n            # numeral immediately before a declaration keyword (0theorem).\n            # Keep spelling/location for literal-hypothesis checks; no numeric\n            # evaluation or independent numeral grammar is introduced here.\n            match = canonical_numeral_spelling.match(text, i)\n            end = match.end() if match is not None else i + 1\n            tokens.append(Token(text[i:end], line, column)); advance(end)\n        elif c.isalpha() or c == '_':", "        elif c.isalpha() or c == '_' or c.isdigit():"),
 ('discard-metadata-numeral-tokens', 'source_checks.py', 'tokens.append(Token(text[i:end], line, column)); advance(end)', 'advance(end)'),
 ('overshoot-metadata-numeral-boundary', 'source_checks.py', 'tokens.append(Token(text[i:end], line, column)); advance(end)', 'tokens.append(Token(text[i:end], line, column)); advance(min(end + 1, n))'),
 ('restore-generic-numeral-swallowing','canonical_lean.py',"match = _NUMERAL.match(text, i)\n            advance(match.end() - i if match else 1)","while i < n and (text[i].isalnum() or text[i] in '._'):\n                advance(1)"),
 ('discard-nested-string-code-holes','canonical_lean.py',"yield from _brace_segments(text, content_start, i - 1, start[0], start[1] + 1)","pass"),
 ('discard-canonical-lexical-errors','source_checks.py',"for line, column, kind in dict.fromkeys(lexical_errors):","for line, column, kind in []:"),
 ('ignore-canonical-metadata-shape','source_checks.py',"findings.append(finding('LEAN_LEXICAL_GAP', 'unsupported canonical lexer metadata shape'))","pass"),
 ('empty-lean-findings','source_checks.py',"return {'findings': findings, 'imports': imports, 'declarations': declarations}","return {'findings': [], 'imports': imports, 'declarations': declarations}"),
 ('canonical-unicode-boundary-weakened','canonical_lean.py',"or _is_letter(ch) or _is_subscript(ch)","or ch == 'λ' or _is_letter(ch) or _is_subscript(ch)"),
 ('disable-placeholder-category','source_checks.py',"placeholders = {'sorry', 'admit', 'sorryAx', 'admitGoal', 'mkSorry'}","placeholders = set()"),
 ('disable-axiom-category','source_checks.py',"if tail in {'axiom', 'constant', 'constants'}:","if False:"),
 ('disable-native-category','source_checks.py',"if tail in native:","if False:"),
 ('disable-unchecked-declaration-refusal','source_checks.py',"if tail in kernel_bypass:","if False:"),
 ('skip-malformed-character-refusal','source_checks.py',"return {'findings': findings, 'imports': imports, 'declarations': declarations}","return {'findings': [f for f in findings if not (f['code'] == 'LEAN_LEXICAL_GAP' and ('character literal' in f['detail'] or 'unterminated_char' in f['detail']))], 'imports': imports, 'declarations': declarations}"),
 ('miss-escaped-placeholder-helpers','source_checks.py',"not token.quoted or tail in {'sorryAx','admitGoal','mkSorry'}","not token.quoted or tail == 'sorryAx'"),
 ('drop-escaped-constant-normalization','source_checks.py',"name = '.'.join(part.removeprefix('\\x00').replace('\\x01', '.') for part in parts)","name = raw_name"),
 ('miss-multiline-commented-native-config','source_checks.py',"if tail in native:","if tail in native and not (tail == 'native' and ('\\n' in text or '/-' in text)):"),
 ('disable-macro-gap-category','source_checks.py',"if tail in expansion:","if False:"),
 ('drop-lean-dependencies','source_checks.py',"'imports': imports, 'declarations': declarations}","'imports': [], 'declarations': declarations}"),
 ('disable-float-literal-category','source_checks.py',"if isinstance(node, ast.Constant) and isinstance(node.value, (float, complex)):","if False:"),
 ('allow-python-division','source_checks.py',"if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Div):","if False:"),
 ('allow-negative-dynamic-powers','source_checks.py',"if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Pow):","if False:"),
 ('allow-unknown-python-calls','source_checks.py',"if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls:","if False:"),
 ('disable-exact-python-profile','source_checks.py',"if not mathematical:\n            continue","if True:\n            continue"),
 ('allow-reflective-attributes','source_checks.py',"if isinstance(node, ast.Attribute) and node.attr not in {'numerator','denominator'}:","if False:"),
 ('allow-protected-name-rebinding','source_checks.py',"if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in protected:","if False:"),
 ('restore-reflective-builtin-loader-gap','source_checks.py',"'__builtins__', 'vars', 'getattr', 'setattr', 'delattr'","'unused_reflective_name'"),
 ('disable-reflective-loader-attributes','source_checks.py',"if isinstance(node, ast.Attribute) and node.attr in {'__builtins__','__globals__','__dict__','__class__','__base__','__bases__','__mro__','__subclasses__','__getattribute__','__getattr__','__code__','__closure__','func_globals','gi_frame','cr_frame','f_globals','f_builtins'}:","if False:"),
 ('truncate-primed-lean-imports','source_checks.py','r"[^\\W\\d][\\w\'!?]*(?:\\.[^\\W\\d][\\w\'!?]*)*"',"r'[^\\W\\d]\\w*(?:\\.[^\\W\\d]\\w*)*'"),
 ('silence-unsupported-lean-import-tail','source_checks.py',"findings.append(finding('LEAN_IMPORT_GAP', 'unsupported import operand or trailing syntax: ' + t.text, t.line))","pass"),
 ('drop-literal-import-operands','source_checks.py',"tokens.append(Token('<literal>', line, column))","pass"),
 ('skip-all-but-first-file','auditor.py',"for path in sorted(set(tracked)|set(files)):","for path in sorted(set(tracked)|set(files))[:1]:"),
 ('allow-incomplete-classification','auditor.py',"if set(tracked)!=set(files):","if False:"),
 ('allow-unknown-classification','auditor.py',"'external_pinned','binary_proof','documentation','data'})","'external_pinned','binary_proof','documentation','data','ignore'})"),
 ('disable-content-hash-check','auditor.py',"if digest(data)!=entry.get('sha256'):","if False:"),
 ('allow-symlink-files','auditor.py',"if any((root.joinpath(*PurePosixPath(path).parts[:i])).is_symlink() for i in range(1,len(PurePosixPath(path).parts)+1)):","if False:"),
 ('allow-unreadable-files','auditor.py',"if not info.st_mode & (stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH):","if False:"),
 ('skip-unreadable-file','auditor.py',"add('FILE_UNREADABLE','no read permissions even under privileged runner',rid,path);continue","continue"),
 ('allow-relabeled-source','auditor.py',"if kind in {'documentation','data'} and (source_like or suffix in BINARY_EXTENSIONS):","if False:"),
 ('skip-retained-text-language-requirement','auditor.py',"kind in RETAINED or kind=='external_pinned' and (source_like or 'module' in entry)","kind in RETAINED and source_like or kind=='external_pinned' and (source_like or 'module' in entry)"),
 ('skip-mapped-external-language-requirement','auditor.py',"kind=='external_pinned' and (source_like or 'module' in entry)","kind=='external_pinned' and source_like"),
 ('allow-unsupported-source-extension-language','auditor.py',"if source_like and expected_language is None:","if False:"),
 ('allow-unpinned-external-source','auditor.py',"if kind=='external_pinned' or 'pin' in entry:","if False:"),
 ('skip-nonmath-review','auditor.py',"if kind=='engineering_python':","if False:"),
 ('skip-binary-proof-bindings','auditor.py',"if kind=='binary_proof' or suffix in BINARY_EXTENSIONS:","if False:"),
 ('ignore-retained-finding-counts','auditor.py',"if actual!=entry.get('expected_findings'):","if False:"),
 ('ignore-quarantine-boundary','auditor.py',"if kind=='quarantined_source':","if False:"),
 ('restore-shadowed-fractions-exemption','auditor.py'," and 'python:fractions' not in module_map:",":"),
 ('drop-python-submodule-candidates','auditor.py',"for candidate in result.get('import_candidates',[]):","for candidate in []:"),
 ('allow-nonmath-dependency-values','auditor.py',"if kind=='math_python' and records[target]['entry']['kind']=='engineering_python':","if False:"),
 ('skip-path-validation','auditor.py',"if not normalized_path(path):","if False:"),
 ('allow-quarantine-dependency','auditor.py',"if records[target]['entry']['kind'] in RETAINED:","if False:"),
 ('skip-vacuity-coverage','auditor.py',"if not isinstance(annotations,dict) or len(keys)!=len(set(keys)) or set(annotations)!=set(keys):","if False:"),
 ('ignore-contradictory-hypotheses','auditor.py',"if declaration['contradictions'] and a['status']!='vacuous':","if False:"),
 ('allow-unbound-nonvacuity','auditor.py',"if a['status']=='nonvacuous':","if False:"),
 ('skip-status-evidence','auditor.py',"if 'status_evidence' in entry:","if False:"),
 ('conflate-status-json-booleans-and-integers','auditor.py',"json.dumps(evidence,sort_keys=True,separators=(',',':'))!=json.dumps(comparable,sort_keys=True,separators=(',',':'))","evidence!=comparable"),
 ('allow-arbitrary-foundation-set','auditor.py',"if not isinstance(approved,list) or any(not isinstance(a,str) for a in approved) or len(approved)!=3 or set(approved)!=APPROVED_AXIOMS:","if False:"),
 ('empty-final-findings','auditor.py',"'retained_count':len(retained_findings),'findings':findings,","'retained_count':len(retained_findings),'findings':[],"),
 ('always-report-static-pass','auditor.py',"'result':'STATIC_PASS' if not findings else 'FAIL'","'result':'STATIC_PASS'"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse_test_result(output, returncode):
    count=re.search(r'Ran (\d+) tests? in ',output)
    tests=int(count.group(1)) if count else 0
    failed=re.search(r'FAILED \(([^)]*)\)',output)
    fields=dict((k,int(v)) for k,v in re.findall(r'(failures|errors)=(\d+)',failed.group(1))) if failed else {}
    failures,errors=fields.get('failures',0),fields.get('errors',0)
    return tests,failures,errors


def classify_test_result(value, code, expected_ids=None):
    """Accept only a complete receipt emitted by the phase-aware runner.

    Text summaries are diagnostic only. This validates runner evidence; it is
    not an authentication scheme for JSON supplied by an untrusted process.
    """
    if type(value) is not dict or type(code) is not int or code not in (0, 1):
        return 'ERROR'
    if value.get('receipt_schema') != 'cbz-unittest-lifecycle-v1':
        return 'ERROR'
    counts=('tests_run','assertion_failures','errors','skipped','expected_failures',
            'unexpected_successes','lifecycle_failures','subtests')
    if any(type(value.get(k)) is not int or value[k] < 0 for k in counts):
        return 'ERROR'
    if type(value.get('successful')) is not bool or value.get('invalid_population') != []:
        return 'ERROR'
    lists=('selected_ids','started_ids','finished_ids','body_entered_ids',
           'body_finished_ids','body_assertion_ids')
    for key in lists:
        items=value.get(key)
        if type(items) is not list or any(type(item) is not str or not item for item in items):
            return 'ERROR'
        if key != 'body_assertion_ids' and len(items) != len(set(items)):
            return 'ERROR'
    selected=value['selected_ids']
    if expected_ids is not None:
        if (type(expected_ids) is not list or not expected_ids
                or any(type(i) is not str or not i for i in expected_ids)
                or len(expected_ids)!=len(set(expected_ids)) or selected!=expected_ids):
            return 'ERROR'
    if not selected or value['tests_run'] != len(selected):
        return 'ERROR'
    if any(value[k] != selected for k in lists[1:5]):
        return 'ERROR'
    if any(item not in selected for item in value['body_assertion_ids']):
        return 'ERROR'
    if value['assertion_failures'] != len(value['body_assertion_ids']):
        return 'ERROR'
    assertions=value.get('assertion_events')
    if type(assertions) is not list:
        return 'ERROR'
    for event in assertions:
        if (type(event) is not dict or set(event)!={'id','source'}
                or event['id'] not in selected or event['source'] not in ('direct','subtest')):
            return 'ERROR'
    if [e['id'] for e in assertions]!=value['body_assertion_ids']:
        return 'ERROR'
    direct=[e['id'] for e in assertions if e['source']=='direct']
    if len(direct)!=len(set(direct)):
        return 'ERROR'
    events=value.get('subtest_events')
    if type(events) is not list or value['subtests'] != len(events):
        return 'ERROR'
    for event in events:
        if (type(event) is not dict or set(event) != {'id','phase','outcome'}
                or event['id'] not in selected or event['phase'] != 'body'
                or event['outcome'] not in ('success','assertion')):
            return 'ERROR'
    assertion_subtests=[e['id'] for e in events if e['outcome']=='assertion']
    if assertion_subtests!=[e['id'] for e in assertions if e['source']=='subtest']:
        return 'ERROR'
    if any(value[k] for k in ('errors','skipped','expected_failures',
                              'unexpected_successes','lifecycle_failures')):
        return 'ERROR'
    if code == 1 and value['assertion_failures'] > 0 and value['successful'] is False:
        return 'KILLED'
    if code == 0 and value['assertion_failures'] == 0 and value['successful'] is True:
        return 'SURVIVED'
    return 'ERROR'


def ordinary_sync_method(method):
    """Only ordinary synchronous methods, including every wraps-linked layer."""
    try:
        unwrapped=inspect.unwrap(method)
    except (ValueError,TypeError):
        return False
    current=method
    while True:
        if not (inspect.isfunction(current) or inspect.ismethod(current)):
            return False
        if (inspect.isgeneratorfunction(current) or inspect.iscoroutinefunction(current)
                or inspect.isasyncgenfunction(current)):
            return False
        if current is unwrapped:
            return True
        current=getattr(current,'__wrapped__',unwrapped)


def inspect_test_population(suite):
    """Discover canonical method identities without calling fixtures or bodies."""
    def leaves(node):
        if isinstance(node,unittest.TestSuite):
            for child in node:
                yield from leaves(child)
        else:
            yield node
    population=list(leaves(suite));selected=[];invalid=[]
    for test in population:
        if not isinstance(test,unittest.TestCase):
            invalid.append('non-TestCase selection')
            continue
        method=getattr(test,'_testMethodName',None)
        canonical=type(test).__module__+'.'+type(test).__qualname__+'.'+str(method)
        selected.append(canonical)
        if (type(method) is not str or not ordinary_sync_method(getattr(test,method,None))
                or test.id() != canonical or isinstance(test,unittest.loader._FailedTest)
                or getattr(test.run,'__func__',None) is not unittest.TestCase.run):
            invalid.append(canonical)
    if len(selected)!=len(set(selected)):
        invalid.append('duplicate selected method identity')
    return population,selected,invalid


def execute_test_suite(suite):
    """Run an exact population and attribute assertions to actual body entry.

    An assertion may leave a body exceptionally; body_finished means the call
    exited, not that its assertion passed. Its phase stays 'body' until unittest
    records the failure, and the following lifecycle wrapper changes the phase.
    Body-scoped subtests carry explicit parent, phase and outcome events.
    Selected methods must be ordinary synchronous methods and return None;
    obtaining a lazy call object does not count as executing its deferred body.
    """
    receipt={'receipt_schema':'cbz-unittest-lifecycle-v1',
             'selected_ids':[],'started_ids':[],'finished_ids':[],
             'body_entered_ids':[],'body_finished_ids':[],'body_assertion_ids':[],'assertion_events':[],
             'invalid_population':[],'lifecycle_failures':0,'subtests':0,'subtest_events':[],
             'failure_events':[]}
    population,selected,invalid=inspect_test_population(suite)
    receipt['selected_ids']=selected
    receipt['invalid_population']=invalid
    if invalid:
        receipt.update(tests_run=0,assertion_failures=0,errors=0,skipped=0,
                       expected_failures=0,unexpected_successes=0,successful=False,
                       log='Refused unsupported selected test population before execution.\n')
        return receipt
    for test in population:
        if not isinstance(test,unittest.TestCase):
            continue
        method=getattr(test,'_testMethodName',None)
        canonical=type(test).__module__+'.'+type(test).__qualname__+'.'+str(method)
        if canonical in invalid:
            continue
        test._cbz_phase='selected'
        test._cbz_body_active=False
        test._cbz_body_exception=None
        test._cbz_dispatch_depth=0
        test._cbz_lifecycle_depth=0
        # Wrap the actual selected method, rather than treating startTest as
        # evidence that setup succeeded or the test body was called.
        original=getattr(test,method)
        @functools.wraps(original)
        def body(*args,_test=test,_original=original,_id=canonical,**kwargs):
            if not (_test._cbz_dispatch_depth>0 and _test._cbz_lifecycle_depth==0):
                receipt['invalid_population'].append('selected method invoked outside body dispatch: '+_id)
                return _original(*args,**kwargs)
            _test._cbz_phase='body'
            _test._cbz_body_active=True
            receipt['body_entered_ids'].append(_id)
            try:
                value=_original(*args,**kwargs)
                if value is not None:
                    receipt['invalid_population'].append('non-None return from selected method: '+_id)
                    # Closing a fresh native coroutine does not enter its body.
                    # Never close custom awaitables or advance any lazy object.
                    if (inspect.iscoroutine(value)
                            and inspect.getcoroutinestate(value)==inspect.CORO_CREATED):
                        value.close()
                    return None
                return value
            except BaseException as exc:
                _test._cbz_body_exception=exc
                raise
            finally:
                _test._cbz_body_active=False
                receipt['body_finished_ids'].append(_id)
        setattr(test,method,body)
        original_dispatch=test._callTestMethod
        def dispatch(*args,_test=test,_original=original_dispatch,**kwargs):
            _test._cbz_dispatch_depth+=1
            try:
                return _original(*args,**kwargs)
            finally:
                _test._cbz_dispatch_depth-=1
        test._callTestMethod=dispatch
        for name,phase in (('_callSetUp','setup'),('_callTearDown','teardown'),('_callCleanup','cleanup')):
            original=getattr(test,name)
            def lifecycle(*args,_test=test,_original=original,_phase=phase,**kwargs):
                _test._cbz_phase=_phase
                _test._cbz_lifecycle_depth+=1
                try:
                    return _original(*args,**kwargs)
                finally:
                    _test._cbz_lifecycle_depth-=1
            setattr(test,name,lifecycle)

    class PhaseResult(unittest.TextTestResult):
        def startTest(self,test):
            receipt['started_ids'].append(test.id())
            super().startTest(test)
        def stopTest(self,test):
            receipt['finished_ids'].append(test.id())
            super().stopTest(test)
        def addFailure(self,test,err):
            phase=getattr(test,'_cbz_phase','unknown')
            is_body=(phase == 'body' and issubclass(err[0],AssertionError)
                     and err[1] is getattr(test,'_cbz_body_exception',None))
            if is_body:
                receipt['body_assertion_ids'].append(test.id())
                receipt['assertion_events'].append({'id':test.id(),'source':'direct'})
            else:
                receipt['lifecycle_failures']+=1
            receipt['failure_events'].append({'id':test.id(),'phase':phase,'kind':'assertion'})
            super().addFailure(test,err)
        def addError(self,test,err):
            receipt['failure_events'].append({'id':test.id(),'phase':getattr(test,'_cbz_phase','fixture'),'kind':'error'})
            super().addError(test,err)
        def addSubTest(self,test,subtest,err):
            receipt['subtests']+=1
            phase=getattr(test,'_cbz_phase','unknown')
            outcome=('success' if err is None else
                     'assertion' if issubclass(err[0],AssertionError) else 'error')
            scoped=(phase=='body' and getattr(test,'_cbz_body_active',False)
                    and getattr(subtest,'test_case',None) is test)
            if not scoped:
                receipt['lifecycle_failures']+=1
            if scoped and outcome=='assertion':
                receipt['body_assertion_ids'].append(test.id())
                receipt['assertion_events'].append({'id':test.id(),'source':'subtest'})
            receipt['subtest_events'].append({'id':test.id(),'phase':phase,'outcome':outcome})
            super().addSubTest(test,subtest,err)

    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=PhaseResult).run(suite)
    receipt.update(tests_run=result.testsRun,assertion_failures=len(result.failures),
                   errors=len(result.errors),skipped=len(result.skipped),
                   expected_failures=len(result.expectedFailures),
                   unexpected_successes=len(result.unexpectedSuccesses),
                   successful=result.wasSuccessful(),log=stream.getvalue())
    return receipt


def internal_test_runner(argv):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--test-root',type=Path,required=True)
    parser.add_argument('--test-pattern',default='test_auditor.py')
    parser.add_argument('--test',action='append',default=[])
    parser.add_argument('--list',action='store_true')
    args=parser.parse_args(argv)
    sys.path.insert(0,str(args.test_root.parent))
    sys.path.insert(0,str(args.test_root))
    loader=unittest.TestLoader()
    suite=(loader.loadTestsFromNames(args.test) if args.test else
           loader.discover(str(args.test_root),pattern=args.test_pattern))
    if args.list:
        _,selected,invalid=inspect_test_population(suite)
        value={'receipt_schema':'cbz-unittest-population-v1','selected_ids':selected,'invalid_population':invalid}
        args.output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        return 0 if selected and not invalid else 1
    value=execute_test_suite(suite)
    args.output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(value['log'],end='')
    return 0 if value['successful'] else 1


def run_one(name, filename, transform, output_dir, tests=None):
    """Execute an unchanged selected behavioral population against a source mutant."""
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    original=(HERE/filename).read_text(encoding='utf-8')
    try:
        changed=transform(original)
        ast.parse(changed,filename=filename)
    except (SyntaxError,ValueError,TypeError) as exc:
        return {'name':name,'file':filename,'outcome':'INVALID','detail':str(exc),'assertion_failures':0,'errors':0}
    with tempfile.TemporaryDirectory(prefix='cbz-layer-a-mutant-') as temporary:
        root=Path(temporary);(root/'tests').mkdir()
        for source in ['auditor.py','source_checks.py','canonical_lean.py']:
            (root/source).write_bytes((HERE/source).read_bytes())
        (root/'tests/test_auditor.py').write_bytes((HERE/'tests/test_auditor.py').read_bytes())
        result_path=root/'test-result.json'
        command=[sys.executable,'-B','-S',str(Path(__file__).resolve()),'--internal-test-runner',
                 '--output',str(result_path),'--test-root',str(root/'tests')]
        for test in tests or []:
            command.extend(['--test',test])
        env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1'
        expected_ids=[];population={}
        try:
            population_path=root/'pristine-population.json'
            list_command=[str(population_path) if item==str(result_path) else item for item in command]+['--list']
            listed=subprocess.run(list_command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
            population=json.loads(population_path.read_text()) if population_path.is_file() else {}
            expected_ids=population.get('selected_ids',[])
            if (listed.returncode!=0 or population.get('receipt_schema')!='cbz-unittest-population-v1'
                    or population.get('invalid_population')!=[]):
                p=listed;value={}
            else:
                (root/filename).write_text(changed,encoding='utf-8')
                p=subprocess.run(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
                value=json.loads(result_path.read_text()) if result_path.is_file() else {}
        except subprocess.TimeoutExpired:
            return {'name':name,'file':filename,'outcome':'ERROR','detail':'behavioral suite timeout','assertion_failures':0,'errors':1}
        except (OSError,ValueError) as exc:
            return {'name':name,'file':filename,'outcome':'ERROR','detail':'runner receipt failure: '+str(exc),'assertion_failures':0,'errors':1}
        output=p.stdout
    log=output_dir/(name+'.log');log.write_text(output,encoding='utf-8')
    outcome=classify_test_result(value,p.returncode,expected_ids=expected_ids)
    return {'name':name,'file':filename,'outcome':outcome,'returncode':p.returncode,
            'tests_run':value.get('tests_run',0),'assertion_failures':value.get('assertion_failures',0),
            'errors':value.get('errors',0),'tests':value,'expected_ids':expected_ids,'pristine_population':population,'original_sha256':sha(original.encode()),
            'mutated_sha256':sha(changed.encode()),'log':str(log),'log_sha256':sha(output.encode())}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='G0 receipt JSON; logs in a receipt-specific sibling directory')
    args=parser.parse_args(argv)
    started=datetime.now(timezone.utc).isoformat()
    source_paths=['auditor.py','source_checks.py','canonical_lean.py','mutation_test.py','tests/test_auditor.py','tests/test_mutation_runner.py','SPEC.md','README.md','LEXER_PROVENANCE.json']
    starting_sources={p:sha((HERE/p).read_bytes()) for p in source_paths}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    logs=args.output.parent/(args.output.stem+'-mutation-logs');logs.mkdir(exist_ok=True)
    baseline_path=logs/'baseline-tests.json'
    command=[sys.executable,'-B','-S',str(Path(__file__).resolve()),'--internal-test-runner',
             '--output',str(baseline_path),'--test-root',str(HERE/'tests'),'--test-pattern','test_*.py']
    population_path=logs/'baseline-population.json'
    list_command=[str(population_path) if item==str(baseline_path) else item for item in command]+['--list']
    listed=subprocess.run(list_command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=HERE,timeout=60)
    try:
        baseline_population=json.loads(population_path.read_text())
    except (OSError,ValueError):
        baseline_population={}
    baseline=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=HERE,timeout=60)
    (logs/'baseline.log').write_text(baseline.stdout,encoding='utf-8')
    try:
        baseline_value=json.loads(baseline_path.read_text())
    except (OSError,ValueError):
        baseline_value={}
    tests,failures,errors=(baseline_value.get(k,0) for k in ('tests_run','assertion_failures','errors'))
    baseline_outcome=(classify_test_result(baseline_value,baseline.returncode,expected_ids=baseline_population.get('selected_ids',[]))
                      if listed.returncode==0 and baseline_population.get('receipt_schema')=='cbz-unittest-population-v1'
                      and baseline_population.get('invalid_population')==[] else 'ERROR')
    results=[]
    if baseline_outcome=='SURVIVED':
        for name,filename,before,after in MUTATIONS:
            def transform(source,before=before,after=after):
                if source.count(before)!=1:
                    raise ValueError('mutation anchor is not unique/present: '+repr(before))
                return source.replace(before,after,1)
            result=run_one(name,filename,transform,logs);results.append(result)
            print(name+': '+result['outcome'],flush=True)
    killed=sum(1 for r in results if r['outcome']=='KILLED')
    ending_sources={p:sha((HERE/p).read_bytes()) for p in source_paths}
    passed=starting_sources==ending_sources and baseline_outcome=='SURVIVED' and len(results)==len(MUTATIONS) and killed==len(MUTATIONS)
    report={'schema_version':1,'gate':'G0','observed_status':'PASS' if passed else 'FAIL',
        'started_utc':started,'ended_utc':datetime.now(timezone.utc).isoformat(),
        'command':[sys.executable,'-B','-S',str(HERE/'mutation_test.py'),'--output',str(args.output)],
        'baseline':{'command':command,'returncode':baseline.returncode,'outcome':baseline_outcome,'tests':baseline_value,'tests_run':tests,'assertion_failures':failures,'errors':errors,'log':str(logs/'baseline.log'),'log_sha256':sha(baseline.stdout.encode())},
        'threshold':{'required_killed':len(MUTATIONS),'declared_mutants':len(MUTATIONS),'killed':killed,'ratio':{'numerator':killed,'denominator':len(MUTATIONS)},'rule':'every declared valid checker-source mutant must be killed by assertions; syntax/import/runtime errors do not count'},
        'mutants':results,'source_sha256':ending_sources,'source_unchanged_during_run':starting_sources==ending_sources,
        'claim':'finite behavioral selftests and actual checker-source mutations; not a proof of universal scanner correctness'}
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print('G0 '+report['observed_status']+': '+str(tests)+' baseline tests; '+str(killed)+'/'+str(len(MUTATIONS))+' checker-source mutants killed')
    return 0 if passed else 1


if __name__=='__main__':
    if sys.argv[1:2] == ['--internal-test-runner']:
        raise SystemExit(internal_test_runner(sys.argv[2:]))
    raise SystemExit(main())
