"""Finite mutation evidence: track real named test bodies and unittest phases.

A direct AssertionError in a registered test body can be mutation evidence.
Fixture/dispatcher failures, errors, skips, expected outcomes and subtest-only
failures cannot. This is an engineering classifier, not mathematical proof.
"""
from __future__ import annotations
import argparse
import functools
import hashlib
import importlib.util
import io
import inspect
import json
from pathlib import Path
import sys
import unittest

VERSION='cbz-mutation-lifecycle-v1'

def optimization_flags():
    return ['-OO'] if sys.flags.optimize==2 else ['-O'] if sys.flags.optimize==1 else []

def _cases(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite): yield from _cases(item)
        else: yield item

def run_suite(suite,stream,expected=None):
    """Run the unchanged suite; phase tracking does not replace its assertions."""
    cases=list(_cases(suite));population=[case.id() for case in cases]
    expected=list(population if expected is None else expected)
    states={};originals=[];events=[]
    population_valid=(bool(expected) and len(set(expected))==len(expected)
                      and len(set(population))==len(population) and population==expected)
    class Result(unittest.TextTestResult):
        def startTest(self,test):
            events.append(test.id());super().startTest(test)
        def addFailure(self,test,err):
            state=states.get(id(test));key=id(err[1])
            if state is None or key not in state['assertions']:
                if state is not None: state['invalid'].append('failure_outside_named_body')
            else: state['body_failures'].append(test.id())
            super().addFailure(test,err)
    def install(case,name,replacement):
        originals.append((case,name,name in case.__dict__,case.__dict__.get(name)))
        setattr(case,name,replacement)
    for case in cases:
        state=dict(phase=None,body_entries=0,body_returns=0,assertions=set(),body_failures=[],invalid=[])
        states[id(case)]=state
        method_name=getattr(case,'_testMethodName','')
        method=getattr(case,method_name,None)
        if not callable(method):
            state['invalid'].append('missing_named_body');continue
        unsupported=(inspect.iscoroutinefunction(method) or inspect.isgeneratorfunction(method)
                     or inspect.isasyncgenfunction(method))
        if unsupported: state['invalid'].append('unsupported_named_body')
        def body(*args,_method=method,_state=state,_unsupported=unsupported,**kwargs):
            _state['body_entries']+=1
            if _unsupported: raise TypeError('unsupported async or generator test body')
            if _state['phase']!='body': _state['invalid'].append('named_body_outside_dispatch')
            try:
                value=_method(*args,**kwargs)
            except AssertionError as error:
                _state['assertions'].add(id(error));raise
            else:
                if value is not None:
                    _state['invalid'].append('non_none_body_return')
                    if inspect.iscoroutine(value): value.close()
                _state['body_returns']+=1;return value
        body=functools.wraps(method)(body)
        install(case,method_name,body)
        for name,phase in (('_callSetUp','setup'),('_callTestMethod','body'),('_callTearDown','teardown'),('_callCleanup','cleanup')):
            original=getattr(case,name)
            def part(*args,_original=original,_phase=phase,_state=state,**kwargs):
                old=_state['phase'];_state['phase']=_phase
                try: return _original(*args,**kwargs)
                except BaseException:
                    if _phase!='body': _state['invalid'].append(_phase+'_exception')
                    raise
                finally: _state['phase']=old
            install(case,name,part)
    try:
        result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Result).run(suite)
    finally:
        for case,name,existed,value in reversed(originals):
            if existed: setattr(case,name,value)
            else: delattr(case,name)
    body_failures=[name for state in states.values() for name in state['body_failures']]
    invalid=[reason for state in states.values() for reason in state['invalid']]
    complete=(population_valid and events==expected and result.testsRun==len(expected)
              and all(state['body_entries']==1 for state in states.values()))
    # A subtest failure is diagnostic only. It never enters body_failures because
    # unittest catches it before the named body raises a direct AssertionError.
    clean_lifecycle=(not invalid and not result.errors and not result.skipped
                     and not result.expectedFailures and not result.unexpectedSuccesses)
    direct_failures=[test for test,_ in result.failures if id(test) in states]
    accounted=(len(direct_failures)==len(body_failures))
    return dict(schema='cbz.mutation-lifecycle.v1',tool=VERSION,optimization=sys.flags.optimize,
                optimization_args=optimization_flags(),population=population,expected_population=expected,population_valid=population_valid,
                started=events,tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                skips=len(result.skipped),expected_failures=len(result.expectedFailures),unexpected_successes=len(result.unexpectedSuccesses),
                body_entries={case.id():states[id(case)]['body_entries'] for case in cases},
                body_failures=body_failures,invalid=invalid,
                baseline_success=bool(complete and clean_lifecycle and result.wasSuccessful()),
                body_kill=bool(complete and clean_lifecycle and accounted and body_failures))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests',required=True,type=Path)
    parser.add_argument('--expected',type=Path)
    args=parser.parse_args(argv)
    stream=io.StringIO()
    try:
        before=hashlib.sha256(args.tests.read_bytes()).hexdigest()
        spec=importlib.util.spec_from_file_location('cbz_test_suite',args.tests)
        tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
        suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
        expected=json.loads(args.expected.read_text()) if args.expected else None
        report=run_suite(suite,stream,expected)
        report['test_sha256']=before
        report['output']=stream.getvalue()
        report['test_source_unchanged']=hashlib.sha256(args.tests.read_bytes()).hexdigest()==before
        if not report['test_source_unchanged']:
            report['baseline_success']=False;report['body_kill']=False
        print(json.dumps(report,sort_keys=True));return 0
    except Exception as error:
        print(json.dumps({'result':'REFUSED','error':type(error).__name__,'output':stream.getvalue()}));return 2

if __name__=='__main__': raise SystemExit(main())
