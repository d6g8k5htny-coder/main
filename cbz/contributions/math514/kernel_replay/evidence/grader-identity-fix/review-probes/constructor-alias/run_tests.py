#!/usr/bin/env python3
"""Structured unittest execution; only complete selected test bodies grade mutants."""
import argparse
import functools
import inspect
import importlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
RECEIPT_SCHEMA = 'cbz-unittest-lifecycle-v1'



class IdentityMismatch(RuntimeError):
    """A selected declaration no longer denotes its frozen body."""


def function_chain(method):
    """Bind actual ordinary function and code objects, including wraps links."""
    chain=[];seen=set()
    while True:
        function=method.__func__ if inspect.ismethod(method) else method
        if not inspect.isfunction(function) or id(function) in seen:
            return None
        seen.add(id(function));chain.append((function,function.__code__))
        if not hasattr(function,'__wrapped__'):
            return tuple(chain)
        method=function.__wrapped__


def same_function_chain(left,right):
    return (left is not None and right is not None and len(left)==len(right)
            and all(a is c and b is d for (a,b),(c,d) in zip(left,right)))


def capture_identity(test):
    if not isinstance(test,unittest.TestCase):
        return None
    cls=type(test);name=getattr(test,'_testMethodName',None)
    if type(name) is not str:
        return None
    instance=function_chain(getattr(test,name,None))
    declared=function_chain(getattr(cls,name,None))
    if instance is None or declared is None:
        return None
    return {'class':cls,'name':name,
            'id':cls.__module__+'.'+cls.__qualname__+'.'+name,
            'instance':instance,'declared':declared,'wrapper':None}


def identity_matches(test,binding):
    if (binding is None or type(test) is not binding['class']
            or getattr(test,'_testMethodName',None)!=binding['name']
            or type(test).__module__+'.'+type(test).__qualname__+'.'+binding['name']!=binding['id']):
        return False
    if not same_function_chain(function_chain(getattr(type(test),binding['name'],None)),binding['declared']):
        return False
    live=getattr(test,binding['name'],None)
    if binding['wrapper'] is not None and live is not binding['wrapper']:
        return False
    original=(binding['instance'][0][0] if binding['wrapper'] is not None else live)
    return same_function_chain(function_chain(original),binding['instance'])


def suite_leaves(suite):
    for child in suite:
        if type(child) is unittest.TestSuite:
            yield from suite_leaves(child)
        else:
            yield child


class IdentityLoader(unittest.TestLoader):
    """Standard declarations only; never invoke module population factories."""
    def __init__(self):
        super().__init__();self.bindings={};self.invalid=[]

    def remember(self,suite):
        for test in suite_leaves(suite):
            binding=capture_identity(test)
            if binding is None:
                self.invalid.append('unsupported selected declaration')
            else:
                self.bindings[id(test)]=binding
        return suite

    def loadTestsFromTestCase(self,testCaseClass):
        return self.remember(super().loadTestsFromTestCase(testCaseClass))

    def loadTestsFromModule(self,module,*,pattern=None):
        # Derive the complete standard population before looking at load_tests.
        suites=[]
        for name in dir(module):
            value=getattr(module,name)
            if (isinstance(value,type) and issubclass(value,unittest.TestCase)
                    and value not in (unittest.TestCase,unittest.FunctionTestCase)):
                suites.append(self.loadTestsFromTestCase(value))
        suite=unittest.TestSuite(suites)
        if module.__dict__.get('load_tests') is not None:
            self.invalid.append('unsupported module load_tests hook: '+module.__name__)
        return suite

    def loadTestsFromName(self,name,module=None):
        # The standard loader also accepts arbitrary callable suite factories.
        # This profile supports modules, TestCase classes and exact methods only.
        try:
            parts=name.split('.')
            if module is None:
                for end in range(len(parts),0,-1):
                    try:
                        current=importlib.import_module('.'.join(parts[:end]));parts=parts[end:]
                        break
                    except ModuleNotFoundError as exc:
                        prefix='.'.join(parts[:end])
                        if exc.name!=prefix and not prefix.startswith(str(exc.name)+'.'):
                            raise
                else:
                    raise ValueError('selector module not found')
            else:
                current=module
            parent=None
            for part in parts:
                parent,current=current,getattr(current,part)
            if inspect.ismodule(current):
                return self.loadTestsFromModule(current)
            if isinstance(current,type) and issubclass(current,unittest.TestCase):
                return self.loadTestsFromTestCase(current)
            if (isinstance(parent,type) and issubclass(parent,unittest.TestCase)
                    and (inspect.isfunction(current) or inspect.ismethod(current))):
                return self.remember(unittest.TestSuite([parent(parts[-1])]))
            raise ValueError('unsupported callable or selector kind')
        except (ImportError,AttributeError,TypeError,ValueError) as exc:
            self.invalid.append('unsupported test selector: '+name+' ('+type(exc).__name__+')')
            return unittest.TestSuite()


def bound_population(suite,bindings=None,invalid=()):
    cases=list(suite_leaves(suite));problems=list(invalid)
    if bindings is None:
        bindings={id(test):capture_identity(test) for test in cases}
    for test in cases:
        if not identity_matches(test,bindings.get(id(test))):
            problems.append('selected class/function/code identity changed before execution')
    return cases,bindings,problems


def require_identities(cases,bindings,invalid):
    if any(not identity_matches(test,bindings.get(id(test))) for test in cases):
        invalid.append('selected class/function/code identity changed during execution')
        raise IdentityMismatch('selected body identity changed')

def method_id(test):
    """Use the actual selected method, rather than an overridable display id."""
    cls = type(test)
    name = getattr(test, '_testMethodName', None)
    if (not isinstance(test, unittest.TestCase) or cls.__module__ == 'unittest.loader'
            or not isinstance(name, str) or not callable(getattr(test, name, None))):
        return None
    return cls.__module__ + '.' + cls.__qualname__ + '.' + name


def supported_method_shape(method):
    """Only synchronous ordinary functions/methods are supported test bodies.

    Inspect every advertised decorator link. Returning suspended work is checked
    separately, since decorators may intentionally omit __wrapped__ metadata.
    """
    try:
        inspect.unwrap(method)  # Reject cyclic decorator metadata.
    except ValueError:
        return False
    current = method
    while True:
        if (not (inspect.isfunction(current) or inspect.ismethod(current))
                or inspect.isgeneratorfunction(current)
                or inspect.iscoroutinefunction(current)
                or inspect.isasyncgenfunction(current)):
            return False
        if not hasattr(current, '__wrapped__'):
            return True
        current = current.__wrapped__


def selected_cases(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from selected_cases(test)
        else:
            yield test


class LifecycleResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started_ids = []
        self.finished_ids = []
        self.body_entered_ids = []
        self.body_finished_ids = []
        self.body_assertion_ids = []
        self.lifecycle_failures = 0
        self.subtests = 0
        self.subtest_events = []
        self.assertion_events = []
        self.phase = {}
        self.body_exception = {}
        self.body_active = set()
        self.dispatch_active = set()
        self.lifecycle_depth = {}
        self.invalid_execution = []

    def startTest(self, test):
        require_identities(self.identity_cases,self.identity_bindings,self.invalid_execution)
        self.started_ids.append(method_id(test))
        self.phase[id(test)] = 'setup'
        super().startTest(test)

    def stopTest(self, test):
        self.finished_ids.append(method_id(test))
        super().stopTest(test)

    def addFailure(self, test, err):
        if (self.phase.get(id(test)) == 'body'
                and self.body_exception.get(id(test)) is err[1]
                and isinstance(err[1], AssertionError)):
            self.body_assertion_ids.append(method_id(test))
            self.assertion_events.append({'parent_id': method_id(test), 'origin': 'body'})
        else:
            self.lifecycle_failures += 1
        super().addFailure(test, err)

    def addSubTest(self, test, subtest, err):
        self.subtests += 1
        phase = self.phase.get(id(test))
        if phase == 'body' and id(test) not in self.body_active:
            phase = 'outside-body'
        if getattr(subtest, 'test_case', None) is not test:
            phase = 'foreign-parent'
        outcome = 'success' if err is None else 'assertion' if isinstance(err[1], AssertionError) else 'error'
        self.subtest_events.append({'parent_id': method_id(test), 'phase': phase, 'outcome': outcome})
        if err is not None and outcome == 'assertion':
            if phase == 'body':
                self.body_assertion_ids.append(method_id(test))
                self.assertion_events.append({'parent_id': method_id(test), 'origin': 'subtest'})
            else:
                self.lifecycle_failures += 1
        super().addSubTest(test, subtest, err)


def instrument(test, result, binding):
    """Keep body exception identity until unittest reports it after this call exits."""
    for name, phase in (('_callSetUp', 'setup'), ('_callTearDown', 'teardown'),
                        ('doCleanups', 'cleanup')):
        original = getattr(test, name)

        def lifecycle(*args, _original=original, _phase=phase, **kwargs):
            result.phase[id(test)] = _phase
            result.lifecycle_depth[id(test)] = result.lifecycle_depth.get(id(test), 0) + 1
            try:
                return _original(*args, **kwargs)
            finally:
                result.lifecycle_depth[id(test)] -= 1
        setattr(test, name, lifecycle)
    original_dispatch = test._callTestMethod

    def dispatch(*args, **kwargs):
        if result.lifecycle_depth.get(id(test), 0):
            result.invalid_execution.append('test dispatch invoked inside a lifecycle hook')
            return original_dispatch(*args, **kwargs)
        result.dispatch_active.add(id(test))
        try:
            return original_dispatch(*args, **kwargs)
        finally:
            result.dispatch_active.discard(id(test))
    test._callTestMethod = dispatch
    original_body = getattr(test, test._testMethodName)

    @functools.wraps(original_body)
    def body(*args, **kwargs):
        require_identities(result.identity_cases,result.identity_bindings,result.invalid_execution)
        if id(test) not in result.dispatch_active or result.lifecycle_depth.get(id(test), 0):
            result.invalid_execution.append('selected method invoked outside body dispatch')
            return original_body(*args, **kwargs)
        result.phase[id(test)] = 'body'
        result.body_entered_ids.append(method_id(test))
        result.body_active.add(id(test))
        try:
            value = original_body(*args, **kwargs)
            if value is not None:
                result.invalid_execution.append('selected synchronous method must return None')
                if inspect.iscoroutine(value) and inspect.getcoroutinestate(value) == inspect.CORO_CREATED:
                    value.close()  # Closing a native unstarted coroutine executes no body.
                return None
            return value
        except BaseException as exc:
            result.body_exception[id(test)] = exc
            raise
        finally:
            result.body_active.discard(id(test))
            result.body_finished_ids.append(method_id(test))
            require_identities(result.identity_cases,result.identity_bindings,result.invalid_execution)
            # _Outcome calls addFailure *after* this wrapper unwinds.
    setattr(test, test._testMethodName, body)
    binding['wrapper']=body


def describe_suite(suite, selection):
    cases = list(selected_cases(suite))
    selected = [method_id(test) for test in cases]
    invalid = []
    if not selected:
        invalid.append('empty selected population')
    if any(item is None for item in selected):
        invalid.append('synthetic or unsupported selected test')
    if len(set(selected)) != len(selected):
        invalid.append('duplicate selected method identity')
    for test, identity in zip(cases, selected):
        if getattr(test.run,'__func__',None) is not unittest.TestCase.run:
            invalid.append('unsupported TestCase.run override')
        if identity is not None and test.id() != identity:
            invalid.append('display identity does not match selected method')
        if identity is not None and not supported_method_shape(getattr(test, test._testMethodName)):
            invalid.append('unsupported selected method shape: ' + identity)
    return cases, {'selection': selection, 'selected_ids': selected, 'invalid_population': invalid}


def run_suite(suite, selection, bindings=None, invalid=()):
    cases, bindings, identity_errors = bound_population(suite,bindings,invalid)
    _, population = describe_suite(suite, selection)
    population['invalid_population'].extend(identity_errors)
    stream = io.StringIO()
    result = LifecycleResult(stream=unittest.runner._WritelnDecorator(stream),
                             descriptions=True, verbosity=2)
    result.identity_cases=cases;result.identity_bindings=bindings
    if not population['invalid_population']:
        for test in cases:
            instrument(test, result,bindings[id(test)])
        try:
            suite.run(result)
        except IdentityMismatch:
            result.stop()
    population['invalid_population'].extend(result.invalid_execution)
    # Render the ordinary summary for humans; grading uses the structured fields.
    result.printErrors()
    stream.write(f'Ran {result.testsRun} tests\n')
    value = {'receipt_schema': RECEIPT_SCHEMA, 'selection': selection,
             **population,
             'tests_run': result.testsRun, 'assertion_failures': len(result.failures),
             'errors': len(result.errors), 'skipped': len(result.skipped),
             'expected_failures': len(result.expectedFailures),
             'unexpected_successes': len(result.unexpectedSuccesses),
             'successful': result.wasSuccessful(), 'log': stream.getvalue()}
    for name in ('started_ids', 'finished_ids', 'body_entered_ids',
                 'body_finished_ids', 'body_assertion_ids', 'lifecycle_failures', 'subtests',
                 'subtest_events', 'assertion_events'):
        value[name] = getattr(result, name)
    return value


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', required=True)
    parser.add_argument('--test', action='append', default=[])
    parser.add_argument('--list', action='store_true', help='discover identities without running fixtures or test bodies')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'tests'))
    loader=IdentityLoader()
    if args.test:
        suite = loader.loadTestsFromNames(args.test)
    else:
        suite = loader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    selection = args.test or ['discover:test_*.py']
    if args.list:
        _, population = describe_suite(suite, selection)
        _,_,errors=bound_population(suite,loader.bindings,loader.invalid)
        population['invalid_population'].extend(errors)
        Path(args.output).write_text(json.dumps(population, indent=2) + '\n')
        return 0 if population['selected_ids'] and not population['invalid_population'] else 1
    value = run_suite(suite, selection,loader.bindings,loader.invalid)
    Path(args.output).write_text(json.dumps(value, indent=2) + '\n')
    print(value['log'], end='')
    complete = (not value['invalid_population'] and value['selected_ids']
                and all(value[key] == value['selected_ids'] for key in
                        ('started_ids', 'finished_ids', 'body_entered_ids', 'body_finished_ids')))
    return 0 if (complete and value['successful'] and not value['skipped']
                 and not value['expected_failures'] and not value['unexpected_successes']
                 and all(event['phase'] == 'body' and event['outcome'] == 'success'
                         for event in value['subtest_events'])
                 and not value['lifecycle_failures']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
