#!/usr/bin/env python3
"""Bounded, non-destructive assertion-only mutation experiment; no math certification."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
CASES=[
 ('comment-parent', 'if value.get("issue_url") != parent:', 'if False:', 'test_comment_parent_binding'),
 ('api-duplicate-json', 'return json.load(resp, object_pairs_hook=_unique_pairs)', 'return json.load(resp)', 'test_api_duplicate_json_members_rejected'),
 ('object-drift', 'before == after', 'True', 'test_changed_blob_is_stale'),
 ('mode-drift', 'before == after', 'before[2] == after[2]', 'test_mode_change_is_stale'),
 ('truncation', 'value.get("truncated") is not False', 'False', 'test_partial_tree_unknown'),
 ('tree-identity', 'value.get("sha") != sha', 'False', 'test_wrong_response_identity_unknown'),
 ('comment-direction', 'if current_z < recorded:', 'if False:', 'test_future_recorded_comment_unknown'),
 ('schema-preflight', 'if problems:', 'if False:', 'test_strict_invalid_records_never_repaired_to_green'),
 ('strict-exit', 'return 1 if args.check and (drift or not report or any(row["status"] != "fresh" for row in report)) else 0', 'return 0', 'test_strict_real_refresh_end_to_end'),
 ('input-drift', 'drift = before != snapshot()', 'drift = False', 'test_strict_rejects_input_drift_during_observation'),
 ('directory-drift', 'before == after', 'True', 'test_directory_changed_even_unchanged_child'),
]

# The child records code-object identities while unittest still holds the
# original traceback. Only the selected body and explicitly captured standard
# unittest assertion machinery may appear after the selected frame.
ORIGIN_RUNNER = r'''
import contextlib
import inspect
import io
import json
import sys
import unittest
import unittest.case

def code_of(value):
    return getattr(getattr(value, "__func__", value), "__code__", None)

oracle = []
for name in dir(unittest.TestCase):
    if name.startswith("assert") or name in ("fail", "_baseAssertEqual", "_formatMessage"):
        code = code_of(getattr(unittest.TestCase, name, None))
        if code is not None:
            oracle.append(code)
for cls, names in ((unittest.case._AssertRaisesContext, ("__exit__",)),
                   (unittest.case._AssertRaisesBaseContext, ("_raiseFailure",))):
    for name in names:
        code = code_of(getattr(cls, name, None))
        if code is not None:
            oracle.append(code)

def is_oracle(code):
    # Code objects can compare equal despite a different source filename.
    return any(code is allowed for allowed in oracle)

test_id = sys.argv[1]
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    suite = unittest.defaultTestLoader.loadTestsFromName(test_id)
    def leaves(node):
        if isinstance(node, unittest.TestSuite):
            for child in node:
                yield from leaves(child)
        else:
            yield node
    cases = list(leaves(suite))
    selected = cases[0] if len(cases) == 1 else None
    selected_name = getattr(selected, "_testMethodName", "")
    selected_code = code_of(getattr(selected, selected_name, None))
    selected_id = selected.id() if selected is not None else None
    state = {"entered": 0, "phase": "initial"}
    if selected is not None:
        selected_method = getattr(selected, selected_name, None)
        if (selected_code is None or inspect.iscoroutinefunction(selected_method) or
                inspect.isgeneratorfunction(selected_method) or inspect.isasyncgenfunction(selected_method)):
            raise ValueError("only synchronous selected test bodies are supported")
        for attr, phase in (("_callSetUp", "setup"), ("_callTearDown", "teardown"),
                            ("_callCleanup", "cleanup")):
            original = getattr(selected, attr)
            def wrap(*args, _original=original, _phase=phase, **kwargs):
                state["phase"] = _phase
                return _original(*args, **kwargs)
            setattr(selected, attr, wrap)
        def call_body(method):
            if code_of(method) is not selected_code:
                raise RuntimeError("selected test body changed")
            state["entered"] += 1
            state["phase"] = "body"
            try:
                returned = method()
                if returned is not None:
                    if inspect.iscoroutine(returned) or inspect.isgenerator(returned):
                        returned.close()
                    raise RuntimeError("unsupported test-body return value")
            finally:
                state["phase"] = "body-exited"
        selected._callTestMethod = call_body

    class Result(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.events = []

        def record(self, test, err, kind):
            case = getattr(test, "test_case", test)
            frames = []
            tb = err[2]
            while tb is not None:
                code = tb.tb_frame.f_code
                frames.append((code, tb.tb_lineno))
                tb = tb.tb_next
            indexes = [i for i, (code, _) in enumerate(frames) if code is selected_code]
            tail = frames[indexes[-1] + 1:] if indexes else frames
            authorized = (kind == "failure" and case is selected and selected_id == test_id
                          and selected_code is not None and len(indexes) == 1
                          and state["entered"] == 1 and state["phase"] in ("body", "body-exited")
                          and all(is_oracle(code) for code, _ in tail))
            self.events.append({"test_id": case.id(), "kind": kind, "authorized": authorized,
                                "exception": err[0].__name__,
                                "frames": [{"name": code.co_name,
                                            "filename": code.co_filename,
                                            "line": line,
                                            "selected": code is selected_code,
                                            "oracle": is_oracle(code)}
                                           for code, line in frames]})

        def addFailure(self, test, err):
            self.record(test, err, "failure")
            super().addFailure(test, err)

        def addError(self, test, err):
            self.record(test, err, "error")
            super().addError(test, err)

        def addSubTest(self, test, subtest, err):
            if err is not None:
                kind = "failure" if issubclass(err[0], test.failureException) else "error"
                self.record(subtest, err, kind)
            super().addSubTest(test, subtest, err)

    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2, resultclass=Result).run(suite)
    report = {"schema": "node-freshness-origin-v2", "test_id": test_id,
              "selected_id": selected_id, "tests_run": result.testsRun,
              "optimize": sys.flags.optimize, "body_entered": state["entered"],
              "failure_count": len(result.failures), "error_count": len(result.errors),
              "skip_count": len(result.skipped), "expected_failure_count": len(result.expectedFailures),
              "unexpected_success_count": len(result.unexpectedSuccesses), "events": result.events,
              "captured_stdout": captured.getvalue()}
print(json.dumps(report, sort_keys=True))
sys.exit(0 if result.wasSuccessful() and not result.skipped and not result.expectedFailures else 1)
'''


def run_selected(root, test_id, optimized):
    if type(optimized) is not bool:
        return {'protocol_error': True, 'error': 'optimized must be boolean'}
    cmd = [sys.executable, '-B', '-S'] + (['-O'] if optimized else []) + ['-c', ORIGIN_RUNNER, test_id]
    env = {key: value for key, value in os.environ.items() if not key.startswith('PYTHON')}
    child = subprocess.run(cmd, cwd=root, env=env, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    try:
        report = json.loads(child.stdout)
        if not isinstance(report, dict) or report.get('schema') != 'node-freshness-origin-v2':
            raise ValueError('unexpected provenance schema')
    except (ValueError, TypeError):
        return {'protocol_error': True, 'child_exit': child.returncode,
                'stdout': child.stdout, 'output': child.stderr}
    report.update(protocol_error=False, child_exit=child.returncode, output=child.stderr,
                  requested_optimize=int(optimized))
    return report


def _common(result, test_id):
    count_keys = ('tests_run', 'failure_count', 'error_count', 'skip_count',
                  'expected_failure_count', 'unexpected_success_count', 'child_exit',
                  'optimize', 'requested_optimize', 'body_entered')
    return (isinstance(result, dict) and all(type(result.get(key)) is int for key in count_keys)
            and result.get('schema') == 'node-freshness-origin-v2'
            and result.get('protocol_error') is False and result.get('test_id') == test_id
            and result.get('selected_id') == test_id and result.get('tests_run') == 1
            and result.get('body_entered') == 1 and result.get('optimize') in (0, 1)
            and result.get('optimize') == result.get('requested_optimize')
            and result.get('error_count') == 0 and result.get('skip_count') == 0
            and result.get('expected_failure_count') == 0
            and result.get('unexpected_success_count') == 0 and isinstance(result.get('events'), list))


def qualified_kill(result, test_id):
    return (_common(result, test_id) and result.get('child_exit') == 1
            and result['failure_count'] > 0
            and len(result['events']) == result['failure_count']
            and all(isinstance(event, dict) and event.get('kind') == 'failure' and event.get('test_id') == test_id
                    and event.get('authorized') is True for event in result['events']))


def baseline_ok(result, test_id):
    return (_common(result, test_id) and result.get('child_exit') == 0
            and result.get('failure_count') == 0 and not result['events'])

def main():
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        dest=Path(tmp)
        for directory in ('tools','tests','records/node_status','architecture','docs/site'):
            (dest/directory).mkdir(parents=True,exist_ok=True)
        for directory in ('tools','tests'):
            (dest/directory/'__init__.py').write_text('')
        paths=['tools/node_status_refresh.py','tools/validate_node_status.py','tests/test_node_status_freshness.py',
               'architecture/node-status.schema.json','docs/site/current.json']
        paths += [str(p.relative_to(ROOT)) for p in (ROOT/'records/node_status').glob('*.json')]
        for path in paths: shutil.copyfile(ROOT/path,dest/path)
        module=dest/'tools/node_status_refresh.py'; original=module.read_text()
        for optimized in (False,True):
            for name,old,new,method in CASES:
                if original.count(old)!=1: raise RuntimeError('nonunique mutation: '+name)
                module.write_text(original)
                test_id='tests.test_node_status_freshness.ContentTests.'+method
                baseline=run_selected(dest,test_id,optimized)
                if not baseline_ok(baseline,test_id): raise RuntimeError('baseline failed or vacuous: '+json.dumps(baseline))
                module.write_text(original.replace(old,new))
                result=run_selected(dest,test_id,optimized)
                killed=qualified_kill(result,test_id)
                rows.append(dict(name=name,optimized=optimized,killed=killed,
                                 output=result.get('output',''),provenance=result))
                if not killed: raise RuntimeError('not an authorized assertion kill: '+json.dumps(result))
        module.write_text(original)
    print(json.dumps({'cases':len(rows),'results':rows},indent=2))
    return 0
if __name__=='__main__': sys.exit(main())
