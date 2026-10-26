"""Worker liveness (audit G1-F2, G3-F1): a transient OSError must not end a worker loop.

The two loops are driven on stub objects, so the tests need neither engine/model.py nor formal_lane.py
(both absent from the never-merge carrier). Each loop is compiled from the exact source text of the method
in its module, so the code under test is the committed code, not a copy.
"""
import ast
import contextlib
import errno
import io
import queue
import sys
import threading
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _method(path, class_name, name, namespace):
    text = path.read_text(encoding='utf8')
    cls = [n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == class_name][-1]
    fn = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name][-1]
    exec(compile(ast.get_source_segment(text, fn), str(path), 'exec'), namespace)
    return namespace[name]


class StopAfter:
    """Stands in for threading.Event as _stop: is_set() turns True after `rounds` loop tests; wait() records delays."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.checks = 0
        self.waits = []

    def is_set(self):
        self.checks += 1
        return self.checks > self.rounds

    def wait(self, delay=None):
        self.waits.append(delay)
        return False


class Wake:
    """Stands in for formal_queue's _wake Event: records waits and clears."""

    def __init__(self):
        self.waits = []
        self.clears = 0

    def wait(self, delay=None):
        self.waits.append(delay)
        return False

    def clear(self):
        self.clears += 1


class Tasks:
    """Stands in for queue.Queue. get() follows a script: an item, or EMPTY for queue.Empty (the default)."""
    EMPTY = object()

    def __init__(self, script=()):
        self.script = list(script)
        self.done = 0
        self.put_items = []

    def get(self, timeout=None):
        item = self.script.pop(0) if self.script else self.EMPTY
        if item is self.EMPTY:
            raise queue.Empty
        return item

    def put(self, item):
        self.put_items.append(item)

    def task_done(self):
        self.done += 1


def _effects(*effects):
    """A callable that raises or returns each effect in turn, repeating the last one."""
    calls = []

    def call(*args, **kwargs):
        effect = effects[min(len(calls), len(effects) - 1)]
        calls.append(args)
        if isinstance(effect, BaseException):
            raise effect
        return effect
    call.calls = calls
    return call


def _enospc():
    return OSError(errno.ENOSPC, 'No space left on device')


def _quietly(worker, owner):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        worker(owner)
    return err.getvalue().splitlines()


class CoordinatorWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = staticmethod(_method(ROOT / 'engine' / 'conductor.py', 'Coordinator', '_worker', {'queue': queue, 'sys': sys}))

    def coordinator(self, rounds, script=(), recover=None, snapshot=None, run_checks=None, append=None):
        return types.SimpleNamespace(
            tasks=Tasks(script), _stop=StopAfter(rounds),
            recover_jobs=recover or _effects(None), snapshot=snapshot or _effects({'pending_job_count': 0}),
            run_checks=run_checks or _effects({'result': 'RECALCULATED_STILL_NEEDS_MATH_VERIFICATION'}),
            journal=types.SimpleNamespace(append=append or _effects(None)))

    def test_idle_oserror_backs_off_and_resumes(self):
        recover = _effects(_enospc(), _enospc(), None)
        c = self.coordinator(rounds=4, recover=recover, snapshot=_effects({'pending_job_count': 1}))
        lines = _quietly(self.worker, c)
        self.assertEqual(len(recover.calls), 4)
        self.assertEqual(c._stop.waits, [.2, .4])
        self.assertEqual(c.tasks.put_items, ['durable-recovery', 'durable-recovery'])
        self.assertEqual(len(lines), 1)
        self.assertIn('No space left on device', lines[0])

    def test_snapshot_oserror_is_transient_too(self):
        c = self.coordinator(rounds=3, snapshot=_effects(OSError(errno.EMFILE, 'Too many open files'), {'pending_job_count': 0}))
        _quietly(self.worker, c)
        self.assertEqual(c._stop.waits, [.2])

    def test_error_record_append_oserror_keeps_worker_and_task_accounting(self):
        c = self.coordinator(rounds=2, script=['replay'], run_checks=_effects(RuntimeError('check failed')),
                             append=_effects(_enospc()))
        _quietly(self.worker, c)
        self.assertEqual(c.tasks.done, 1)
        self.assertEqual(len(c.journal.append.calls), 1)
        self.assertEqual(c._stop.waits, [.2])

    def test_run_checks_failure_is_still_recorded(self):
        c = self.coordinator(rounds=1, script=['replay'], run_checks=_effects(RuntimeError('check failed')))
        self.assertEqual(_quietly(self.worker, c), [])
        self.assertEqual(c.tasks.done, 1)
        kind, note = c.journal.append.calls[0]
        self.assertEqual(kind, 'CHECK_RUN')
        self.assertEqual(note['result'], 'CHECK_ERROR')
        self.assertEqual(c._stop.waits, [])

    def test_backoff_doubles_to_a_cap_and_resets_after_progress(self):
        c = self.coordinator(rounds=9, recover=_effects(*([_enospc()] * 7 + [None, _enospc()])))
        lines = _quietly(self.worker, c)
        self.assertEqual(c._stop.waits, [.2, .4, .8, 1.6, 3.2, 5.0, 5.0, .2])
        self.assertEqual(len(lines), 2)  # one line per backoff streak

    def test_backoff_resets_after_a_processed_task(self):
        c = self.coordinator(rounds=3, script=[Tasks.EMPTY, 'replay', Tasks.EMPTY], recover=_effects(_enospc()))
        _quietly(self.worker, c)
        self.assertEqual(c.tasks.done, 1)
        self.assertEqual(c._stop.waits, [.2, .2])

    def test_integrity_value_error_still_ends_the_worker(self):
        c = self.coordinator(rounds=5, recover=_effects(ValueError('tampered event at line 3')))
        with self.assertRaisesRegex(ValueError, 'tampered event'):
            self.worker(c)
        self.assertEqual(c._stop.waits, [])

    def test_integrity_error_while_recording_a_failure_still_ends_the_worker(self):
        c = self.coordinator(rounds=5, script=['replay'], run_checks=_effects(RuntimeError('check failed')),
                             append=_effects(ValueError('incomplete journal tail')))
        with self.assertRaisesRegex(ValueError, 'incomplete journal tail'):
            self.worker(c)
        self.assertEqual(c.tasks.done, 1)


class JournalError(ValueError):
    """Same base class as formal_queue.JournalError."""


class FormalQueueWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = staticmethod(_method(ROOT / 'formal_queue.py', 'FormalQueue', '_worker', {'JournalError': JournalError, 'sys': sys}))

    def formal_queue(self, rounds, *effects, wake=None):
        return types.SimpleNamespace(_stop=StopAfter(rounds), _wake=wake or Wake(), _run_one=_effects(*effects))

    def test_oserror_from_run_one_backs_off_and_resumes(self):
        q = self.formal_queue(4, _enospc(), OSError(errno.EIO, 'Input/output error'), True, False)
        lines = _quietly(self.worker, q)
        self.assertEqual(len(q._run_one.calls), 4)
        self.assertEqual(q._stop.waits, [0.2, 0.4])
        self.assertEqual(q._wake.waits, [0.2])
        self.assertEqual(len(lines), 1)

    def test_backoff_waits_on_stop_even_with_a_pending_wake(self):
        wake = threading.Event()
        wake.set()  # e.g. an enqueue() or stop() before the fault: must not make the backoff return at once
        q = self.formal_queue(3, _enospc(), _enospc(), False, wake=wake)
        _quietly(self.worker, q)
        self.assertEqual(q._stop.waits, [0.2, 0.4])
        self.assertEqual(len(q._run_one.calls), 3)
        self.assertFalse(wake.is_set())  # the idle poll still clears it

    def test_backoff_doubles_to_a_cap_and_resets_after_a_job(self):
        q = self.formal_queue(9, *([_enospc()] * 7 + [True, _enospc()]))
        lines = _quietly(self.worker, q)
        self.assertEqual(q._stop.waits, [0.2, 0.4, 0.8, 1.6, 3.2, 5.0, 5.0, 0.2])
        self.assertEqual(len(lines), 2)

    def test_idle_poll_resets_backoff(self):
        q = self.formal_queue(3, _enospc(), False, _enospc())
        _quietly(self.worker, q)
        self.assertEqual(q._stop.waits, [0.2, 0.2])
        self.assertEqual(q._wake.waits, [0.2])

    def test_journal_error_still_stops_the_worker(self):
        q = self.formal_queue(5, JournalError('journal sequence, schema or digest mismatch'))
        self.worker(q)
        self.assertEqual(len(q._run_one.calls), 1)
        self.assertEqual(q._stop.waits, [])

    def test_other_exceptions_are_not_swallowed(self):
        for exc in (TypeError('unexpected'), ValueError('not a journal error'), KeyError('lease')):
            with self.subTest(type(exc).__name__):
                q = self.formal_queue(5, exc)
                with self.assertRaises(type(exc)):
                    self.worker(q)
                self.assertEqual(q._stop.waits, [])

    def test_idle_poll_unchanged(self):
        q = self.formal_queue(2, False, False)
        self.assertEqual(_quietly(self.worker, q), [])
        self.assertEqual(q._wake.waits, [0.2, 0.2])
        self.assertEqual(q._wake.clears, 2)
        self.assertEqual(q._stop.waits, [])


if __name__ == '__main__':
    unittest.main()
