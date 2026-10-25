"""G2-02: the journal byte bound is applied to the exact canonical record.

proof_conductor._append used to add len(str(payload)) to the current size.
That is neither the envelope (seq, timestamp, type, prev_hash, event_hash) nor
the JSON spelling of the payload. Journal.append now measures canonical(d)+LF
inside its own transaction and rejects before opening the file for append.
"""
import inspect
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
FIXED = datetime(2026, 10, 25, 23, 30, 0, 123456, tzinfo=timezone.utc)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED


class JournalSerializedSizeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        engine = str(ROOT / 'engine')
        added = engine not in sys.path
        if added:
            sys.path.insert(0, engine)
        stubbed = False
        try:
            import model  # noqa: F401
        except ImportError:
            stub = types.ModuleType('model')
            stub.load = stub.validate = lambda *args, **kwargs: None
            stub.fingerprints = lambda *args, **kwargs: {}
            stub.digest = lambda *args, **kwargs: ''
            stub.GateError = type('GateError', (Exception,), {})
            sys.modules['model'] = stub
            stubbed = True
        import conductor
        cls.conductor = conductor

        def restore():
            if stubbed:
                sys.modules.pop('model', None)
            if added and engine in sys.path:
                sys.path.remove(engine)
        cls.addClassCleanup(restore)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workdir = Path(temp.name) / 'journal'
        clock = patch.object(self.conductor, 'datetime', _FixedDatetime)
        clock.start()
        self.addCleanup(clock.stop)

    def _record(self, journal, etype, payload):
        """The exact bytes Journal.append would write next."""
        c = self.conductor
        d = {'seq': len(journal.events) + 1, 'when_utc': FIXED.isoformat(), 'type': etype,
             'payload': payload, 'prev_hash': journal.events[-1]['event_hash'] if journal.events else c.ZERO}
        d['event_hash'] = c.hashlib.sha256(c.canonical(d)).hexdigest()
        return c.canonical(d) + b'\n'

    def _size(self, journal):
        return journal.path.stat().st_size if journal.path.exists() else 0

    def test_limit_is_exact_plus_minus_one_byte(self):
        journal = self.conductor.Journal(self.workdir)
        journal.append('seed', {'msg': 'first'})
        payload = {'msg': 'second'}
        record = self._record(journal, 'note', payload)
        before = journal.path.read_bytes()
        limit = len(before) + len(record)
        with self.assertRaisesRegex(ValueError, 'byte bound exceeded'):
            journal.append('note', payload, max_bytes=limit - 1)
        self.assertEqual(journal.path.read_bytes(), before)
        self.assertEqual(len(journal.events), 1)
        self.assertEqual(len(self.conductor.Journal(self.workdir).events), 1)
        event = journal.append('note', payload, max_bytes=limit)
        self.assertEqual(journal.path.read_bytes(), before + record)
        self.assertEqual(self._size(journal), limit)
        self.assertEqual(event['seq'], 2)

    def test_quote_and_escape_heavy_payload_is_sized_by_its_json_spelling(self):
        journal = self.conductor.Journal(self.workdir)
        journal.append('seed', {'msg': 'first'})
        payload = {'q': '"' * 1200, 'ctl': '\x01' * 100, 'nb': 'é' * 20}
        record = self._record(journal, 'note', payload)
        old_metric = len(str(payload).encode('utf8'))
        # JSON escapes each '"' (repr does not) and spells controls as \u0001
        # (repr uses \x01), so the real record is much larger than str(payload).
        self.assertGreater(len(record), old_metric + 1000)
        before = journal.path.read_bytes()
        limit = len(before) + old_metric
        with self.assertRaisesRegex(ValueError, 'byte bound exceeded'):
            journal.append('note', payload, max_bytes=limit)
        self.assertEqual(journal.path.read_bytes(), before)
        journal.append('note', payload, max_bytes=len(before) + len(record))
        reloaded = self.conductor.Journal(self.workdir)
        self.assertEqual(reloaded.events[-1]['payload'], payload)

    def test_payload_that_repr_overstates_is_accepted_when_it_fits(self):
        journal = self.conductor.Journal(self.workdir)
        journal.append('seed', {'msg': 'first'})
        # repr spells U+2028 as a six-character escape and doubles backslashes
        # in the same way JSON does; canonical JSON writes U+2028 as 3 raw bytes.
        payload = {'sep': ' ' * 600}
        record = self._record(journal, 'note', payload)
        self.assertGreater(len(str(payload).encode('utf8')), len(record))
        before = journal.path.read_bytes()
        journal.append('note', payload, max_bytes=len(before) + len(record))
        self.assertEqual(journal.path.read_bytes(), before + record)

    def test_small_payload_envelope_is_counted(self):
        journal = self.conductor.Journal(self.workdir)
        payload = {}
        record = self._record(journal, 'note', payload)
        # The envelope (timestamp, hashes, sequence) dominates a tiny payload.
        self.assertGreater(len(record), len(str(payload).encode('utf8')) + 150)
        with self.assertRaisesRegex(ValueError, 'byte bound exceeded'):
            journal.append('note', payload, max_bytes=len(record) - 1)

    def test_missing_journal_file_is_bounded_and_not_created_on_rejection(self):
        journal = self.conductor.Journal(self.workdir)
        self.assertFalse(journal.path.exists())
        record = self._record(journal, 'note', {'msg': 'x'})
        with self.assertRaisesRegex(ValueError, 'byte bound exceeded'):
            journal.append('note', {'msg': 'x'}, max_bytes=len(record) - 1)
        self.assertFalse(journal.path.exists())
        self.assertEqual(journal.events, [])
        journal.append('note', {'msg': 'x'}, max_bytes=len(record))
        self.assertEqual(journal.path.read_bytes(), record)

    def test_idempotent_replay_writes_nothing_and_is_not_rejected(self):
        journal = self.conductor.Journal(self.workdir)
        first = journal.append('note', {'msg': 'x'}, event_id='e-1')
        size = self._size(journal)
        again = journal.append('note', {'msg': 'x'}, event_id='e-1', max_bytes=size)
        self.assertEqual(again, first)
        self.assertEqual(self._size(journal), size)

    def test_unbounded_callers_are_unchanged(self):
        journal = self.conductor.Journal(self.workdir)
        record = self._record(journal, 'note', {'msg': 'x'})
        journal.append('note', {'msg': 'x'})
        self.assertEqual(journal.path.read_bytes(), record)

    def test_proof_conductor_delegates_exact_sizing_and_keeps_event_bound(self):
        source = (ROOT / 'proof_conductor.py').read_text(encoding='utf8')
        start = source.index('    def _append(self, kind, payload):')
        body = source[start:source.index('\n    def ', start + 1)]
        self.assertIn('self.journal.append(kind,payload,max_bytes=MAX_JOURNAL_BYTES)', body)
        self.assertIn('len(self.journal.events) >= MAX_EVENTS', body)
        self.assertNotIn('str(payload)', body)
        self.assertIn('max_bytes', inspect.signature(self.conductor.Journal.append).parameters)


if __name__ == '__main__':
    unittest.main()
