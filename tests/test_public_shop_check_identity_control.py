"""Proposed W8b control (Grok Bot agent 2): tools/public_shop_check.py must bind every
pinned public input to its declared URL/commit and to BOTH its declared byte count and
sha256 before any bytes reach the downstream shop checks. Offline: urlopen, ROOT and
subprocess.run are replaced. Positive case: exact pinned bytes are accepted, so a
reject-everything implementation fails. Negative cases assert the specific handled
refusal (ValueError with the exact message) and that no downstream check consumed the
bytes. They do not accept an arbitrary exception or exit."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from urllib.parse import quote

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/public_shop_check.py'
SPEC = importlib.util.spec_from_file_location('public_shop_check_control', SCRIPT)
shop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shop)

PAYLOADS = {'coefficient': b'{"coefficient": 1}\n', 'proof': b'theorem fixture : True := trivial\n',
            'imports': b'import Fixture\n', 'query': b'{"query": "fixture"}\n'}


def pin(field, raw, commit='1' * 40):
    repository = 'd6g8k5htny-coder/query-' if field == 'query' else 'd6g8k5htny-coder/Math-'
    path = 'fixtures/' + field + ' file.txt'
    url = 'https://raw.githubusercontent.com/' + repository + '/' + commit + '/' + quote(path, safe='/')
    return {'repository': repository, 'commit': commit, 'path': path, 'url': url,
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


class FakeResponse:
    def __init__(self, raw):
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, limit):
        return self.raw[:limit]


class PublicShopIdentityControl(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / 'docs/site').mkdir(parents=True)
        (self.root / 'docs/notebooks').mkdir(parents=True)
        (self.root / 'docs/notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb').write_text(
            json.dumps({'nbformat': 4, 'cells': [{'cell_type': 'markdown', 'source': 'fixture'}]}))
        self.config = {field: pin(field, raw) for field, raw in PAYLOADS.items()}
        self.served = {self.config[f]['url']: raw for f, raw in PAYLOADS.items()}

    def run_main(self):
        (self.root / 'docs/site/config.json').write_text(json.dumps(self.config))
        fetched, out = [], io.StringIO()

        def fake_urlopen(url, timeout):
            fetched.append(url)
            return FakeResponse(self.served[url])
        with mock.patch.object(shop, 'ROOT', self.root), \
                mock.patch.object(shop, 'urlopen', fake_urlopen), \
                mock.patch.object(shop.subprocess, 'run') as run, \
                contextlib.redirect_stdout(out):
            try:
                return shop.main(), out.getvalue(), run, fetched, None
            except Exception as error:  # returned so each test asserts its exact type and message
                return None, out.getvalue(), run, fetched, error

    def assert_refused(self, outcome, message):
        result, out, run, fetched, error = outcome
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), message)
        self.assertEqual(run.call_count, 0, 'refused bytes must not reach the downstream checks')
        self.assertNotIn('checked', out)
        return fetched

    # --- positive -----------------------------------------------------------------
    def test_exact_pinned_bytes_are_accepted_and_handed_downstream(self):
        result, out, run, fetched, error = self.run_main()
        self.assertIsNone(error)
        self.assertIsNone(result)
        self.assertEqual(fetched, [self.config[f]['url'] for f in ('coefficient', 'proof', 'imports', 'query')])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][:3], ['python3', '-B', 'tools/public_shop_data.py'])
        self.assertIn('Pinned public sources, static interactions and notebook structure checked.', out)

    # --- negatives: the specific handled refusal -----------------------------------
    def test_same_length_substituted_bytes_are_refused(self):
        # Only the sha256 predicate can see this (kills no-check and length-only variants).
        self.served[self.config['proof']['url']] = PAYLOADS['proof'].upper()
        self.assert_refused(self.run_main(), 'Public source identity mismatch: proof')

    def test_declared_length_mismatch_is_refused_even_with_the_right_digest(self):
        # Correct bytes and sha256, wrong declared length (kills no-check and digest-only variants).
        self.config['imports']['bytes'] += 1
        self.assert_refused(self.run_main(), 'Public source identity mismatch: imports')

    def test_url_not_derived_from_the_pin_is_refused_before_fetch(self):
        self.config['query']['url'] = self.config['query']['url'].replace('1' * 40, '2' * 40)
        fetched = self.assert_refused(self.run_main(), 'Pin URL/identity mismatch')
        self.assertNotIn(self.config['query']['url'], fetched)

    def test_short_commit_is_refused_before_fetch(self):
        self.config['coefficient'] = pin('coefficient', PAYLOADS['coefficient'], commit='1' * 39)
        self.served = {self.config[f]['url']: raw for f, raw in PAYLOADS.items()}
        fetched = self.assert_refused(self.run_main(), 'Pin URL/identity mismatch')
        self.assertEqual(fetched, [])


if __name__ == '__main__':
    unittest.main()
