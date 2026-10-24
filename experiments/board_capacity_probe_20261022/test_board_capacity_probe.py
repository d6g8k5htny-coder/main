"""Offline synthetic contract tests; no GitHub calls or board mutations."""
from __future__ import annotations
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

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'board_capacity_probe.py'
URL = 'https://github.com/d6g8k5htny-coder/main/issues/475'
NOW = '2026-10-21T22:45:00Z'

def fixture(count=186):
    return {'observed_at': '2026-10-21T22:44:00Z', 'issue': {
        'url': 'https://api.github.com/repos/d6g8k5htny-coder/main/issues/475',
        'repository_url': 'https://api.github.com/repos/d6g8k5htny-coder/main',
        'html_url': URL, 'id': 5966843259, 'number': 475,
        'state': 'open', 'locked': False, 'comments': count}}

class ProbeContract(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.is_file(), 'board-capacity prototype not implemented')
        spec = importlib.util.spec_from_file_location('capacity_under_test', SOURCE)
        self.p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.p)

    def check(self, value=None, **kwargs):
        return self.p.assess(fixture() if value is None else value,
                             URL, NOW, **kwargs)

    def unknown(self, value, reason, **kwargs):
        got = self.check(value, **kwargs)
        self.assertEqual(got['status'], 'UNKNOWN')
        self.assertEqual(got['capacity'], 'UNKNOWN')
        self.assertIn(reason, got['reasons'])
        self.assertTrue(got['attention_required'])
        self.assertEqual(self.p.exit_code(got), 2)
        return got

    def test_normal_is_advisory_not_permission(self):
        got = self.check()
        self.assertEqual(got['status'], 'OK')
        self.assertEqual(got['capacity'], 'NORMAL')
        self.assertEqual(got['comments'], 186)
        self.assertEqual(got['age_seconds'], 60)
        self.assertEqual(got['posting_permission'], 'NOT_ESTABLISHED')
        self.assertEqual(got['scientific_effect'], 'NONE')
        self.assertIs(got['advisory_only'], True)
        self.assertFalse(got['attention_required'])
        self.assertEqual(self.p.exit_code(got), 0)

    def test_exact_threshold_boundaries(self):
        for count, level in [(0,'NORMAL'),(1999,'NORMAL'),(2000,'PREPARE'),
                             (2249,'PREPARE'),(2250,'CUTOVER'),(2499,'CUTOVER'),
                             (2500,'CUTOVER'),(999999,'CUTOVER')]:
            with self.subTest(count=count):
                got = self.check(fixture(count))
                self.assertEqual(got['capacity'], level)
                self.assertEqual(got['status'], 'OK')
                self.assertEqual(self.p.exit_code(got), 0 if level=='NORMAL' else 1)

    def test_configurable_thresholds(self):
        self.assertEqual(self.check(fixture(9), prepare_at=10, cutover_at=20)['capacity'],'NORMAL')
        self.assertEqual(self.check(fixture(10), prepare_at=10, cutover_at=20)['capacity'],'PREPARE')
        self.assertEqual(self.check(fixture(20), prepare_at=10, cutover_at=20)['capacity'],'CUTOVER')
        for kw in ({'prepare_at':0},{'prepare_at':True},{'prepare_at':2000.0},
                   {'prepare_at':2250},{'cutover_at':1999},{'max_age_seconds':0},
                   {'max_age_seconds':True},{'max_age_seconds':float('inf')}):
            with self.subTest(kw=kw): self.unknown(fixture(),'INVALID_POLICY',**kw)

    def test_missing_fields_never_become_zero_or_unlocked(self):
        for key in tuple(fixture()['issue']):
            x=fixture();del x['issue'][key]
            with self.subTest(key=key): self.unknown(x,'INVALID_ISSUE')
        for x in ({}, {'issue':fixture()['issue']}, {'observed_at':NOW},
                  {'observed_at':NOW,'issue':[]}, {'observed_at':NOW,'issue':None}):
            with self.subTest(x=x): self.unknown(x,'INVALID_SNAPSHOT')

    def test_count_is_nonnegative_integer_not_boolean_or_text(self):
        for value in (True,False,None,-1,1.0,'0','186',[],{}):
            with self.subTest(value=value): self.unknown(fixture(value),'INVALID_ISSUE')

    def test_identity_types(self):
        for key,value in [('id',True),('id',0),('number',True),('number',0),
                          ('number','475'),('locked',0),('locked','false'),
                          ('state','unknown'),('state',None)]:
            x=fixture();x['issue'][key]=value
            with self.subTest(key=key,value=value): self.unknown(x,'INVALID_ISSUE')

    def test_all_identity_components_are_bound(self):
        for key,value in [('number',419),('html_url',URL.replace('/475','/419')),
                          ('url',fixture()['issue']['url'].replace('/475','/419')),
                          ('repository_url',fixture()['issue']['repository_url']+'-other')]:
            x=fixture();x['issue'][key]=value
            with self.subTest(key=key): self.unknown(x,'ROUTE_MISMATCH')

    def test_pull_requests_are_not_boards_even_with_null_marker(self):
        for value in ({'url':'https://api.github.com/repos/a/b/pulls/475'},None,False,{}):
            x=fixture();x['issue']['pull_request']=value
            with self.subTest(value=value): self.unknown(x,'NOT_ISSUE')

    def test_url_full_match(self):
        for url in (URL+'/',URL+'#issuecomment-1',URL+'?x=1',URL+'\n',
                    URL.replace('https:','http:'),URL.replace('github.com','github.com.evil'),
                    URL.replace('/475','/0475'),URL.replace('/475','/٤٧٥'),
                    URL.replace('/issues/','/pull/'),URL.replace('/main/','/../'),
                    URL.replace('github.com/','name@github.com/')):
            with self.subTest(url=url):
                got=self.p.assess(fixture(),url,NOW)
                self.assertEqual(got['status'],'UNKNOWN')
                self.assertIn('INVALID_ACTIVE_URL',got['reasons'])

    def test_case_insensitive_repository_only(self):
        x=fixture()
        for key in ('html_url','url','repository_url'):
            x['issue'][key]=x['issue'][key].replace('d6g8k5htny-coder/main','D6G8K5HTNY-CODER/MAIN')
        self.assertEqual(self.check(x)['status'],'OK')
        x=fixture();x['issue']['url']=x['issue']['url'].replace('/repos/','/REPOS/')
        self.unknown(x,'ROUTE_MISMATCH')

    def test_stale_and_future_observations(self):
        x=fixture();x['observed_at']='2026-10-21T22:29:59Z'
        self.unknown(x,'STALE_SNAPSHOT')
        x['observed_at']='2026-10-21T22:45:01Z'
        self.unknown(x,'FUTURE_SNAPSHOT')
        x['observed_at']='2026-10-21T22:30:00Z'
        self.assertEqual(self.check(x)['status'],'OK')
        x['observed_at']='2026-10-21T17:44:00-05:00'
        self.assertEqual(self.check(x)['age_seconds'],60)

    def test_invalid_timestamps(self):
        for t in ('2026-10-21','2026-10-21T22:44:00','yesterday',
                  '2026-99-21T22:44:00Z','2026-10-21T22:44:00Z\n',None,True):
            x=fixture();x['observed_at']=t
            with self.subTest(t=t): self.unknown(x,'INVALID_TIME')
        got=self.p.assess(fixture(),URL,'bad')
        self.assertEqual(got['status'],'UNKNOWN')
        self.assertIn('INVALID_TIME',got['reasons'])

    def test_updated_at_is_not_observation_time(self):
        x=fixture();x['issue']['updated_at']='2020-01-01T00:00:00Z'
        self.assertEqual(self.check(x)['status'],'OK')
        del x['observed_at'];x['issue']['updated_at']=NOW
        self.unknown(x,'INVALID_SNAPSHOT')

    def test_closed_or_locked_is_not_permission(self):
        for state,locked,reasons in [('closed',False,['CLOSED']),('open',True,['LOCKED']),
                                     ('closed',True,['CLOSED','LOCKED'])]:
            x=fixture(2250);x['issue'].update(state=state,locked=locked)
            with self.subTest(state=state,locked=locked):
                got=self.check(x)
                self.assertEqual(got['status'],'UNAVAILABLE')
                self.assertEqual(got['capacity'],'CUTOVER')
                self.assertEqual(got['reasons'],reasons)
                self.assertEqual(self.p.exit_code(got),1)
                self.assertEqual(got['posting_permission'],'NOT_ESTABLISHED')

    def test_no_title_body_or_unrelated_fields_reflected(self):
        x=fixture();x['issue'].update(body='PRIVATE_CANARY',title='PRIVATE_CANARY',
                                    secret={'anything':'PRIVATE_CANARY'})
        before=copy.deepcopy(x)
        got=self.check(x)
        self.assertNotIn('PRIVATE_CANARY',json.dumps(got))
        self.assertEqual(x,before)
        self.assertNotIn('id',got)

    def test_invalid_snapshot_types(self):
        for x in ([],None,False,3,'not a snapshot'):
            with self.subTest(x=x):
                got=self.p.assess(x,URL,NOW)
                self.assertEqual(got['status'],'UNKNOWN')
                self.assertIn('INVALID_SNAPSHOT',got['reasons'])

    def test_strict_json(self):
        for raw in (b'{"x":1,"x":2}',b'{"issue":{"comments":1,"comments":2}}',
                    b'{"x":NaN}',b'{"x":Infinity}',b'{"x":-Infinity}',b'{}garbage',b'\xff'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError): self.p.decode_snapshot(raw)
        self.assertEqual(self.p.decode_snapshot(json.dumps(fixture()).encode()),fixture())

    def test_input_size_bound(self):
        with self.assertRaises(ValueError):
            self.p.decode_snapshot(b' '*(self.p.MAX_INPUT_BYTES+1))

    def run_cli(self, raw, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'snapshot.json';src.write_bytes(raw)
            p=subprocess.run([sys.executable,'-B','-S',*(['-O'] if sys.flags.optimize else []),
                str(SOURCE),str(src),'--active-url',URL,'--now',NOW,*extra],
                capture_output=True,timeout=5,env={'PATH':os.defpath,'LC_ALL':'C.UTF-8'})
            self.assertEqual(src.read_bytes(),raw)
            self.assertEqual(p.stderr,b'')
            self.assertTrue(p.stdout.strip(), 'CLI returned no diagnostic JSON')
            return p,json.loads(p.stdout)

    def test_cli_all_outcomes_and_exact_input_digest(self):
        for count,rc,level in [(186,0,'NORMAL'),(2000,1,'PREPARE'),(2250,1,'CUTOVER')]:
            raw=json.dumps(fixture(count)).encode()
            with self.subTest(count=count):
                p,got=self.run_cli(raw)
                self.assertEqual(p.returncode,rc)
                self.assertEqual(got['capacity'],level)
                self.assertEqual(got['input_sha256'],hashlib.sha256(raw).hexdigest())
        x=fixture();x['observed_at']='2026-10-20T22:44:00Z'
        p,got=self.run_cli(json.dumps(x).encode())
        self.assertEqual(p.returncode,2)
        self.assertEqual(got['status'],'UNKNOWN')

    def test_cli_malformed_never_prints_payload_or_success(self):
        for raw in (b'PRIVATE_CANARY',b'{"comments":0,"comments":2001}',b'{"secret":NaN}'):
            with self.subTest(raw=raw):
                p,got=self.run_cli(raw)
                self.assertEqual(p.returncode,2)
                self.assertEqual(got['status'],'UNKNOWN')
                self.assertNotIn(b'PRIVATE_CANARY',p.stdout)

    def test_cli_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=subprocess.run([sys.executable,'-B','-S',str(SOURCE),str(Path(tmp)/'absent'),
                              '--active-url',URL,'--now',NOW],capture_output=True,timeout=5)
        self.assertEqual(p.returncode,2)
        self.assertEqual(p.stderr,b'')
        self.assertEqual(json.loads(p.stdout)['reasons'],['INPUT_UNREADABLE'])

if __name__=='__main__':
    unittest.main(verbosity=2)
