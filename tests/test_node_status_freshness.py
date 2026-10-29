"""Content freshness, not last-touch commit equality; transport is injected offline."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import node_status_refresh as n

PIN='1'*40
HEAD='2'*40
TREE='3'*40
BLOB='4'*40
URL=f'https://github.com/d6g8k5htny-coder/main/blob/{PIN}/file'

def record():
    return dict(schema_version=0,node_id='x',owner=None,sources=[dict(url=URL,rev=PIN)],source_rev=PIN,status='fresh',scientific_effect='NONE')

class ContentTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        self.responses={
          '/repos/d6g8k5htny-coder/main/commits/HEAD':{'sha':HEAD,'commit':{'tree':{'sha':TREE}}},
          f'/repos/d6g8k5htny-coder/main/git/commits/{PIN}':{'sha':PIN,'tree':{'sha':TREE}},
          f'/repos/d6g8k5htny-coder/main/git/trees/{TREE}':{'sha':TREE,'truncated':False,'tree':[{'path':'file','mode':'100644','type':'blob','sha':BLOB}]}}
    def api(self,path):
        self.calls.append(path)
        return copy.deepcopy(self.responses[path])
    def observer(self):
        self.assertTrue(hasattr(n,'ContentObserver'), 'missing content-bound observer')
        return n.ContentObserver(self.api)
    def test_snapshot_commit_difference_is_not_content_drift(self):
        obs=self.observer(); out=n.refresh_record(record(),obs,'2026-10-29T20:00:00Z')
        self.assertEqual(out['status'],'fresh'); self.assertEqual(out['source_rev'],PIN)
        obs(URL); self.assertEqual(self.calls.count('/repos/d6g8k5htny-coder/main/commits/HEAD'),1)
        self.assertEqual(obs.heads,{'main':HEAD})
    def test_changed_blob_is_stale(self):
        other='5'*40
        self.responses[f'/repos/d6g8k5htny-coder/main/git/commits/{PIN}']['tree']['sha']=other
        self.responses[f'/repos/d6g8k5htny-coder/main/git/trees/{other}']={'sha':other,'truncated':False,'tree':[{'path':'file','mode':'100644','type':'blob','sha':'6'*40}]}
        self.assertEqual(n.refresh_record(record(),self.observer())['status'],'stale')
    def test_mode_change_is_stale(self):
        other='5'*40
        self.responses[f'/repos/d6g8k5htny-coder/main/git/commits/{PIN}']['tree']['sha']=other
        self.responses[f'/repos/d6g8k5htny-coder/main/git/trees/{other}']={'sha':other,'truncated':False,'tree':[{'path':'file','mode':'100755','type':'blob','sha':BLOB}]}
        self.assertEqual(n.refresh_record(record(),self.observer())['status'],'stale')
    def test_partial_tree_unknown(self):
        self.responses[f'/repos/d6g8k5htny-coder/main/git/trees/{TREE}']['truncated']=True
        self.assertEqual(n.refresh_record(record(),self.observer())['status'],'unknown')
    def test_missing_path_unknown(self):
        self.responses[f'/repos/d6g8k5htny-coder/main/git/trees/{TREE}']['tree']=[]
        self.assertEqual(n.refresh_record(record(),self.observer())['status'],'unknown')
    def test_wrong_response_identity_unknown(self):
        self.responses[f'/repos/d6g8k5htny-coder/main/git/trees/{TREE}']['sha']='9'*40
        self.assertEqual(n.refresh_record(record(),self.observer())['status'],'unknown')
    def test_access_error_unknown(self):
        self.responses.clear()
        self.assertEqual(n.refresh_record(record(),self.observer())['status'],'unknown')
    def test_url_kind_mismatch_unknown(self):
        rec=record(); rec['sources'][0]['url']=URL.replace('/blob/','/tree/')
        self.assertEqual(n.refresh_record(rec,self.observer())['status'],'unknown')
    def test_strict_empty_non_green(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('sys.stdout'):
                self.assertNotEqual(n.main([tmp,'--check']),0)

    def test_directory_changed_even_unchanged_child(self):
        other='5'*40
        self.responses[f'/repos/d6g8k5htny-coder/main/git/commits/{PIN}']['tree']['sha']=other
        self.responses[f'/repos/d6g8k5htny-coder/main/git/trees/{other}']={'sha':other,'truncated':False,'tree':[{'path':'file','mode':'040000','type':'tree','sha':'6'*40}]}
        self.responses[f'/repos/d6g8k5htny-coder/main/git/trees/{TREE}']['tree'][0].update(mode='040000',type='tree')
        rec=record();rec['sources'][0]['url']=URL.replace('/blob/','/tree/')
        self.assertEqual(n.refresh_record(rec,self.observer())['status'],'stale')
    def test_duplicate_and_malformed_tree_entries_unknown(self):
        key=f'/repos/d6g8k5htny-coder/main/git/trees/{TREE}'
        original=copy.deepcopy(self.responses[key])
        for mutate in (lambda d:d['tree'].append(d['tree'][0]),
                       lambda d:d.pop('truncated'),
                       lambda d:d['tree'][0].update(mode='120000'),
                       lambda d:d['tree'][0].update(sha='no'),
                       lambda d:d['tree'][0].update(type='tree')):
            with self.subTest(mutate=mutate):
                self.responses[key]=copy.deepcopy(original);mutate(self.responses[key])
                self.assertEqual(n.refresh_record(record(),self.observer())['status'],'unknown')
    def test_comment_time_and_identity(self):
        url='https://github.com/d6g8k5htny-coder/main/issues/9#issuecomment-12'
        key='/repos/d6g8k5htny-coder/main/issues/comments/12'
        rec=record();rec['sources']=[dict(url=url,rev='2026-10-22T00:00:00Z')];rec['source_rev']=rec['sources'][0]['rev']
        for value,status in (({'id':12,'updated_at':'2026-10-22T00:00:00Z'},'fresh'),
                             ({'id':12,'updated_at':'2026-10-23T00:00:00Z'},'stale'),
                             ({'id':13,'updated_at':'2026-10-22T00:00:00Z'},'unknown'),
                             ({'id':12,'updated_at':'2026-99-22T00:00:00Z'},'unknown')):
            self.responses[key]=dict(value,issue_url="https://api.github.com/repos/d6g8k5htny-coder/main/issues/9")
            self.assertEqual(n.refresh_record(rec,self.observer())['status'],status)
    def test_strict_status_exit_and_report_only_bytes(self):
        for status in ('fresh','stale','unknown'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                source=Path(__file__).resolve().parents[1]/'records/node_status/eng.hard-gate.json'
                path=Path(tmp)/source.name;path.write_bytes(source.read_bytes());before=path.read_bytes()
                with patch.object(n,'ContentObserver',return_value=self.observer()), patch.object(n,'refresh_dir',return_value=[{'status':status}]),patch('sys.stdout'):
                    self.assertEqual(n.main([tmp,'--check']),0 if status=='fresh' else 1)
                self.assertEqual(path.read_bytes(),before)
    def test_same_repo_head_failure_stays_unknown(self):
        self.responses['/repos/d6g8k5htny-coder/main/commits/HEAD']['sha']='invalid'
        self.assertEqual(n.refresh_record(record(),self.observer())['status'],'unknown')

    def test_future_recorded_comment_unknown(self):
        url='https://github.com/d6g8k5htny-coder/main/issues/9#issuecomment-12'
        self.responses['/repos/d6g8k5htny-coder/main/issues/comments/12']={'id':12,'updated_at':'2026-10-22T00:00:00Z','issue_url':'https://api.github.com/repos/d6g8k5htny-coder/main/issues/9'}
        rec=record();rec['sources']=[dict(url=url,rev='2026-10-23T00:00:00Z')]
        self.assertEqual(n.refresh_record(rec,self.observer())['status'],'unknown')
    def test_strict_invalid_records_never_repaired_to_green(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'x.json').write_text(json.dumps(record()))
            with patch.object(n,'ContentObserver',return_value=self.observer()),patch('sys.stdout'):
                self.assertEqual(n.main([tmp,'--check']),1)

    def test_strict_real_refresh_end_to_end(self):
        source=Path(__file__).resolve().parents[1]/'records/node_status/eng.hard-gate.json'
        for status in ('fresh','stale','unknown'):
            with self.subTest(status=status),tempfile.TemporaryDirectory() as tmp:
                rec=json.loads(source.read_text());rec['sources']=[dict(url=URL,rev=PIN)];rec['source_rev']=PIN
                path=Path(tmp)/source.name;path.write_text(json.dumps(rec));before=path.read_bytes()
                if status=='stale':
                    other='5'*40
                    self.responses[f'/repos/d6g8k5htny-coder/main/git/commits/{PIN}']['tree']['sha']=other
                    self.responses[f'/repos/d6g8k5htny-coder/main/git/trees/{other}']={'sha':other,'truncated':False,'tree':[{'path':'file','mode':'100644','type':'blob','sha':'6'*40}]}
                if status=='unknown':
                    self.responses.clear()
                with patch.object(n,'ContentObserver',return_value=self.observer()),patch('sys.stdout'):
                    self.assertEqual(n.main([tmp,'--check']),0 if status=='fresh' else 1)
                self.assertEqual(path.read_bytes(),before)

    def test_strict_rejects_input_drift_during_observation(self):
        source=Path(__file__).resolve().parents[1]/'records/node_status/eng.hard-gate.json'
        with tempfile.TemporaryDirectory() as tmp:
            rec=json.loads(source.read_text());rec['sources']=[dict(url=URL,rev=PIN)];rec['source_rev']=PIN
            path=Path(tmp)/source.name;path.write_text(json.dumps(rec))
            def drift(endpoint):
                path.write_text('{"scientific_effect":"ACCEPTED"}')
                return self.api(endpoint)
            obs=n.ContentObserver(drift)
            with patch.object(n,'ContentObserver',return_value=obs),patch('sys.stdout'):
                self.assertEqual(n.main([tmp,'--check']),1)

    def test_comment_parent_binding(self):
        stamp = '2026-10-22T00:00:00Z'
        for repo in ('main', 'Math-'):
            parent = f'https://api.github.com/repos/d6g8k5htny-coder/{repo}/issues/9'
            for form in ('issues', 'pull'):
                url = f'https://github.com/d6g8k5htny-coder/{repo}/{form}/9#issuecomment-12'
                rec = record()
                rec['sources'] = [dict(url=url, rev=stamp)]
                rec['source_rev'] = stamp
                for candidate in (parent, None, '', parent + '/', parent + '?x=1',
                                  parent.replace('/9', '/10'), parent.replace('/issues/', '/pulls/'),
                                  parent.replace('/repos/', '/wrong/'), 9, {},
                                  parent.replace('/'+repo+'/', '/other/')):
                    with self.subTest(repo=repo, form=form, parent=candidate):
                        value = dict(id=12, updated_at=stamp)
                        if candidate is not None:
                            value['issue_url'] = candidate
                        out = n.refresh_record(rec, n.ContentObserver(lambda path: value))
                        self.assertEqual(out['status'], 'fresh' if candidate == parent else 'unknown')
                        if candidate != parent:
                            self.assertIn('parent identity mismatch', out['refresh_error'])

    def test_api_duplicate_json_members_rejected(self):
        for raw in (b'{"id":13,"id":12}', b'{"tree":{"sha":"bad","sha":"good"}}'):
            with self.subTest(raw=raw), patch.object(n.urllib.request, 'urlopen', return_value=io.BytesIO(raw)):
                with self.assertRaisesRegex(n.SourceError, 'duplicate JSON key'):
                    n._api_get('/repos/d6g8k5htny-coder/main/issues/comments/12')
