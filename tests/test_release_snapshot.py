import hashlib
import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest
import zipfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_snapshot', ROOT / 'tools/release_snapshot.py')
m = importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader and (ROOT/'tools/release_snapshot.py').exists():
    spec.loader.exec_module(m)

class SnapshotTests(unittest.TestCase):
    def test_implementation_exists(self):
        self.assertTrue(hasattr(m, 'snapshot_local'), 'read-only collector is not implemented')

    def test_redirects_are_refused(self):
        self.assertTrue(hasattr(m, "NoRedirect"), "authenticated redirects must fail closed")
        self.assertIsNone(m.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://outside.invalid"))

    def test_sha(self):
        self.assertEqual(m.checked_sha('a'*40), 'a'*40)
        for value in ('main', 'abc', 'A'*40, 'a'*39+'g', '', None, 12):
            with self.subTest(value=value), self.assertRaises(ValueError): m.checked_sha(value)

    def test_owner(self):
        self.assertEqual(m.repo_url('Math-'), 'https://github.com/d6g8k5htny-coder/Math-.git')
        for value in ('other/repo', '../main', 'main;id', '', 'https://evil.invalid'):
            with self.subTest(value=value), self.assertRaises(ValueError): m.repo_url(value)

    def test_paths(self):
        self.assertEqual(m.checked_path('formal/A.lean'), 'formal/A.lean')
        for value in ('../a', '/a', 'a/../b', 'a//b', './a', 'a\\b', '', 'a\x00b'):
            with self.subTest(value=value), self.assertRaises(ValueError): m.checked_path(value)

    def test_pagination(self):
        calls=[]
        def get(path):
            calls.append(path)
            return list(range(100)) if 'page=1&' in path else [100]
        out=m.pages(get, 'repos/x/pulls')
        self.assertEqual(len(out),101)
        self.assertEqual(len(calls),2)

    def test_pagination_key(self):
        self.assertEqual(m.pages(lambda p:{'check_runs':[{'id':1}]}, 'repos/x/check-runs', 'check_runs'),[{'id':1}])

    def test_missing_key_not_empty(self):
        with self.assertRaises(ValueError): m.pages(lambda p:{}, 'repos/x/check-runs', 'check_runs')

    def test_error_not_empty(self):
        def bad(path): raise PermissionError('403')
        with self.assertRaises(PermissionError): m.pages(bad,'repos/x/pulls')

    def test_wrong_page_type(self):
        with self.assertRaises(ValueError): m.pages(lambda p:{'items':[]}, 'repos/x/pulls')

    def test_blob_hash(self):
        b=b'proof\n'; oid=hashlib.sha1(b'blob 6\0'+b).hexdigest()
        self.assertEqual(m.blob_digest(oid,b),hashlib.sha256(b).hexdigest())
        with self.assertRaises(ValueError): m.blob_digest(oid,b'wrong\n')

    def test_empty_output_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'out'; m.fresh_output(p)
            (p/'old').write_text('old')
            with self.assertRaises(FileExistsError): m.fresh_output(p)

    def test_exact_archive_with_export_ignore_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d); repo=root/'repo'; repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.DEVNULL).decode().strip()
            git('init','-q');git('config','user.email','test@example.invalid');git('config','user.name','test')
            (repo/'proof.txt').write_bytes(b'exact proof\n')
            (repo/'.gitattributes').write_text('proof.txt export-ignore\n')
            (repo/'alias').symlink_to('proof.txt')
            git('add','.');git('commit','-qm','fixture')
            sha=git('rev-parse','HEAD'); dest=root/'out';dest.mkdir()
            result=m.snapshot_local(repo,sha,dest/'source.zip')
            with zipfile.ZipFile(dest/'source.zip') as z:
                self.assertEqual(z.read('proof.txt'),b'exact proof\n')
                self.assertEqual(z.read('alias'),b'proof.txt')
                self.assertEqual(set(z.namelist()), {'.gitattributes','alias','proof.txt'})
            self.assertEqual(result['commit'],sha)
            self.assertEqual(len(result['members']),3)
            self.assertEqual(result['unmaterialized_gitlinks'],[])
            self.assertEqual(result['sha256'],hashlib.sha256((dest/'source.zip').read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError): m.snapshot_local(repo,sha,dest/'source.zip')

    def test_gitlink_is_recorded_not_followed(self):
        with tempfile.TemporaryDirectory() as d:
            repo=pathlib.Path(d)/'repo';repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.DEVNULL).decode().strip()
            git('init','-q');git('config','user.email','test@example.invalid');git('config','user.name','test')
            (repo/'x').write_text('x');git('add','.');git('commit','-qm','one');parent=git('rev-parse','HEAD')
            git('update-index','--add','--cacheinfo',f'160000,{parent},dependency');git('commit','-qm','gitlink')
            sha=git('rev-parse','HEAD');out=pathlib.Path(d)/'out.zip'
            result=m.snapshot_local(repo,sha,out)
            self.assertEqual(result['unmaterialized_gitlinks'],[{'path':'dependency','commit':parent}])
            with zipfile.ZipFile(out) as z: self.assertNotIn('dependency',z.namelist())

    def test_keyed_short_page_cannot_hide_total(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            m.pages(lambda _: {'total_count':250, 'check_runs':[{'id':1}]},
                    'repos/x/check-runs', 'check_runs')

    def test_keyed_total_count_success(self):
        self.assertEqual(m.pages(lambda _: {'total_count':1, 'check_runs':[{'id':1}]},
                         'repos/x/check-runs', 'check_runs'), [{'id':1}])

    def test_keyed_bad_total_count(self):
        for count in (True, -1, '1', 0.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                m.pages(lambda _: {'total_count':count, 'check_runs':[]},
                        'repos/x/check-runs', 'check_runs')

    def test_keyed_total_drift_fails_closed(self):
        def get(path):
            return {'total_count':101, 'check_runs':list(range(100))} if 'page=1&' in path else {'total_count':102,'check_runs':[100,101]}
        with self.assertRaisesRegex(ValueError, 'changed'):
            m.pages(get, 'repos/x/check-runs', 'check_runs')

    def test_control_character_paths_rejected(self):
        for value in ('line\nbreak', 'carriage\rreturn', 'tab\tfile', 'del\x7fete'):
            with self.subTest(value=value), self.assertRaises(ValueError): m.checked_path(value)

    def test_no_published_zip_after_late_identity_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d); repo=root/'repo';repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.DEVNULL).decode().strip()
            git('init','-q');git('config','user.email','test@example.invalid');git('config','user.name','test')
            (repo/'a.txt').write_text('a');(repo/'b.txt').write_text('b')
            git('add','.');git('commit','-qm','two files');sha=git('rev-parse','HEAD')
            out=root/'source.zip'; original=m.blob_digest; seen=[]
            def corrupt_second(oid,data):
                seen.append(oid)
                if len(seen)==2: raise ValueError('injected identity mismatch')
                return original(oid,data)
            with mock.patch.object(m, 'blob_digest', side_effect=corrupt_second):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    m.snapshot_local(repo,sha,out)
            self.assertEqual(len(seen),2)
            self.assertFalse(out.exists(), 'failed snapshot must not publish a readable partial zip')
            self.assertFalse(list(root.glob('.snapshot-*')), 'temporary snapshot was not cleaned')

if __name__ == '__main__': unittest.main()
