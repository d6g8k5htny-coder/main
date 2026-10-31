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
    def test_malformed_api_targets_preserve_partial_receipt_and_later_repository(self):
        # Regression: target-schema validation must not abort the collector.
        cases = ('metadata', 'branch', 'commit', 'empty-commit', 'pr-number',
                 'pr-head', 'pr-base', 'pr-author')
        for fault in cases:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as d:
                root = pathlib.Path(d)
                fixture = root / 'fixture'; fixture.mkdir()
                def git(*args):
                    return subprocess.check_output(['git', '-C', str(fixture), *args],
                                                   stderr=subprocess.DEVNULL).decode().strip()
                git('init', '-q'); git('config', 'user.email', 'test@example.invalid')
                git('config', 'user.name', 'test')
                (fixture / 'proof.txt').write_bytes(b'unchanged source\n')
                git('add', '.'); git('commit', '-qm', 'fixture')
                sha = git('rev-parse', 'HEAD')
                class API:
                    def get(self, path):
                        name = path.split('/')[2]
                        resource = '/'.join(path.split('/')[3:]).split('?')[0]
                        broken = name == 'main'
                        if not resource:
                            if broken and fault == 'metadata': return ['not metadata']
                            return {'private': False, 'full_name': f'{m.OWNER}/{name}',
                                    'default_branch': None if broken and fault == 'branch' else 'main'}
                        if resource == 'commits/main':
                            if broken and fault == 'empty-commit': return {}
                            return {'sha': 'invalid' if broken and fault == 'commit' else sha}
                        if resource == 'pulls':
                            if broken and fault == 'pr-number': return [{'number': '../outside'}]
                            return [{'number': 7}] if broken and fault.startswith('pr-') else []
                        if resource == 'pulls/7':
                            return {'number': 7, 'head': {'sha': 'invalid' if fault == 'pr-head' else sha,
                                                         'repo': {'full_name': f'{m.OWNER}/main'}},
                                    'base': {'sha': 'invalid' if fault == 'pr-base' else sha},
                                    'user': {} if fault == 'pr-author' else {'login': 'fixture-author'}}
                        if resource.endswith('check-runs'): return {'check_runs': [], 'total_count': 0}
                        if resource == 'actions/runs': return {'workflow_runs': [], 'total_count': 0}
                        if resource.endswith('/status'): return {'state': 'success'}
                        return []
                out = root / 'out'
                with mock.patch.object(m, 'REPOS', ('main', 'Math-')), \
                     mock.patch.object(m, 'Client', API), \
                     mock.patch.object(m, 'repo_url', return_value=str(fixture)):
                    result = m.collect(out)
                self.assertFalse(result['complete'])
                self.assertTrue(any(e['repository'] == 'main' for e in result['errors']))
                self.assertEqual([r['repository'] for r in result['repositories']],
                                 [f'{m.OWNER}/main', f'{m.OWNER}/Math-'])
                self.assertEqual(result['repositories'][1]['default_commit'], sha)
                self.assertEqual(len(result['repositories'][1]['snapshots']), 1)
                self.assertEqual(json.loads((out / 'SUMMARY.json').read_text()), result)
                sums = {}
                for line in (out / 'SHA256SUMS').read_text().splitlines():
                    digest, path = line.split('  ', 1); sums[path] = digest
                self.assertIn('SUMMARY.json', sums)
                for path, digest in sums.items():
                    self.assertEqual(hashlib.sha256((out / path).read_bytes()).hexdigest(), digest)

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


def _identity_fixture(root):
    fixture = root / 'fixture'; fixture.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(fixture), *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    git('init', '-q'); git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'test')
    (fixture / 'proof.txt').write_bytes(b'unchanged source\n')
    git('add', '.'); git('commit', '-qm', 'fixture')
    return fixture, git('rev-parse', 'HEAD')


def _identity_collect(root, overrides):
    """Mocked collect(); `overrides` maps (repository, resource) to raw rows."""
    fixture, sha = _identity_fixture(root)
    calls = []
    class API:
        def get(self, path):
            name = path.split('/')[2]
            resource = '/'.join(path.split('/')[3:]).split('?')[0]
            page = 2 if 'page=2&' in path else 1
            calls.append((name, resource))
            for (repo, wanted), value in overrides.items():
                if repo == name and (resource == wanted or
                                     (wanted.startswith('*') and resource.endswith(wanted[1:]))):
                    return value(page) if callable(value) else value
            if not resource:
                return {'private': False, 'full_name': f'{m.OWNER}/{name}', 'default_branch': 'main'}
            if resource == 'commits/main':
                return {'sha': sha}
            if resource == 'pulls':
                return [{'number': 7}] if name == 'main' else []
            if resource == 'pulls/7':
                return {'number': 7, 'head': {'sha': sha, 'repo': {'full_name': f'{m.OWNER}/main'}},
                        'base': {'sha': sha}, 'user': {'login': 'fixture-author'}}
            if resource.endswith('check-runs'): return {'check_runs': [], 'total_count': 0}
            if resource == 'actions/runs': return {'workflow_runs': [], 'total_count': 0}
            if resource.endswith('/status'): return {'state': 'success'}
            return []
    out = root / 'out'
    with mock.patch.object(m, 'REPOS', ('main', 'Math-')), \
         mock.patch.object(m, 'Client', API), \
         mock.patch.object(m, 'repo_url', return_value=str(fixture)):
        result = m.collect(out)
    return result, out, sha, calls


class IdentityPolicyTests(unittest.TestCase):
    # (resource label in the error record, fake-API resource, raw output path, raw rows)
    ROUTES = (
        ('/pulls/7/files', 'pulls/7/files', 'pr-7/files.json',
         [{'filename': 'a.txt', 'sha': 'b1'}, {'filename': 'a.txt', 'sha': 'b2'}]),
        ('/pulls/7/reviews', 'pulls/7/reviews', 'pr-7/reviews.json',
         [{'id': 11, 'state': 'COMMENTED'}, {'id': 11, 'state': 'COMMENTED'}]),
        ('/issues/7/comments', 'issues/7/comments', 'pr-7/comments.json',
         [{'id': 12, 'body': 'x'}, {'id': 12, 'body': 'y'}]),
        ('/pulls/7/comments', 'pulls/7/comments', 'pr-7/inline-comments.json', [{'id': 13}, {'id': 13}]),
        ('/check-runs', '*/check-runs', 'pr-7/checks.json', [{'id': 14}, {'id': 14}]),
        ('/actions/runs', 'actions/runs', 'pr-7/runs.json', [{'id': 15}, {'id': 15}]),
        ('/rulesets', 'rulesets', 'rulesets.json', [{'id': 16}, {'id': 16}]),
    )
    WRAPPED = {'/check-runs': 'check_runs', '/actions/runs': 'workflow_runs'}

    def test_duplicate_identity_is_refused_on_every_route_with_raw_rows_kept(self):
        for label, resource, raw_path, rows in self.ROUTES:
            with self.subTest(route=label), tempfile.TemporaryDirectory() as d:
                value = rows
                if label in self.WRAPPED:
                    value = {'total_count': len(rows), self.WRAPPED[label]: rows}
                result, out, sha, _ = _identity_collect(pathlib.Path(d), {('main', resource): value})
                errors = [e for e in result['errors'] if e['repository'] == 'main']
                self.assertFalse(result['complete'])
                self.assertEqual(len(errors), 1, errors)
                self.assertIn(label, errors[0]['resource'])
                self.assertIn('duplicate', errors[0]['error'])
                self.assertEqual(json.loads((out / 'main' / raw_path).read_text()), rows)
                later = result['repositories'][1]
                self.assertEqual(later['default_commit'], sha)
                self.assertEqual(len(later['snapshots']), 1)

    def test_duplicate_valid_pr_numbers_refuse_the_open_pr_collection(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{'number': 7, 'v': 'first'}, {'number': 7, 'v': 'second'}]
            result, out, sha, calls = _identity_collect(pathlib.Path(d), {('main', 'pulls'): rows})
            errors = [e for e in result['errors'] if e['repository'] == 'main']
            self.assertEqual([e['resource'] for e in errors], ['/pulls?state=open'])
            self.assertIn('duplicate', errors[0]['error'])
            self.assertIn('not processed', errors[0]['error'])
            self.assertEqual(json.loads((out / 'main' / 'open-prs.json').read_text()), rows)
            main = result['repositories'][0]
            self.assertNotIn('open_pr_count', main)
            self.assertEqual(main['pull_requests'], [])
            self.assertNotIn(('main', 'pulls/7'), calls)
            self.assertFalse((out / 'main' / 'pr-7').exists())
            self.assertEqual(len(result['repositories'][1]['snapshots']), 1)

    def test_invalid_then_valid_pr_keeps_partial_progress(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{'number': '../outside'}, {'number': 7}]
            result, out, sha, calls = _identity_collect(pathlib.Path(d), {('main', 'pulls'): rows})
            errors = [e for e in result['errors'] if e['repository'] == 'main']
            self.assertEqual(len(errors), 1)
            self.assertIn('positive integer PR number', errors[0]['error'])
            main = result['repositories'][0]
            self.assertEqual(main['open_pr_count'], 2)
            self.assertEqual([p['number'] for p in main['pull_requests']], [7])
            self.assertTrue((out / 'main' / 'pr-7' / 'detail.json').exists())

    def test_same_blob_at_different_filenames_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{'filename': 'a.txt', 'sha': 'same'}, {'filename': 'b.txt', 'sha': 'same'},
                    {'filename': 'A.txt', 'sha': 'same'}, {'filename': 'a.txt ', 'sha': 'same'}]
            result, out, sha, calls = _identity_collect(pathlib.Path(d), {('main', 'pulls/7/files'): rows})
            self.assertTrue(result['complete'], result['errors'])
            self.assertEqual(json.loads((out / 'main' / 'pr-7' / 'files.json').read_text()), rows)

    def test_repeated_id_across_pages_with_matching_total_is_refused(self):
        def runs(page):
            if page == 1:
                return {'total_count': 101, 'workflow_runs': [{'id': i} for i in range(1, 101)]}
            return {'total_count': 101, 'workflow_runs': [{'id': 100}]}
        with tempfile.TemporaryDirectory() as d:
            result, out, sha, calls = _identity_collect(pathlib.Path(d), {('main', 'actions/runs'): runs})
            errors = [e for e in result['errors'] if e['repository'] == 'main']
            self.assertEqual(len(errors), 1)
            self.assertIn('duplicate', errors[0]['error'])
            self.assertEqual(len(json.loads((out / 'main' / 'pr-7' / 'runs.json').read_text())), 101)

    def test_identity_values(self):
        check = m.check_identity
        check([{'id': 1}, {'id': 2}], m.ID_IDENTITY)
        check([], m.ID_IDENTITY)
        for rows in ([{'id': 1}, {'id': 1}], [{'id': 3, 'x': 1}, {'id': 3, 'x': 2}],
                     [{'id': True}], [{'id': '1'}], [{'id': 1.0}], [{'id': 0}], [{'id': -1}],
                     [{}], ['1'], [None]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                check(rows, m.ID_IDENTITY)
        check([{'filename': 'a'}, {'filename': 'A'}, {'filename': 'a '}], m.FILENAME_IDENTITY)
        for rows in ([{'filename': 'a', 'sha': 'x'}, {'filename': 'a', 'sha': 'y'}],
                     [{'filename': ''}], [{'filename': 1}], [{'sha': 'x'}], ['a']):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                check(rows, m.FILENAME_IDENTITY)
        # Malformed PR rows are left to the per-row path; only valid numbers must be unique.
        check([{'number': 'x'}, {'number': 'x'}, {'number': 7}, None], m.PR_IDENTITY)
        with self.assertRaises(ValueError):
            check([{'number': 7}, {'number': 'x'}, {'number': 7}], m.PR_IDENTITY)

    def test_missing_identity_policy_fails_closed(self):
        for policy in (None, 'id', object()):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                m.check_identity([{'id': 1}], policy)

    def test_every_paginated_read_names_its_identity_policy(self):
        import ast
        tree = ast.parse((ROOT / 'tools/release_snapshot.py').read_text())
        expected = {'/pulls?state=open': 'PR_IDENTITY', "f'/pulls/{n}/files'": 'FILENAME_IDENTITY',
                    "f'/pulls/{n}/reviews'": 'ID_IDENTITY', "f'/issues/{n}/comments'": 'ID_IDENTITY',
                    "f'/pulls/{n}/comments'": 'ID_IDENTITY', "f'/commits/{sha}/check-runs'": 'ID_IDENTITY',
                    "f'/actions/runs?head_sha={sha}'": 'ID_IDENTITY', '/rulesets': 'ID_IDENTITY'}
        seen = {}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'read'):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            if not (isinstance(kw.get('paginated'), ast.Constant) and kw['paginated'].value is True):
                self.assertNotIn('identity', kw)
                continue
            first = node.args[0]
            label = first.value if isinstance(first, ast.Constant) else ast.get_source_segment(
                (ROOT / 'tools/release_snapshot.py').read_text(), first)
            self.assertIn('identity', kw, label)
            self.assertIsInstance(kw['identity'], ast.Name, label)
            seen[label] = kw['identity'].id
        self.assertEqual(seen, expected)


if __name__ == '__main__': unittest.main()
