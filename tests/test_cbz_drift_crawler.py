"""Tests for the CBZ-0 dynamic federation drift crawler (gate G7). Scientific effect: NONE.

A synthetic federation is built from local Git repositories behind made-up
GitHub URLs (mapped with --local), so the production code path is exercised.
CBZ_DRIFT_CRAWLER may name an alternative crawler file for the mutation harness.
"""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get('CBZ_DRIFT_CRAWLER', ROOT / 'tools/cbz/drift_crawler.py'))
sys.path.insert(0, str(TARGET.parent))
spec = importlib.util.spec_from_file_location('cbz_drift_crawler_under_test', TARGET)
crawler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crawler)

OWNER = 'https://github.com/fed/'
MATHLIB = 'd13f23b7' + '0' * 32


def git(path, *args):
    return subprocess.run(['git', '-C', str(path), *args], check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout.decode().strip()


class Federation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.lib = self.repo('lib')
        self.write(self.lib, 'Pkg/Spec.lean', 'namespace Fed\ntheorem t : 1 = 1 := rfl\nend Fed\n')
        self.write(self.lib, 'Pkg/lake-manifest.json', json.dumps({'packages': [
            {'name': 'mathlib', 'type': 'git', 'url': 'https://github.com/leanprover-community/mathlib4', 'rev': MATHLIB}]}))
        self.l1 = self.commit(self.lib, 'L1')
        self.write(self.lib, 'Pkg/Spec.lean', 'namespace Fed\ntheorem t : 2 = 2 := rfl\nend Fed\n')
        self.l2 = self.commit(self.lib, 'L2')
        git(self.lib, 'checkout', '-q', '-b', 'side')
        self.write(self.lib, 'side.txt', 'side\n')
        self.side = self.commit(self.lib, 'side work')
        git(self.lib, 'checkout', '-q', 'main')
        self.consumer = self.repo('consumer')
        self.write(self.consumer, 'pins.json', json.dumps({'source': {'repository': 'fed/lib', 'commit': self.l1}}))
        self.c1 = self.commit(self.consumer, 'consumer')
        self.root = self.repo('root')
        self.gitlink(self.root, 'repos/lib', self.l2, OWNER + 'lib.git')
        self.gitlink(self.root, 'repos/consumer', self.c1, OWNER + 'consumer')
        self.commit_index(self.root, 'root pins')
        self.allowlist = self.base / 'allow.json'

    def tearDown(self):
        self.tmp.cleanup()

    def repo(self, name):
        path = self.base / name
        path.mkdir()
        git(path, 'init', '-q', '-b', 'main')
        git(path, 'config', 'user.email', 'cbz@example.invalid')
        git(path, 'config', 'user.name', 'cbz')
        return path

    def write(self, repo, rel, text):
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')

    def commit(self, repo, message):
        git(repo, 'add', '-A')
        git(repo, 'commit', '-q', '-m', message)
        return git(repo, 'rev-parse', 'HEAD')

    def commit_index(self, repo, message):
        """Commit exactly what is staged: `add -A` would stage gitlinks as deletions."""
        git(repo, 'commit', '-q', '-m', message)
        return git(repo, 'rev-parse', 'HEAD')

    def gitlink(self, repo, path, sha, url):
        modules = repo / '.gitmodules'
        old = modules.read_text(encoding='utf-8') if modules.exists() else ''
        modules.write_text(old + f'[submodule "{path}"]\n\tpath = {path}\n\turl = {url}\n', encoding='utf-8')
        git(repo, 'add', '.gitmodules')
        git(repo, 'update-index', '--add', '--cacheinfo', f'160000,{sha},{path}')

    def run_crawler(self, *extra, local=None):
        mapping = {'root': self.root, 'lib': self.lib, 'consumer': self.consumer}
        mapping.update(local or {})
        args = ['--root', OWNER + 'root', '--offline', '--cache', str(self.base / 'cache'),
                '--allowlist', str(self.allowlist), '--json', str(self.base / 'r.json'), *extra]
        for name, path in mapping.items():
            args += ['--local', f'{OWNER}{name}={path}']
        with contextlib.redirect_stdout(io.StringIO()):
            code = crawler.main(args)
        return code, json.loads((self.base / 'r.json').read_text(encoding='utf-8'))

    def blocking(self, report, check):
        return [f for f in report['findings'] if f['blocking'] and f['check'] == check]

    # -------------------------------------------------------------- topology

    def test_topology_is_discovered_from_root_pins(self):
        code, report = self.run_crawler()
        self.assertEqual(code, 0)
        urls = sorted(r['url'] for r in report['repositories'])
        self.assertEqual(urls, [OWNER + 'consumer', OWNER + 'lib.git', OWNER + 'root'])
        self.assertEqual(report['summary']['edges'], {'lake': 0, 'record': 1, 'submodule': 2})

    def test_discovery_is_transitive_through_records(self):
        extra = self.repo('extra')
        self.write(extra, 'README.md', 'extra\n')
        e1 = self.commit(extra, 'extra')
        self.write(self.lib, 'refs.json', json.dumps([{'repo': 'fed/extra', 'sha': e1}]))
        l3 = self.commit(self.lib, 'pin extra')
        code, report = self.run_crawler(local={'extra': extra})
        self.assertEqual(code, 0)
        self.assertIn(OWNER + 'extra', [r['url'] for r in report['repositories']])
        self.assertEqual(l3, next(r['head'] for r in report['repositories'] if r['url'].endswith('lib.git')))

    def test_unreachable_repository_blocks(self):
        self.write(self.consumer, 'more.json', json.dumps({'repository': 'fed/ghost', 'commit': self.l1}))
        self.commit(self.consumer, 'pin ghost')
        code, report = self.run_crawler()
        self.assertEqual(code, 1)
        self.assertTrue(any('ghost' in f.get('to', f.get('repository', '')) for f in self.blocking(report, 'P1')))

    def test_unreachable_root_blocks(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = crawler.main(['--root', OWNER + 'nowhere', '--offline', '--cache', str(self.base / 'cache'),
                                 '--allowlist', str(self.allowlist), '--json', str(self.base / 'r.json')])
        report = json.loads((self.base / 'r.json').read_text(encoding='utf-8'))
        self.assertEqual(code, 1)
        self.assertEqual(self.blocking(report, 'P1')[0]['repository'], OWNER + 'nowhere')

    def test_head_override_and_repository_emission(self):
        self.write(self.consumer, 'pins.json', json.dumps({'repository': 'fed/lib', 'commit': '9' * 40}))
        broken = self.commit(self.consumer, 'bad pin')
        emitted = self.base / 'repos.txt'
        code, _ = self.run_crawler('--head', f'{OWNER}consumer={self.c1}', '--emit-repos', str(emitted))
        self.assertEqual(code, 0)
        lines = emitted.read_text(encoding='utf-8').split()
        self.assertIn(f'consumer={self.consumer}@{self.c1}', lines)
        self.assertEqual(len(lines), 3)
        code, _ = self.run_crawler('--head', f'{OWNER}consumer={broken}')
        self.assertEqual(code, 1)

    # -------------------------------------------------------------- pins

    def test_missing_commit_blocks(self):
        self.write(self.consumer, 'pins.json', json.dumps({'repository': 'fed/lib', 'commit': '9' * 40}))
        self.commit(self.consumer, 'bad pin')
        code, report = self.run_crawler()
        self.assertEqual(code, 1)
        self.assertEqual(len(self.blocking(report, 'P1')), 1)

    def test_off_default_commit_blocks_until_archived(self):
        self.write(self.consumer, 'pins.json', json.dumps({'repository': 'fed/lib', 'commit': self.side}))
        self.commit(self.consumer, 'side pin')
        code, report = self.run_crawler()
        self.assertEqual(code, 1)
        self.assertEqual(len(self.blocking(report, 'P2')), 1)
        git(self.lib, 'update-ref', 'refs/remotes/origin/archive/cbz0', self.side)
        code, report = self.run_crawler()
        self.assertEqual(code, 0)
        archived = [f for f in report['findings'] if f.get('status') == 'ARCHIVED']
        self.assertEqual(archived[0]['tags'], ['refs/heads/archive/cbz0'])

    def test_blob_identity_is_checked(self):
        blob = git(self.lib, 'rev-parse', f'{self.l1}:Pkg/Spec.lean')
        self.write(self.consumer, 'pins.json', json.dumps(
            {'repository': 'fed/lib', 'commit': self.l1, 'path': 'Pkg/Spec.lean', 'blob': blob}))
        self.commit(self.consumer, 'blob pin')
        self.assertEqual(self.run_crawler()[0], 0)
        self.write(self.consumer, 'pins.json', json.dumps(
            {'repository': 'fed/lib', 'commit': self.l1, 'path': 'Pkg/Spec.lean', 'blob': 'f' * 40}))
        self.commit(self.consumer, 'wrong blob')
        code, report = self.run_crawler()
        self.assertEqual(code, 1)
        self.assertEqual(self.blocking(report, 'P3')[0]['actual_blob'], blob)

    def test_content_pin_to_a_blob_is_not_a_missing_commit(self):
        blob = git(self.lib, 'rev-parse', f'{self.l1}:Pkg/Spec.lean')
        self.write(self.consumer, 'pins.json', json.dumps({'repository': 'fed/lib', 'sha': blob}))
        self.commit(self.consumer, 'content pin')
        self.assertEqual(self.run_crawler()[0], 0)

    def test_test_data_is_excluded_and_counted(self):
        self.write(self.consumer, 'tests/fixtures/case.json', json.dumps({'repository': 'fed/lib', 'commit': 'a' * 40}))
        self.commit(self.consumer, 'fixture')
        code, report = self.run_crawler()
        self.assertEqual(code, 0)
        self.assertEqual(report['summary']['excluded_test_records'], 1)

    def test_mathlib_skew_blocks(self):
        self.write(self.consumer, 'Other/lake-manifest.json', json.dumps({'packages': [
            {'name': 'mathlib', 'type': 'git', 'url': 'https://github.com/leanprover-community/mathlib4', 'rev': 'e' * 40}]}))
        self.commit(self.consumer, 'other mathlib')
        code, report = self.run_crawler()
        self.assertEqual(code, 1)
        self.assertEqual(self.blocking(report, 'P4')[0]['error'], 'Mathlib revision skew')

    def test_toolchain_skew_blocks(self):
        self.write(self.lib, 'Pkg/lean-toolchain', 'leanprover/lean4:v4.34.1\n')
        self.commit(self.lib, 'toolchain')
        self.write(self.consumer, 'X/lean-toolchain', 'leanprover/lean4:v4.20.0\n')
        self.commit(self.consumer, 'old toolchain')
        code, report = self.run_crawler()
        self.assertEqual(code, 1)
        self.assertEqual(self.blocking(report, 'P4')[0]['error'], 'Lean toolchain skew')

    def test_root_pin_lag_is_reported_and_blocks_only_at_lock(self):
        self.write(self.lib, 'later.txt', 'later\n')
        self.commit(self.lib, 'later')
        code, report = self.run_crawler()
        self.assertEqual(code, 0)
        lag = [f for f in report['findings'] if f['check'] == 'L' and f['to'].endswith('lib.git')]
        self.assertEqual(lag[0]['behind_default_head'], 1)
        code, report = self.run_crawler('--require-current-root-pins')
        self.assertEqual(code, 1)

    def test_gitlink_without_url_blocks(self):
        git(self.root, 'update-index', '--add', '--cacheinfo', f'160000,{self.l1},repos/orphan')
        self.commit_index(self.root, 'orphan gitlink')
        code, report = self.run_crawler()
        self.assertEqual(code, 1)
        self.assertIn('no .gitmodules URL', self.blocking(report, 'P1')[0]['error'])

    # -------------------------------------------------------------- declarations

    def test_older_snapshot_copy_is_not_drift(self):
        self.write(self.consumer, 'inputs/Spec.lean', 'namespace Fed\ntheorem t : 1 = 1 := rfl\nend Fed\n')
        self.commit(self.consumer, 'snapshot copy of L1')
        code, report = self.run_crawler()
        self.assertEqual(code, 0)
        self.assertEqual([f['status'] for f in report['findings'] if f['check'] == 'D1'], ['SNAPSHOT'])

    def test_diverged_copy_is_drift(self):
        self.write(self.consumer, 'inputs/Spec.lean', 'namespace Fed\ntheorem t : 3 = 3 := rfl\nend Fed\n')
        self.commit(self.consumer, 'diverged copy')
        code, report = self.run_crawler()
        self.assertEqual(code, 1)
        self.assertEqual(self.blocking(report, 'D1')[0]['status'], 'DRIFT')

    def test_allowlisted_divergence_needs_the_exact_pair(self):
        self.write(self.consumer, 'inputs/Spec.lean', 'namespace Fed\ntheorem t : 3 = 3 := rfl\nend Fed\n')
        self.commit(self.consumer, 'diverged copy')
        _, report = self.run_crawler()
        pair = self.blocking(report, 'D1')[0]['occurrences']
        row = {'name': 'Fed.t', 'occurrences': pair, 'reason': 'intentional fork kept for a recorded reason'}
        self.allowlist.write_text(json.dumps({'schema': 1, 'allowed': [row]}), encoding='utf-8')
        code, report = self.run_crawler()
        self.assertEqual(code, 0)
        self.assertEqual([f['status'] for f in report['findings'] if f['check'] == 'D1'], ['ALLOWED'])
        row['occurrences'] = [pair[0], pair[1].replace(pair[1][-8:], '0' * 8)]
        self.allowlist.write_text(json.dumps({'schema': 1, 'allowed': [row]}), encoding='utf-8')
        self.assertEqual(self.run_crawler()[0], 1)

    def test_identical_copies_are_not_reported(self):
        self.write(self.consumer, 'inputs/Spec.lean', 'namespace Fed\ntheorem t : 2 = 2 := rfl\nend Fed\n')
        self.commit(self.consumer, 'identical copy of L2')
        code, report = self.run_crawler()
        self.assertEqual(code, 0)
        self.assertEqual([f for f in report['findings'] if f['check'] == 'D1'], [])
        self.assertEqual(report['declarations']['shared_names'], 1)

    def test_bare_names_are_out_of_scope(self):
        self.write(self.consumer, 'X.lean', 'theorem t : 5 = 5 := rfl\n')
        self.write(self.lib, 'Y.lean', 'theorem t : 6 = 6 := rfl\n')
        self.commit(self.consumer, 'bare')
        self.commit(self.lib, 'bare')
        self.assertEqual(self.run_crawler()[0], 0)


if __name__ == '__main__':
    unittest.main()
