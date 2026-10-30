"""Tests for tools/cbz_drift: the federation drift crawler and its Lean declaration lexer.

Every check runs against a synthetic federation of real git repositories built in a
temporary directory, so each finding is produced by the same code path as a live crawl.
`ScannerMutationTests` rewrites the crawler in designated ways and requires a designated
test to fail by assertion (A8); a mutant that only breaks import or syntax does not count.

CBZ_DRIFT_DIR selects the tool directory under test (default: tools/cbz_drift).
"""
import hashlib
import json
import os
import pathlib
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = pathlib.Path(os.environ.get('CBZ_DRIFT_DIR', ROOT / 'tools' / 'cbz_drift'))
sys.path.insert(0, str(TOOL))
import crawl  # noqa: E402
import lean_decls  # noqa: E402

ENV = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t',
           GIT_COMMITTER_EMAIL='t@t', GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_TERMINAL_PROMPT='0')

LEAN_A = textwrap.dedent('''\
    import Mathlib
    /-- doc -/
    namespace Fed
    theorem t (n : Nat) : n + 0 = n := by simp
    def f : Nat := 3
    private def p : Nat := 1
    structure S where
      x : Nat
      y : Nat
    end Fed
    ''')


def sh(cwd, *args, data=None):
    r = subprocess.run(['git', '-C', str(cwd), *args], input=data, capture_output=True, env=ENV)
    if r.returncode:
        raise AssertionError('git %s: %s' % (args, r.stderr.decode()))
    return r.stdout.decode().strip()


class Fed:
    """A synthetic federation: seed `ws`, Lean repos `a` and `b`, registry `reg`."""

    def __init__(self, base):
        self.base = pathlib.Path(base)
        self.urls = {}

    def url(self, name):
        return self.urls.get(name, 'https://github.com/fed/%s.git' % name)

    def repo(self, name):
        d = self.base / name
        if not d.exists():
            d.mkdir()
            sh(d, 'init', '-q', '-b', 'main')
            sh(d, 'remote', 'add', 'origin', 'https://github.com/fed/%s.git' % name)
        return d

    def commit(self, name, files, msg='c', remove=()):
        d = self.repo(name)
        for path, text in files.items():
            p = d / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(text if isinstance(text, bytes) else text.encode())
            sh(d, 'add', path)
        for path in remove:
            sh(d, 'rm', '-q', path)
        sh(d, 'commit', '-q', '--allow-empty', '-m', msg)
        sha = sh(d, 'rev-parse', 'HEAD')
        sh(d, 'update-ref', 'refs/remotes/origin/main', sha)
        return sha

    def build(self, a_lean=LEAN_A, b_lean=LEAN_A):
        self.pins = {}
        self.pins['a'] = self.commit('a', {'lakefile.toml': 'name = "a"\n', 'lean-toolchain': 'v4.0.0\n',
                                           'Fed/A.lean': a_lean, 'data/x.txt': 'payload\n'})
        self.pins['b'] = self.commit('b', {'lakefile.toml': 'name = "b"\n', 'lean-toolchain': 'v4.0.0\n',
                                           'Copy/A.lean': b_lean})
        payload = b'payload\n'
        self.registry = {
            'repositories': {'A': {'full_name': 'fed/a'}},
            'artifacts': [{'key': 'x', 'repository': 'A', 'commit': self.pins['a'], 'path': 'data/x.txt',
                           'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}],
        }
        self.pins['reg'] = self.commit('reg', {'registry.json': json.dumps(self.registry)})
        self.members = ['a', 'b', 'reg']
        return self.write_seed()

    def write_seed(self, pins=None, gitlinks=None, listed=None, modules=None, extra=None, update=None):
        pins = dict(self.pins if pins is None else pins)
        gitlinks = dict(pins if gitlinks is None else gitlinks)
        names = sorted(self.members)
        modules = names if modules is None else modules
        listed = names if listed is None else listed
        gm = ''.join('[submodule "repos/%s"]\n\tpath = repos/%s\n\turl = %s\n%s' % (
            n, n, self.url(n), '\tupdate = none\n' if update == n else '') for n in modules)
        pj = {'schema': 'ulw.pins.v1', 'repos': [
            {'name': n, 'path': 'repos/%s' % n, 'url': self.url(n),
             'pinned_sha': pins[n], 'opt_in': update == n} for n in names if n in pins]}
        reg_blob = sh(self.repo('reg'), 'rev-parse', '%s:registry.json' % pins.get('reg', self.pins['reg']))
        wsj = {'repositories': [{'full_name': 'fed/ws'}] + [{'full_name': 'fed/%s' % n} for n in listed],
               'canonical_inputs': {'repository_registry': {'repository': 'fed/reg', 'path': 'registry.json',
                                                            'observed_blob': reg_blob}}}
        files = {'.gitmodules': gm, 'pins.json': json.dumps(pj), 'workspace/repositories.json': json.dumps(wsj)}
        files.update(extra or {})
        d = self.repo('ws')
        for path, text in files.items():
            p = d / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            sh(d, 'add', path)
        tracked = sh(d, 'ls-files', '-s', 'repos').splitlines()
        for line in tracked:
            sh(d, 'rm', '-q', '--cached', line.split('\t')[1])
        for n, sha in gitlinks.items():
            if n in modules:
                sh(d, 'update-index', '--add', '--cacheinfo', '160000,%s,repos/%s' % (sha, n))
        sh(d, 'commit', '-q', '--allow-empty', '-m', 'seed')
        sha = sh(d, 'rev-parse', 'HEAD')
        sh(d, 'update-ref', 'refs/remotes/origin/main', sha)
        return sha

    def crawl(self, **kw):
        return crawl.run('fed/ws', [str(self.base)], **kw)


def codes(rep, level=None):
    return sorted(f['code'] for f in rep['findings'] if level is None or f['level'] == level)


class FedCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fed = Fed(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()


class CleanFederationTests(FedCase):
    def test_clean_federation_has_no_block_and_is_not_vacuous(self):
        self.fed.build()
        rep = self.fed.crawl()
        self.assertEqual(codes(rep, 'BLOCK'), [])
        self.assertEqual(rep['blocking'], 0)
        self.assertEqual({r['repository'] for r in rep['repositories'] if r['resolved']},
                         {'fed/ws', 'fed/a', 'fed/b', 'fed/reg'})
        self.assertEqual(rep['stats']['registry_artifacts_verified'], 1)
        self.assertEqual(rep['stats']['lean_shared_names'], 3)   # Fed.t, Fed.f, Fed.S (private p excluded)
        self.assertEqual(rep['stats']['lean_files'], 2)

    def test_cli_exit_status(self):
        self.fed.build()
        out = os.path.join(self.tmp.name, 'r.json')
        r = subprocess.run([sys.executable, '-B', str(TOOL / 'crawl.py'), '--seed', 'fed/ws', '--search', self.tmp.name,
                            '--json', out], capture_output=True, env=ENV)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(out) as fh:
            self.assertEqual(json.load(fh)['blocking'], 0)
        self.fed.commit('b', {'Copy/A.lean': LEAN_A.replace('n + 0 = n', '0 + n = n')})
        self.fed.pins['b'] = sh(self.fed.repo('b'), 'rev-parse', 'HEAD')
        self.fed.write_seed()
        r = subprocess.run([sys.executable, '-B', str(TOOL / 'crawl.py'), '--seed', 'fed/ws', '--search', self.tmp.name],
                           capture_output=True, env=ENV)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn(b'BLOCK LEAN-DRIFT', r.stdout)
        r = subprocess.run([sys.executable, '-B', str(TOOL / 'crawl.py'), '--seed', 'fed/nope', '--search', self.tmp.name],
                           capture_output=True, env=ENV)
        self.assertEqual(r.returncode, 2)


class LeanDriftTests(FedCase):
    def drift(self, b_lean):
        self.fed.build(b_lean=b_lean)
        return self.fed.crawl()

    def test_changed_theorem_statement_blocks(self):
        rep = self.drift(LEAN_A.replace('n + 0 = n', '0 + n = n'))
        self.assertEqual(codes(rep, 'BLOCK'), ['LEAN-DRIFT'])
        f = [f for f in rep['findings'] if f['code'] == 'LEAN-DRIFT'][0]
        self.assertEqual(f['name'], 'Fed.t')
        self.assertEqual(len(f['variants']), 2)

    def test_changed_definition_value_blocks(self):
        rep = self.drift(LEAN_A.replace('def f : Nat := 3', 'def f : Nat := 4'))
        self.assertEqual([f['name'] for f in rep['findings'] if f['code'] == 'LEAN-DRIFT'], ['Fed.f'])

    def test_changed_structure_field_blocks(self):
        rep = self.drift(LEAN_A.replace('  y : Nat', '  y : Int'))
        self.assertEqual([f['name'] for f in rep['findings'] if f['code'] == 'LEAN-DRIFT'], ['Fed.S'])

    def test_changed_kind_blocks(self):
        rep = self.drift(LEAN_A.replace('def f : Nat := 3', 'abbrev f : Nat := 3'))
        self.assertEqual([f['name'] for f in rep['findings'] if f['code'] == 'LEAN-DRIFT'], ['Fed.f'])

    def test_proof_comment_and_layout_changes_do_not_block(self):
        b = LEAN_A.replace(':= by simp', ':= by\n  omega -- different proof').replace(
            'theorem t (n : Nat)', 'theorem t -- the statement\n    (n : Nat) /- note -/')
        rep = self.drift(b)
        self.assertEqual(codes(rep, 'BLOCK'), [])

    def test_private_names_are_not_compared(self):
        rep = self.drift(LEAN_A.replace('private def p : Nat := 1', 'private def p : Nat := 2'))
        self.assertEqual(codes(rep, 'BLOCK'), [])
        self.assertEqual(rep['stats']['lean_shared_names'], 3)

    def test_differently_namespaced_names_are_not_compared(self):
        rep = self.drift(LEAN_A.replace('namespace Fed', 'namespace Other').replace('end Fed', 'end Other')
                         .replace('def f : Nat := 3', 'def f : Nat := 9'))
        self.assertEqual(codes(rep, 'BLOCK'), [])
        self.assertEqual(rep['stats']['lean_shared_names'], 0)

    def test_lean_syntax_error_blocks(self):
        rep = self.drift(LEAN_A + '\n/- never closed\n')
        self.assertIn('LEAN-SYNTAX', codes(rep, 'BLOCK'))

    def test_override_tests_a_candidate_head(self):
        self.fed.build()
        cand = self.fed.commit('b', {'Copy/A.lean': LEAN_A.replace('def f : Nat := 3', 'def f : Nat := 5')})
        sh(self.fed.repo('b'), 'update-ref', 'refs/remotes/origin/main', self.fed.pins['b'])
        self.assertEqual(codes(self.fed.crawl(), 'BLOCK'), [])
        rep = self.fed.crawl(overrides={'fed/b': cand})
        self.assertEqual(codes(rep, 'BLOCK'), ['LEAN-DRIFT'])

    def test_tips_mode_reads_default_tips(self):
        self.fed.build()
        self.fed.commit('b', {'Copy/A.lean': LEAN_A.replace('def f : Nat := 3', 'def f : Nat := 5')})
        self.assertEqual(codes(self.fed.crawl(), 'BLOCK'), [])
        self.assertEqual(codes(self.fed.crawl(at='tips'), 'BLOCK'), ['LEAN-DRIFT'])


class PinTests(FedCase):
    def test_gitlink_disagreeing_with_pins_blocks(self):
        self.fed.build()
        other = self.fed.commit('b', {'z': 'z'})
        self.fed.write_seed(gitlinks=dict(self.fed.pins, b=other))
        self.assertIn('PIN-SOURCES', codes(self.fed.crawl(), 'BLOCK'))

    def test_submodule_without_pin_blocks(self):
        self.fed.build()
        self.fed.write_seed(pins={k: v for k, v in self.fed.pins.items() if k != 'b'}, gitlinks=self.fed.pins)
        self.assertIn('PIN-SOURCES', codes(self.fed.crawl(), 'BLOCK'))

    def test_pinned_but_unlisted_blocks(self):
        self.fed.build()
        self.fed.write_seed(listed=['a', 'reg'])
        self.assertIn('PIN-SOURCES', codes(self.fed.crawl(), 'BLOCK'))

    def test_gitlink_without_gitmodules_entry_blocks(self):
        self.fed.build()
        d = self.fed.repo('ws')
        sh(d, 'update-index', '--add', '--cacheinfo', '160000,%s,vendor/stray' % self.fed.pins['b'])
        sh(d, 'commit', '-q', '-m', 'stray')
        sh(d, 'update-ref', 'refs/remotes/origin/main', sh(d, 'rev-parse', 'HEAD'))
        rep = self.fed.crawl()
        self.assertEqual([f['message'] for f in rep['findings'] if f['level'] == 'BLOCK'],
                         ['gitlink vendor/stray has no .gitmodules entry'])

    def test_missing_pin_blocks(self):
        self.fed.build()
        self.fed.write_seed(pins=dict(self.fed.pins, b='0' * 39 + '1'))
        self.assertIn('PIN-MISSING', codes(self.fed.crawl(), 'BLOCK'))

    def test_pin_off_the_default_line_blocks(self):
        self.fed.build()
        d = self.fed.repo('b')
        sh(d, 'checkout', '-q', '-b', 'side')
        side = self.fed.commit('b', {'s': 's'})
        sh(d, 'checkout', '-q', 'main')
        sh(d, 'update-ref', 'refs/remotes/origin/main', self.fed.pins['b'])
        self.fed.write_seed(pins=dict(self.fed.pins, b=side))
        self.assertIn('PIN-OFFLINE', codes(self.fed.crawl(), 'BLOCK'))

    def test_stale_pin_is_info_only(self):
        self.fed.build()
        self.fed.commit('b', {'later': 'x'})
        rep = self.fed.crawl()
        self.assertEqual(codes(rep, 'BLOCK'), [])
        self.assertEqual(codes(rep, 'INFO'), ['PIN-STALE'])

    def test_shallow_clone_cannot_decide_ancestry(self):
        self.fed.build()
        self.fed.commit('b', {'later': 'x'})
        src = self.fed.repo('b')
        shutil.move(str(src), str(self.fed.base / 'b-full'))
        subprocess.run(['git', 'clone', '-q', '--depth', '1', '--no-local', 'file://%s' % (self.fed.base / 'b-full'),
                        str(src)], check=True, env=ENV, capture_output=True)
        sh(src, 'remote', 'set-url', 'origin', 'https://github.com/fed/b.git')
        sh(src, 'fetch', '-q', '--depth', '1', 'file://%s' % (self.fed.base / 'b-full'), self.fed.pins['b'])
        sh(src, 'update-ref', 'refs/remotes/origin/main', sh(src, 'rev-parse', 'HEAD'))
        shutil.move(str(self.fed.base / 'b-full'), self.tmp.name + '-unused')
        try:
            msgs = [f['message'] for f in self.fed.crawl()['findings'] if f['code'] == 'PIN-OFFLINE']
            # a shallow clone cannot show ancestry; saying "not an ancestor" would be untrue (A7)
            self.assertEqual(len(msgs), 1)
            self.assertIn('shallow clone', msgs[0])
        finally:
            shutil.rmtree(self.tmp.name + '-unused')

    def test_unresolved_repository_blocks_unless_opt_in(self):
        self.fed.build()
        shutil.move(str(self.fed.repo('b')), self.tmp.name + '-moved')
        try:
            self.assertIn('UNRESOLVED', codes(self.fed.crawl(), 'BLOCK'))
            # the seed itself still has b's gitlink; mark b opt-in
            shutil.move(self.tmp.name + '-moved', str(self.fed.base / 'b'))
            self.fed.write_seed(update='b')
            shutil.move(str(self.fed.base / 'b'), self.tmp.name + '-moved')
            rep = self.fed.crawl()
            self.assertNotIn('UNRESOLVED', codes(rep, 'BLOCK'))
            self.assertIn('UNRESOLVED', codes(rep, 'INFO'))
        finally:
            if os.path.exists(self.tmp.name + '-moved'):
                shutil.rmtree(self.tmp.name + '-moved')

    def test_malformed_pins_json_blocks(self):
        self.fed.build()
        self.fed.write_seed(extra={'pins.json': '{"repos": ['})
        self.assertIn('MALFORMED', codes(self.fed.crawl(), 'BLOCK'))


class RegistryTests(FedCase):
    def rebuild(self, **change):
        self.fed.build()
        self.fed.registry['artifacts'][0].update(change)
        self.fed.pins['reg'] = self.fed.commit('reg', {'registry.json': json.dumps(self.fed.registry)})
        self.fed.write_seed()
        return self.fed.crawl()

    def test_wrong_digest_blocks(self):
        self.assertIn('REGISTRY-BYTES', codes(self.rebuild(sha256='0' * 64), 'BLOCK'))

    def test_wrong_size_blocks(self):
        self.assertIn('REGISTRY-BYTES', codes(self.rebuild(bytes=9), 'BLOCK'))

    def test_absent_path_blocks(self):
        self.assertIn('REGISTRY-BYTES', codes(self.rebuild(path='data/none.txt'), 'BLOCK'))

    def test_missing_commit_blocks(self):
        self.assertIn('PIN-MISSING', codes(self.rebuild(commit='1' * 40), 'BLOCK'))

    def test_evolved_artifact_is_info_only(self):
        self.fed.build()
        self.fed.pins['a'] = self.fed.commit('a', {'data/x.txt': 'payload v2\n'})
        self.fed.write_seed()
        rep = self.fed.crawl()
        self.assertEqual(codes(rep, 'BLOCK'), [])
        self.assertIn('REGISTRY-EVOLVED', codes(rep, 'INFO'))

    def test_observed_input_blob_difference_is_info(self):
        self.fed.build()
        self.fed.registry['note'] = 'changed'
        self.fed.commit('reg', {'registry.json': json.dumps(self.fed.registry)})
        rep = self.fed.crawl(at='tips')
        self.assertIn('INPUT-OBSERVED', codes(rep, 'INFO'))
        self.assertEqual(codes(rep, 'BLOCK'), [])


class TopologyTests(FedCase):
    def test_added_repository_is_crawled_without_code_change(self):
        self.fed.build()
        self.fed.pins['c'] = self.fed.commit('c', {'lakefile.toml': 'name = "c"\n', 'lean-toolchain': 'v4.0.0\n',
                                                   'C.lean': LEAN_A.replace('def f : Nat := 3', 'def f : Nat := 7')})
        self.fed.members.append('c')
        self.fed.write_seed()
        rep = self.fed.crawl()
        self.assertIn('fed/c', {r['repository'] for r in rep['repositories'] if r['resolved']})
        self.assertEqual([f['name'] for f in rep['findings'] if f['code'] == 'LEAN-DRIFT'], ['Fed.f'])

    def test_removed_repository_is_not_crawled(self):
        self.fed.build()
        self.fed.members.remove('b')
        del self.fed.pins['b']
        self.fed.write_seed()
        rep = self.fed.crawl()
        self.assertNotIn('fed/b', {r['repository'] for r in rep['repositories']})
        self.assertEqual(codes(rep, 'BLOCK'), [])

    def test_registry_artifact_repositories_join_the_crawl(self):
        self.fed.build()
        self.fed.members.remove('a')
        del self.fed.pins['a']
        self.fed.write_seed()
        rep = self.fed.crawl()
        self.assertIn('fed/a', {r['repository'] for r in rep['repositories'] if r['resolved']})
        self.assertEqual(rep['stats'].get('registry_artifacts_verified', 0), 1)

    def test_lake_dependency_on_a_crawled_repository_is_followed(self):
        # A present historical revision is INFO. (This fixture used an absent revision until
        # R2 at #712/6511043716 showed that an absent one must block; see the next test.)
        self.fed.build()
        old = self.fed.pins['a']
        self.fed.pins['a'] = self.fed.commit('a', {'later': 'x'})
        manifest = {'packages': [{'type': 'git', 'url': 'https://github.com/fed/a', 'rev': old, 'name': 'a'}]}
        self.fed.pins['b'] = self.fed.commit('b', {'lake-manifest.json': json.dumps(manifest)})
        self.fed.write_seed()
        rep = self.fed.crawl()
        self.assertIn('LAKE-DEP', codes(rep, 'INFO'))
        self.assertEqual(codes(rep, 'BLOCK'), [])
        self.assertIn({'from': 'fed/b', 'to': 'fed/a', 'via': 'lake'}, rep['edges'])

    def add_lake_edge(self, holder, name, url, rev):
        manifest = {'packages': [{'type': 'git', 'url': url, 'rev': rev, 'name': name}]}
        self.fed.pins[holder] = self.fed.commit(holder, {'lake-manifest.json': json.dumps(manifest)})
        self.fed.write_seed()

    def lean_c(self):
        return self.fed.commit('c', {'lakefile.toml': 'name = "c"\n', 'lean-toolchain': 'v4.0.0\n',
                                     'C.lean': LEAN_A.replace('def f : Nat := 3', 'def f : Nat := 7')})

    def test_missing_lake_revision_blocks(self):
        self.fed.build()
        self.add_lake_edge('b', 'a', 'https://github.com/fed/a.git', '2' * 40)
        self.assertEqual(codes(self.fed.crawl(), 'BLOCK'), ['PIN-MISSING'])

    def test_lake_dependency_reached_only_by_lake_is_scanned(self):
        self.fed.build()
        self.add_lake_edge('b', 'c', 'https://github.com/fed/c.git', self.lean_c())
        rep = self.fed.crawl()
        self.assertIn('fed/c', {r['repository'] for r in rep['repositories'] if r['resolved']})
        self.assertEqual([f['name'] for f in rep['findings'] if f['code'] == 'LEAN-DRIFT'], ['Fed.f'])

    def test_unheld_lake_dependency_is_cloned(self):
        self.fed.build()
        sha = self.lean_c()
        remote = pathlib.Path(self.tmp.name + '-lake') / 'fed' / 'c'
        remote.parent.mkdir(parents=True)
        shutil.move(str(self.fed.repo('c')), str(remote))
        try:
            self.add_lake_edge('b', 'c', remote.as_uri(), sha)
            clones = pathlib.Path(self.tmp.name + '-lake') / 'clones'
            clones.mkdir()
            rep = self.fed.crawl(clone_into=str(clones))
            self.assertIn('fed/c', {r['repository'] for r in rep['repositories'] if r['resolved']})
            self.assertEqual([f['name'] for f in rep['findings'] if f['code'] == 'LEAN-DRIFT'], ['Fed.f'])
        finally:
            shutil.rmtree(self.tmp.name + '-lake')

    def test_external_lake_dependency_is_not_crawled(self):
        self.fed.build()
        self.add_lake_edge('b', 'mathlib', 'https://github.com/leanprover-community/mathlib4', '3' * 40)
        rep = self.fed.crawl()
        self.assertEqual(codes(rep, 'BLOCK'), [])
        self.assertEqual(rep['stats']['lake_external_edges'], 1)

    def test_child_records_extend_the_graph(self):
        self.fed.build()
        sha = self.lean_c()
        child = self.fed.repo('a')
        sh(child, 'update-index', '--add', '--cacheinfo', '160000,%s,repos/c' % sha)
        self.fed.pins['a'] = self.fed.commit('a', {
            '.gitmodules': '[submodule "c"]\npath = repos/c\nurl = https://github.com/fed/c.git\n',
            'pins.json': json.dumps({'repos': [{'url': 'https://github.com/fed/c.git', 'path': 'repos/c', 'pinned_sha': sha}]}),
            'workspace/repositories.json': json.dumps({'repositories': [{'full_name': 'fed/a'}, {'full_name': 'fed/c'}]}),
        })
        self.fed.write_seed()
        rep = self.fed.crawl()
        self.assertIn({'from': 'fed/a', 'to': 'fed/c', 'via': 'gitmodules+pins+listed'}, rep['edges'])
        self.assertEqual([f['name'] for f in rep['findings'] if f['code'] == 'LEAN-DRIFT'], ['Fed.f'])

    def test_registry_only_repository_is_expanded(self):
        # fed/a is reached only through a registry artifact; its own lake edge must still be followed
        self.fed.build()
        sha = self.lean_c()
        manifest = {'packages': [{'type': 'git', 'url': 'https://github.com/fed/c', 'rev': sha, 'name': 'c'}]}
        self.fed.commit('a', {'lake-manifest.json': json.dumps(manifest)})
        self.fed.members.remove('a')
        del self.fed.pins['a']
        self.fed.write_seed()
        rep = self.fed.crawl()
        self.assertIn({'from': 'fed/a', 'to': 'fed/c', 'via': 'lake'}, rep['edges'])
        self.assertIn('fed/c', {r['repository'] for r in rep['repositories'] if r['resolved']})

    def test_fetch_refreshes_the_default_tip(self):
        self.fed.build()
        with tempfile.TemporaryDirectory() as base:
            cache = pathlib.Path(base) / 'cached'
            subprocess.run(['git', 'clone', '-q', self.fed.repo('b').as_uri(), str(cache)], check=True, env=ENV,
                           capture_output=True)
            advanced = self.fed.commit('b', {'tip': 'new\n'})
            self.assertNotEqual(crawl.Repo('fed/b', cache).default_tip(), advanced)
            self.assertEqual(crawl.Repo('fed/b', cache, fetch=True).default_tip(), advanced)

    def test_failed_refresh_blocks(self):
        self.fed.build()
        sh(self.fed.repo('b'), 'remote', 'set-url', 'origin', 'file://%s/nowhere/fed/b' % self.tmp.name)
        rep = self.fed.crawl(fetch=True)
        self.assertIn('FETCH-FAILED', codes(rep, 'BLOCK'))

    def test_unheld_repository_is_cloned_from_its_recorded_url(self):
        self.fed.build()
        remote = pathlib.Path(self.tmp.name + '-remote') / 'fed' / 'b'
        remote.parent.mkdir(parents=True)
        shutil.move(str(self.fed.repo('b')), str(remote))
        try:
            self.fed.urls['b'] = 'file://%s' % remote
            self.fed.write_seed()
            self.assertIn('UNRESOLVED', codes(self.fed.crawl(), 'BLOCK'))
            clones = pathlib.Path(self.tmp.name + '-remote') / 'clones'
            clones.mkdir()
            rep = self.fed.crawl(clone_into=str(clones))
            self.assertEqual(codes(rep, 'BLOCK'), [])
            self.assertEqual(rep['stats']['lean_shared_names'], 3)
            self.assertTrue((clones / 'fed__b').is_dir())
        finally:
            shutil.rmtree(self.tmp.name + '-remote')

    def test_identity_normalization(self):
        for u in ('https://github.com/Fed/A.git', 'git@github.com:fed/a.git', 'fed/a',
                  'http://proxy@127.0.0.1:9/git/fed/a', 'https://github.com/fed/a/'):
            self.assertEqual(crawl.repo_key(u), 'fed/a')


class LeanDeclTests(unittest.TestCase):
    def names(self, text):
        return [(d['name'], d['kind']) for d in lean_decls.declarations(text)]

    def test_namespaces_sections_and_root(self):
        text = 'namespace A.B\nsection S\ntheorem x : True := trivial\nend S\ndef _root_.y : Nat := 1\nend A.B\ndef z := 2\n'
        self.assertEqual(self.names(text), [('A.B.x', 'theorem'), ('y', 'def'), ('z', 'def')])

    def test_unbalanced_scopes_raise(self):
        for text in ('namespace A\nend B\n', 'end A\n', 'section\nend X\n'):
            with self.subTest(text=text), self.assertRaises(lean_decls.LeanSyntaxError):
                lean_decls.declarations(text)

    def test_header_stops_at_top_level_assign_and_line_leading_bar(self):
        ds = lean_decls.declarations('theorem a (h : |x| ≤ 1 := 2) : |x| ≤ 1 := by simp\n'
                                     'def g : Nat → Nat\n  | 0 => 1\n  | n+1 => n\n')
        self.assertEqual(ds[0]['header'], '( h : | x | ≤ 1 := 2 ) : | x | ≤ 1')
        self.assertEqual(ds[1]['header'], ': Nat → Nat')
        self.assertIn('| 0 = > 1', ds[1]['interface'])

    def test_theorem_interface_excludes_proof_definition_includes_value(self):
        t, d = lean_decls.declarations('theorem t : 1 = 1 := rfl\ndef d : Nat := 7\n')
        self.assertEqual(t['interface'], ': 1 = 1')
        self.assertEqual(d['interface'], ': Nat := 7')

    def test_comments_and_strings(self):
        ds = lean_decls.declarations('-- theorem fake : False\n/- def fake2 := 1 -/\ndef s := "theorem fake3 : False"\n')
        self.assertEqual([d['name'] for d in ds], ['s'])

    def test_modifiers_attributes_and_anonymous_instance(self):
        ds = lean_decls.declarations('@[simp] protected noncomputable def a := 1\ninstance : Inhabited Nat := ⟨0⟩\n'
                                     'instance (priority := low) named : Inhabited Int := ⟨0⟩\nprivate theorem p : True := trivial\n')
        self.assertEqual([(d['name'], d['private'], d['modifiers']) for d in ds],
                         [('a', False, ['noncomputable', 'protected']), ('named', False, []), ('p', True, [])])

    def test_malformed_raise(self):
        for text in ('/- open', 'def s := "open', 'def «open := 1', 'theorem : True := trivial', 'namespace\n'):
            with self.subTest(text=text), self.assertRaises(lean_decls.LeanSyntaxError):
                lean_decls.declarations(text)


# Each mutant: (label, exact source substring, replacement, file, designated test).
MUTANTS = [
    ('theorem statement ignored', "'interface': header if dkind in STATEMENT_ONLY else", "'interface': '' if dkind in STATEMENT_ONLY else",
     'lean_decls.py', 'LeanDriftTests.test_changed_theorem_statement_blocks'),
    ('definition value ignored', "else ' '.join(hdr + body)", "else header", 'lean_decls.py',
     'LeanDriftTests.test_changed_definition_value_blocks'),
    ('kind ignored', "variants.setdefault((l['kind'], l['interface']), [])", "variants.setdefault(('', l['interface']), [])",
     'crawl.py', 'LeanDriftTests.test_changed_kind_blocks'),
    ('drift downgraded', "self.add(BLOCK, 'LEAN-DRIFT'", "self.add(INFO, 'LEAN-DRIFT'", 'crawl.py',
     'LeanDriftTests.test_changed_theorem_statement_blocks'),
    ('private compared', "                    if d['private']:\n                        continue\n", "", 'crawl.py',
     'LeanDriftTests.test_private_names_are_not_compared'),
    ('namespace dropped', "'.'.join(current_ns() + [name])", "name", 'lean_decls.py',
     'LeanDriftTests.test_differently_namespaced_names_are_not_compared'),
    ('comments kept', "            fresh = first\n            continue\n        if text.startswith('/-', i):",
     "            yield 'sym', text[i:i + 2], start_line, first\n            fresh = first\n            continue\n        if text.startswith('/-', i):",
     'lean_decls.py', 'LeanDriftTests.test_proof_comment_and_layout_changes_do_not_block'),
    ('lean syntax swallowed', "self.add(BLOCK, 'LEAN-SYNTAX', str(exc)", "self.add(INFO, 'LEAN-SYNTAX', str(exc)", 'crawl.py',
     'LeanDriftTests.test_lean_syntax_error_blocks'),
    ('unbalanced end accepted', "            if sparts != parts:\n", "            if False:\n", 'lean_decls.py',
     'LeanDeclTests.test_unbalanced_scopes_raise'),
    ('bar ends every header', "(k == 'sym' and t == '|' and fst)", "(k == 'sym' and t == '|')", 'lean_decls.py',
     'LeanDeclTests.test_header_stops_at_top_level_assign_and_line_leading_bar'),
    ('depth ignored', "                if depth == 0:\n                    if k == 'ident' and t in COMMANDS and not",
     "                if True:\n                    if k == 'ident' and t in COMMANDS and not", 'lean_decls.py',
     'LeanDeclTests.test_header_stops_at_top_level_assign_and_line_leading_bar'),
    ('override ignored', "        if key in self.overrides:\n", "        if False:\n", 'crawl.py',
     'LeanDriftTests.test_override_tests_a_candidate_head'),
    ('tips mode ignored', "elif self.at == 'pins' and key in self.pins", "elif key in self.pins", 'crawl.py',
     'LeanDriftTests.test_tips_mode_reads_default_tips'),
    ('gitlink unchecked', "if 'gitlink' in rec and 'pins' in rec and rec['gitlink'] != rec['pins']:",
     "if False:", 'crawl.py', 'PinTests.test_gitlink_disagreeing_with_pins_blocks'),
    ('gitmodules/pins membership unchecked', "if gm is not None and pins is not None and ('gitmodules' in have) != ('pins' in have):",
     "if False:", 'crawl.py', 'PinTests.test_submodule_without_pin_blocks'),
    ('listing unchecked', "if reg is not None and (have & {'gitmodules', 'pins'}) and 'listed' not in have:", "if False:",
     'crawl.py', 'PinTests.test_pinned_but_unlisted_blocks'),
    ('stray gitlink unchecked', "            if typ == 'commit' and not any(", "            if False and not any(", 'crawl.py',
     'PinTests.test_gitlink_without_gitmodules_entry_blocks'),
    ('missing pin passes', "self.add(BLOCK, 'PIN-MISSING', 'pinned commit", "self.add(INFO, 'PIN-MISSING', 'pinned commit", 'crawl.py',
     'PinTests.test_missing_pin_blocks'),
    ('ancestry unchecked', "            elif not repo.is_ancestor(pin, tip):", "            elif False:", 'crawl.py',
     'PinTests.test_pin_off_the_default_line_blocks'),
    ('stale pin blocks', "self.add(INFO, 'PIN-STALE'", "self.add(BLOCK, 'PIN-STALE'", 'crawl.py', 'PinTests.test_stale_pin_is_info_only'),
    ('shallow trusted', "            if repo.shallow:\n", "            if False:\n", 'crawl.py',
     'PinTests.test_shallow_clone_cannot_decide_ancestry'),
    ('unresolved passes', "self.add(INFO if key in self.optional else BLOCK, 'UNRESOLVED'", "self.add(INFO, 'UNRESOLVED'", 'crawl.py',
     'PinTests.test_unresolved_repository_blocks_unless_opt_in'),
    ('opt-in ignored', "self.add(INFO if key in self.optional else BLOCK, 'UNRESOLVED'", "self.add(BLOCK, 'UNRESOLVED'", 'crawl.py',
     'PinTests.test_unresolved_repository_blocks_unless_opt_in'),
    ('malformed json passes', "self.add(BLOCK, 'MALFORMED', '%s is not valid JSON", "self.add(INFO, 'MALFORMED', '%s is not valid JSON", 'crawl.py',
     'PinTests.test_malformed_pins_json_blocks'),
    ('digest unchecked', "if len(data) != size or hashlib.sha256(data).hexdigest() != digest:", "if len(data) != size:", 'crawl.py',
     'RegistryTests.test_wrong_digest_blocks'),
    ('size unchecked', "if len(data) != size or hashlib.sha256(data).hexdigest() != digest:", "if hashlib.sha256(data).hexdigest() != digest:", 'crawl.py',
     'RegistryTests.test_wrong_size_blocks'),
    ('absent artifact passes', "            if data is None:\n                self.add(BLOCK, 'REGISTRY-BYTES'", "            if data is None:\n                self.add(INFO, 'REGISTRY-BYTES'", 'crawl.py',
     'RegistryTests.test_absent_path_blocks'),
    ('artifact commit unchecked', "self.add(BLOCK, 'PIN-MISSING', 'artifact %s", "self.add(INFO, 'PIN-MISSING', 'artifact %s", 'crawl.py',
     'RegistryTests.test_missing_commit_blocks'),
    ('evolution blocks', "self.add(INFO, 'REGISTRY-EVOLVED'", "self.add(BLOCK, 'REGISTRY-EVOLVED'", 'crawl.py',
     'RegistryTests.test_evolved_artifact_is_info_only'),
    ('observed blob unchecked', "if spec.get('observed_blob') and (ent is None or ent[2] != spec['observed_blob']):", "if False:", 'crawl.py',
     'RegistryTests.test_observed_input_blob_difference_is_info'),
    ('registry not followed', "                self.check_artifacts(repo, ref, spec['path'], doc)", "                pass", 'crawl.py',
     'TopologyTests.test_registry_artifact_repositories_join_the_crawl'),
    ('lake records dropped', "                self.lake_records.append((key, ref, lm, pkg, dep))\n", "", 'crawl.py',
     'TopologyTests.test_lake_dependency_on_a_crawled_repository_is_followed'),
    ('clone ignored', "        if gitdir is None and self.clone_into:\n", "        if False:\n", 'crawl.py',
     'TopologyTests.test_unheld_repository_is_cloned_from_its_recorded_url'),
    ('lake revision unchecked', "            if not drepo.has_commit(pkg['rev']):", "            if False:", 'crawl.py',
     'TopologyTests.test_missing_lake_revision_blocks'),
    ('lake targets not queued', "                out.append(dep)\n", "", 'crawl.py',
     'TopologyTests.test_lake_dependency_reached_only_by_lake_is_scanned'),
    ('lake url not recorded', "                self.urls.setdefault(dep, pkg['url'])\n", "", 'crawl.py',
     'TopologyTests.test_unheld_lake_dependency_is_cloned'),
    ('federation boundary ignored', "                if dep.split('/')[0] not in self.owners:", "                if False:", 'crawl.py',
     'TopologyTests.test_external_lake_dependency_is_not_crawled'),
    ('children not expanded', "            for new in self.expand(key, repo):", "            for new in (self.expand(key, repo) if key == self.seed_key else []):", 'crawl.py',
     'TopologyTests.test_child_records_extend_the_graph'),
    ('registry repositories not queued', "                            out.append(repo_key(ent['full_name']))", "                            pass", 'crawl.py',
     'TopologyTests.test_registry_only_repository_is_expanded'),
    ('no refresh', "            r = git(gitdir, *args, check=False)", "            r = git(gitdir, 'status', check=False)", 'crawl.py',
     'TopologyTests.test_fetch_refreshes_the_default_tip'),
    ('failed refresh passes', "        if self.repos[key].fetch_failed:", "        if False:", 'crawl.py',
     'TopologyTests.test_failed_refresh_blocks'),
    ('case-sensitive identity', "        return ('%s/%s' % (m.group(1), m.group(2))).lower()", "        return '%s/%s' % (m.group(1), m.group(2))", 'crawl.py',
     'TopologyTests.test_identity_normalization'),
    ('exit ignores blocks', "    return 1 if rep['blocking'] else 0", "    return 0", 'crawl.py', 'CleanFederationTests.test_cli_exit_status'),
]


class ScannerMutationTests(unittest.TestCase):
    """A8: each mutant must make its designated test fail by assertion, nothing else."""

    def run_designated(self, tool_dir, test):
        env = dict(ENV, CBZ_DRIFT_DIR=str(tool_dir), PYTHONDONTWRITEBYTECODE='1')
        mode = ['-O'] if sys.flags.optimize else []      # children run in the parent's interpreter mode
        return subprocess.run([sys.executable, '-B', *mode, '-m', 'unittest', 'test_cbz_drift.%s' % test],
                              cwd=str(pathlib.Path(__file__).parent), capture_output=True, env=env, timeout=600)

    def copy_tool(self, base):
        dst = pathlib.Path(base) / 'cbz_drift'
        shutil.copytree(TOOL, dst, ignore=shutil.ignore_patterns('__pycache__'))
        return dst

    def test_harness_control_every_designated_test_passes_unmutated(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = self.copy_tool(tmp)
            for test in sorted({m[4] for m in MUTANTS}):
                r = self.run_designated(dst, test)
                self.assertEqual(r.returncode, 0, '%s: %s' % (test, r.stderr.decode()[-2000:]))
                self.assertRegex(r.stderr.decode(), r'Ran 1 test')

    def test_every_mutant_is_killed_by_its_designated_assertion(self):
        survivors = []
        for label, old, new, fname, test in MUTANTS:
            with self.subTest(mutant=label), tempfile.TemporaryDirectory() as tmp:
                dst = self.copy_tool(tmp)
                src = (dst / fname).read_text()
                self.assertEqual(src.count(old), 1, 'mutant %r does not apply exactly once' % label)
                (dst / fname).write_text(src.replace(old, new))
                py_compile.compile(str(dst / fname), doraise=True)   # a syntax error cannot count as a kill
                r = self.run_designated(dst, test)
                err = r.stderr.decode()
                # failures=1 with no errors: the designated assertion failed; an exception would count as an error
                killed = r.returncode != 0 and re.search(r'FAILED \(failures=\d+\)\s*$', err)
                if not killed:
                    survivors.append((label, err[-1500:]))
        self.assertEqual(survivors, [], 'mutants survived or failed for the wrong reason: %s' % survivors)


if __name__ == '__main__':
    unittest.main()
