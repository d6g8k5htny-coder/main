"""Negative controls for the trusted public contribution boundary; no network."""
import base64
import copy
import hashlib
import importlib.util
import json
import contextlib
import io
import os
import tempfile
from unittest import mock
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('intake', Path(__file__).resolve().parents[1] / 'tools/public_intake_check.py')
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)

REPO = 'd6g8k5htny-coder/main'
HEAD = 'a' * 40
BASE = 'b' * 40
SOURCE = 'c' * 40
PARENT = 'f' * 40  # an ancestor of the PR head that no branch of the pillar contains
MATH = 'd6g8k5htny-coder/Math-'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def compare(commit, status='behind', ahead_by=0, merge_base=None):
    return {'status': status, 'ahead_by': ahead_by, 'behind_by': 3,
            'merge_base_commit': {'sha': commit if merge_base is None else merge_base}}


class FakeAPI:
    def __init__(self):
        self.routes = {}
        self.calls = []

    def get(self, route):
        self.calls.append(route)
        if route not in self.routes:
            raise AssertionError('Unexpected API request: ' + route)
        result = self.routes[route]
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI()
        self.pr = {'number': 7, 'state': 'open', 'changed_files': 3,
                   'head': {'sha': HEAD, 'repo': {'full_name': 'visitor/main'}},
                   'base': {'sha': BASE, 'ref': 'main', 'repo': {'full_name': REPO}},
                   'labels': []}
        self.event = {'number': 7, 'repository': {'full_name': REPO},
                      'pull_request': copy.deepcopy(self.pr)}
        self.package = 'incoming/example/'
        self.raw = {'RESULT.md': b'# Result\nIllustration only; scientific effect NONE.\n',
                    'output.json': b'{"illustration":true}\n'}
        self.source = b'public source numbers\n'
        self.manifest = {'schema': 1, 'scientific_effect': 'NONE', 'review_status': 'REVIEW_REQUIRED',
                         'sources': [{'repository': 'd6g8k5htny-coder/Math-', 'path': 'coefficients/example.json',
                                      'commit': SOURCE, 'sha256': sha(self.source)}],
                         'artifacts': []}
        self.save()

    def save(self):
        self.manifest['artifacts'] = [{'path': name, 'bytes': len(raw), 'sha256': sha(raw)} for name, raw in self.raw.items()]
        self.files = dict(self.raw, **{'IDENTITY.json': json.dumps(self.manifest).encode()})
        self.install()

    def install(self):
        entries = []
        changes = []
        for name, raw in self.files.items():
            path = self.package + name
            entries.append({'path': path, 'type': 'blob', 'mode': '100644', 'sha': blob(raw), 'size': len(raw)})
            changes.append({'filename': path, 'status': 'added', 'sha': blob(raw)})
            self.api.routes[f'/repos/{REPO}/git/blobs/{blob(raw)}'] = {'encoding': 'base64', 'content': base64.b64encode(raw).decode(), 'size': len(raw), 'sha': blob(raw)}
        self.pr['changed_files'] = len(changes)
        self.api.routes[f'/repos/{REPO}/pulls/7'] = self.pr
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = changes
        self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1'] = {'truncated': False, 'tree': entries}
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1'] = {'truncated': False, 'tree': []}
        sr = self.manifest['sources'][0]
        if sr['commit'] not in (HEAD, BASE):  # a test citing the PR's own commits sets its routes itself
            self.install_source(sr['repository'], sr['commit'], sr['path'], self.source)

    def install_source(self, source_repo, commit, path, raw, branches=('main',), reachable_from=('main',)):
        # What GitHub serves for any commit object in the repository network, plus the
        # compare/branches routes that decide whether a branch of the pillar contains it.
        self.api.routes[f'/repos/{source_repo}'] = {'private': False, 'full_name': source_repo, 'default_branch': 'main'}
        self.api.routes[f'/repos/{source_repo}/git/commits/{commit}'] = {'sha': commit}
        self.api.routes[f'/repos/{source_repo}/git/trees/{commit}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': path, 'mode': '100644', 'type': 'blob', 'size': len(raw), 'sha': blob(raw)}]}
        self.api.routes[f'/repos/{source_repo}/git/blobs/{blob(raw)}'] = {
            'encoding': 'base64', 'size': len(raw), 'content': base64.b64encode(raw).decode(), 'sha': blob(raw)}
        self.api.routes[f'/repos/{source_repo}/contents/{path}?ref={commit}'] = {
            'type': 'file', 'encoding': 'base64', 'size': len(raw), 'content': base64.b64encode(raw).decode(), 'sha': blob(raw)}
        self.api.routes[f'/repos/{source_repo}/branches?per_page=100&page=1'] = [{'name': b, 'commit': {'sha': 'd' * 40}} for b in branches]
        for b in branches:
            # Not contained: GitHub reports the object as diverged from the branch
            # ("this commit does not belong to any branch on this repository").
            self.api.routes[f'/repos/{source_repo}/compare/{b}...{commit}?per_page=1'] = (
                compare(commit) if b in reachable_from else compare(commit, 'diverged', 2, BASE))

    def run_check(self):
        return intake.check(self.event, self.api, REPO)

    def reject(self, message):
        with self.assertRaisesRegex((ValueError, OSError), message):
            self.run_check()

    def test_valid_fork_without_label_is_restricted_and_verified(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertEqual(result['scientific_effect'], 'NONE')
        self.assertEqual(result['verified_sources'], 1)
        self.assertEqual(result['head'], HEAD)

    def test_ordinary_same_repo_engineering_is_allowed(self):
        self.pr['head']['repo']['full_name'] = REPO
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = [
            {'filename': name, 'status': 'modified', 'sha': 'd' * 40}
            for name in ['.github/workflows/check.yml', 'tools/check.py', 'docs/theorem α.md']]
        self.assertEqual(self.run_check()['lane'], 'maintainer-engineering')
        self.assertEqual(len(self.api.calls), 3)

    def test_label_removal_does_not_bypass_same_repo_incoming(self):
        self.pr['head']['repo']['full_name'] = REPO
        self.raw['STATUS.md'] = b'ACCEPT\n'
        self.save()
        self.reject('protected')

    def test_label_alone_restricts_engineering_edits(self):
        self.pr['head']['repo']['full_name'] = REPO
        self.pr['labels'] = [{'name': 'results-for-review'}]
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'][0]['filename'] = 'README.md'
        self.reject('incoming-only')

    def test_fork_cannot_modify_guard_without_label(self):
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'][0]['filename'] = 'tools/public_intake_check.py'
        self.reject('incoming-only')

    def test_protected_status_and_claim_paths_refused(self):
        for name in ['STATUS.md', 'PROOF_INDEX.md', 'LANDING_CLAIMS.json', 'claims/graph.json']:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.raw[name] = b'changed'
                self.save()
                self.reject('protected')
                self.raw = old

    def test_rename_from_status_cannot_hide_as_incoming(self):
        change = self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'][0]
        change.update(status='renamed', previous_filename='STATUS.md')
        self.reject('add-only')

    def test_existing_package_is_immutable(self):
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1']['tree'] = [
            {'path': self.package + 'old.txt', 'mode': '100644', 'type': 'blob', 'sha': 'd'*40}]
        self.reject('new package')

    def test_symlink_and_executable_and_submodule_refused(self):
        entry = self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1']['tree'][0]
        for mode, kind in [('120000', 'blob'), ('100755', 'blob'), ('160000', 'commit')]:
            with self.subTest(mode=mode):
                entry.update(mode=mode, type=kind)
                self.reject('regular non-executable')

    def test_truncated_tree_and_file_listing_refused(self):
        self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1']['truncated'] = True
        self.reject('truncated')
        self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1']['truncated'] = False
        self.pr['changed_files'] += 1
        self.reject('incomplete')

    def test_missing_or_false_digest_refused(self):
        for digest in ['0'*64, None]:
            with self.subTest(digest=digest):
                self.manifest['artifacts'][0]['sha256'] = digest
                self.files['IDENTITY.json'] = json.dumps(self.manifest).encode()
                self.install()
                self.reject('artifact')

    def test_unlisted_artifact_and_duplicate_manifest_key_refused(self):
        self.files['hidden.txt'] = b'not listed'
        self.install()
        self.reject('exactly cover')
        self.files.pop('hidden.txt')
        self.files['IDENTITY.json'] = b'{"schema":1,"schema":1}'
        self.install()
        self.reject('duplicate')

    def test_source_pin_and_public_bytes_verified(self):
        self.source = b'changed source'
        self.install()
        self.reject('source digest')

    def test_private_and_foreign_sources_refused(self):
        self.api.routes['/repos/d6g8k5htny-coder/Math-']['private'] = True
        self.reject('public')
        self.manifest['sources'][0]['repository'] = 'stranger/repo'
        self.save()
        self.reject('source repository')

    def test_missing_head_and_stale_head_fail_closed(self):
        self.pr['head']['repo'] = None
        self.reject('head repository')
        self.pr['head']['repo'] = {'full_name': 'visitor/main'}
        self.pr['head']['sha'] = 'd' * 40
        self.reject('head changed')

    def test_bool_is_not_size_or_schema(self):
        self.manifest['artifacts'][0]['bytes'] = True
        self.files['IDENTITY.json'] = json.dumps(self.manifest).encode()
        self.install()
        self.reject('artifact')

    def test_path_escape_and_active_content_refused(self):
        for name in ['../evil.txt', 'nested/../../evil.txt', 'plot.svg', 'run.py', 'notebook.ipynb', '.env', 'page.html']:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.raw[name] = b'payload'
                self.save()
                self.reject('path|extension')
                self.raw = old

    def test_known_secret_refused_without_printing_value(self):
        self.raw['output.json'] = ('{"token":"' + 'ghp_' + 'X'*36 + '"}').encode()
        self.save()
        self.reject('possible credential')

    def test_false_scientific_status_refused(self):
        self.manifest['scientific_effect'] = 'ACCEPT'
        self.save()
        self.reject('review-only')

    def test_api_failure_cannot_be_passed(self):
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = OSError('unavailable')
        self.reject('unavailable')


    def test_source_commit_must_resolve_to_exact_commit(self):
        self.api.routes[f'/repos/d6g8k5htny-coder/Math-/git/commits/{SOURCE}'] = {'sha': 'd'*40}
        self.reject('exact commit')

    def test_blob_sha_tampering_refused(self):
        path = f'/repos/{REPO}/git/blobs/{blob(self.files["RESULT.md"])}'
        self.api.routes[path]['content'] = base64.b64encode(b'X' * len(self.files['RESULT.md'])).decode()
        self.reject('Git blob identity')

    def test_digest_must_be_sha256_and_commit_must_be_40_hex(self):
        self.manifest['sources'][0]['commit'] = 'main'
        self.save()
        self.reject('full SHA')

    def test_size_limits_apply_before_blob_fetch(self):
        self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1']['tree'][0]['size'] = 262145
        self.reject('size limit')
        self.assertFalse(any('/git/blobs/' in path for path in self.api.calls))

    def test_second_package_refused(self):
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'][0]['filename'] = 'incoming/other/RESULT.md'
        self.reject('one package')

    def test_nonfinite_and_invalid_utf8_json_refused(self):
        for data in [b'{"x": NaN}', b'\xff']:
            with self.subTest(data=data):
                self.raw['output.json'] = data
                self.save()
                self.reject('nonfinite|decode')

    def test_invalid_png_refused_without_decoding_image(self):
        self.raw['plot.png'] = b'not a png'
        self.save()
        self.reject('PNG signature')

    def test_renamed_out_of_incoming_still_routes_to_intake(self):
        self.pr['head']['repo']['full_name'] = REPO
        rows = self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1']
        rows[0].update(filename='docs/result.md', previous_filename='incoming/old/RESULT.md', status='renamed')
        self.reject('incoming-only')

    def test_deleted_or_modified_intake_refused(self):
        row = self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'][0]
        for status in ['removed', 'modified', 'copied']:
            with self.subTest(status=status):
                row['status'] = status
                self.reject('add-only')

    def test_complete_pagination_before_maintainer_bypass(self):
        self.pr['head']['repo']['full_name'] = REPO
        self.pr['changed_files'] = 101
        rows = [{'filename': f'docs/item{i}.md', 'status': 'added'} for i in range(101)]
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = rows[:100]
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=2'] = rows[100:]
        self.assertEqual(self.run_check()['lane'], 'maintainer-engineering')
        self.assertTrue(any('page=2' in route for route in self.api.calls))
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=2'] = [{'filename': 'incoming/hidden/result.md', 'status': 'added'}]
        self.reject('too many package files')

    def test_pr_race_fails_at_final_readback(self):
        original_get = self.api.get
        reads = 0
        def changing_get(route):
            nonlocal reads
            result = original_get(route)
            if route == f'/repos/{REPO}/pulls/7':
                reads += 1
                if reads == 2:
                    result['head']['sha'] = 'e' * 40
            return result
        self.api.get = changing_get
        self.reject('changed during verification')

    def test_non_open_pr_and_wrong_base_are_refused(self):
        self.pr['state'] = 'closed'
        self.reject('not open')
        self.pr['state'] = 'open'
        self.pr['base']['ref'] = 'other'
        self.reject('targets main')

    def test_cli_pass_and_failure_are_distinct_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            event_file = Path(folder) / 'event.json'
            event_file.write_text(json.dumps(self.event))
            env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': REPO, 'GH_TOKEN': 'test-only'}
            with mock.patch.dict(os.environ, env), mock.patch.object(intake, 'API', return_value=self.api), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(intake.main(['--event', str(event_file)]), 0)
                self.assertEqual(json.loads(output.getvalue())['lane'], 'incoming')
                self.raw['output.json'] = ('{"token":"' + 'ghp_' + 'X'*36 + '"}').encode()
                self.save()
                output.truncate(0)
                output.seek(0)
                self.assertEqual(intake.main(['--event', str(event_file)]), 1)
                self.assertEqual(json.loads(output.getvalue())['result'], 'REJECTED')
                self.assertNotIn('X'*36, output.getvalue())

    def test_cli_untrusted_event_refused_before_network(self):
        with mock.patch.dict(os.environ, {'GITHUB_EVENT_NAME': 'pull_request'}), mock.patch.object(intake, 'API') as api, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(intake.main([]), 1)
            api.assert_not_called()

    def test_transport_does_not_follow_redirects(self):
        with self.assertRaisesRegex(ValueError, 'redirects refused'):
            intake.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.invalid/token')

    def test_source_symlink_refused_despite_file_like_contents_response(self):
        # GitHub's contents endpoint follows this link and reports the target as
        # type=file. The immutable tree still names a mode120000 symlink blob.
        route = f'/repos/d6g8k5htny-coder/Math-/git/trees/{SOURCE}?recursive=1'
        entry = self.api.routes[route]['tree'][0]
        entry.update(mode='120000', sha=blob(b'target.json'), size=11)
        self.reject('regular non-executable public file')
        self.assertFalse(any('/contents/' in path for path in self.api.calls))

    def test_source_tree_and_blob_sha_must_match(self):
        route = f'/repos/d6g8k5htny-coder/Math-/git/trees/{SOURCE}?recursive=1'
        self.api.routes[route]['tree'][0]['sha'] = 'd'*40
        self.api.routes['/repos/d6g8k5htny-coder/Math-/git/blobs/' + 'd'*40] = {
            'encoding': 'base64', 'size': len(self.source), 'content': base64.b64encode(self.source).decode(), 'sha': blob(self.source)}
        self.reject('Git blob identity mismatch')

    def test_known_credential_in_package_path_is_refused_without_logging(self):
        token = 'sk-' + 'a' * 40
        self.package = 'incoming/' + token + '/'
        self.save()
        with tempfile.TemporaryDirectory() as folder:
            event_file = Path(folder) / 'event.json'
            event_file.write_text(json.dumps(self.event))
            env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': REPO, 'GH_TOKEN': 'test-only'}
            with mock.patch.dict(os.environ, env), mock.patch.object(intake, 'API', return_value=self.api), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(intake.main(['--event', str(event_file)]), 1)
                self.assertIn('possible credential in path', output.getvalue())
                self.assertNotIn(token, output.getvalue())

    def reachability_calls(self):
        return [c for c in self.api.calls if '/compare/' in c or '/branches?' in c]

    def test_fabricated_source_in_fork_head_ancestor_refused(self):
        # refs/pull/7/head brings the submitter's own PARENT commit into the pillar's object
        # store, so /git/commits, /git/trees and /git/blobs resolve it although no branch has it.
        forged = b'{"enclosure": [0, 0], "note": "bytes chosen by the submitter"}\n'
        self.manifest['sources'] = [{'repository': REPO, 'path': 'coefficients/side24_v1/ENCLOSURE.json',
                                     'commit': PARENT, 'sha256': sha(forged)}]
        self.save()
        self.install_source(REPO, PARENT, 'coefficients/side24_v1/ENCLOSURE.json', forged, reachable_from=())
        self.reject('not reachable from any branch')
        self.assertFalse(any(f'/git/trees/{PARENT}' in c or f'/git/blobs/{blob(forged)}' in c for c in self.api.calls))

    def test_self_citation_of_pr_head_refused(self):
        # HEAD's tree and blobs are served already because the checker reads the package from them.
        own = self.raw['output.json']
        self.manifest['sources'] = [{'repository': REPO, 'path': self.package + 'output.json', 'commit': HEAD, 'sha256': sha(own)}]
        self.save()
        self.api.routes[f'/repos/{REPO}'] = {'private': False, 'full_name': REPO, 'default_branch': 'main'}
        self.api.routes[f'/repos/{REPO}/git/commits/{HEAD}'] = {'sha': HEAD}
        self.api.routes[f'/repos/{REPO}/branches?per_page=100&page=1'] = [{'name': 'main', 'commit': {'sha': BASE}}]
        self.api.routes[f'/repos/{REPO}/compare/main...{HEAD}?per_page=1'] = compare(HEAD, 'ahead', 1, BASE)
        self.reject('not reachable from any branch')

    def test_source_reachable_only_from_research_branch_accepted(self):
        raw = b'public numbers on a research branch\n'
        branch = 'chatgpt/drive-github-hardening-20260919'
        self.manifest['sources'] = [{'repository': REPO, 'path': 'research/numbers.json', 'commit': 'd' * 40, 'sha256': sha(raw)}]
        self.save()
        self.install_source(REPO, 'd' * 40, 'research/numbers.json', raw, branches=('main', branch), reachable_from=(branch,))
        self.assertEqual(self.run_check()['verified_sources'], 1)
        self.assertIn(f'/repos/{REPO}/compare/{branch}...{"d" * 40}?per_page=1', self.api.calls)
        self.assertFalse(any('page=2' in c for c in self.api.calls))  # a short page ends the enumeration

    def test_reachable_default_branch_costs_one_compare_call(self):
        self.assertEqual(self.run_check()['verified_sources'], 1)
        self.assertEqual(self.reachability_calls(), [f'/repos/{MATH}/compare/main...{SOURCE}?per_page=1'])
        self.assertEqual(len([c for c in self.api.calls if MATH in c]), 5)

    def test_compare_ahead_diverged_or_nonzero_ahead_by_refused(self):
        route = f'/repos/{MATH}/compare/main...{SOURCE}?per_page=1'
        for status, ahead_by, merge_base in [('ahead', 1, BASE), ('diverged', 2, BASE), ('behind', 1, SOURCE),
                                             ('identical', 1, SOURCE), ('ahead', 0, SOURCE), ('unknown', 0, SOURCE)]:
            with self.subTest(status=status, ahead_by=ahead_by):
                self.api.routes[route] = compare(SOURCE, status, ahead_by, merge_base)
                self.reject('not reachable from any branch')

    def test_malformed_compare_response_is_never_reachability(self):
        route = f'/repos/{MATH}/compare/main...{SOURCE}?per_page=1'
        good = compare(SOURCE)
        shapes = [('non-dict', [good]), ('empty', {}), ('missing merge base', {k: v for k, v in good.items() if k != 'merge_base_commit'}),
                  ('merge base not a dict', dict(good, merge_base_commit=SOURCE)), ('wrong merge base', compare(SOURCE, merge_base=BASE)),
                  ('bool ahead_by', dict(good, ahead_by=False)), ('string ahead_by', dict(good, ahead_by='0')),
                  ('missing status', {k: v for k, v in good.items() if k != 'status'})]
        for label, data in shapes:
            with self.subTest(label=label):
                self.api.routes[route] = data
                self.reject('not reachable from any branch')
        self.api.routes[route] = good
        self.api.routes[f'/repos/{MATH}/branches?per_page=100&page=1'] = {'name': 'main'}
        self.api.routes[route] = compare(SOURCE, 'diverged', 2, BASE)
        self.reject('branch listing unavailable')

    def test_unsafe_default_branch_is_skipped_and_branch_list_consulted(self):
        main_compare = f'/repos/{MATH}/compare/main...{SOURCE}?per_page=1'
        for name in ['main..evil', 'ma in', '-flag', 'x\ny', 'tab\t', '.hidden', '/main', 'main?x=1', 'main#1', None, 5, ['main']]:
            with self.subTest(name=repr(name)):
                self.api.routes[f'/repos/{MATH}']['default_branch'] = name
                self.api.routes[f'/repos/{MATH}/branches?per_page=100&page=1'] = [{'name': name}, {'name': 'main'}]
                self.api.calls.clear()
                self.assertEqual(self.run_check()['verified_sources'], 1)
                self.assertEqual(self.reachability_calls(), [f'/repos/{MATH}/branches?per_page=100&page=1', main_compare])
                self.api.routes[f'/repos/{MATH}/branches?per_page=100&page=1'] = [{'name': name}, 'main', {'commit': {}}]
                self.api.calls.clear()
                self.reject('not reachable from any branch')
                self.assertEqual([c for c in self.api.calls if '/compare/' in c], [])

    def test_branch_enumeration_stops_at_page_bound(self):
        self.api.routes[f'/repos/{MATH}/compare/main...{SOURCE}?per_page=1'] = compare(SOURCE, 'diverged', 2, BASE)
        names = ['main'] + [f'topic-{i}' for i in range(1, 300)]
        for page in range(1, 4):
            self.api.routes[f'/repos/{MATH}/branches?per_page=100&page={page}'] = [{'name': n} for n in names[(page - 1) * 100:page * 100]]
        for n in names[1:]:
            self.api.routes[f'/repos/{MATH}/compare/{n}...{SOURCE}?per_page=1'] = compare(SOURCE, 'diverged', 2, BASE)
        # A fourth page would vouch; FakeAPI raises AssertionError if it is ever requested.
        self.api.routes[f'/repos/{MATH}/branches?per_page=100&page=4'] = [{'name': 'vouching'}]
        self.api.routes[f'/repos/{MATH}/compare/vouching...{SOURCE}?per_page=1'] = compare(SOURCE)
        self.reject('not reachable from any branch')
        self.assertEqual([c for c in self.api.calls if '/branches?' in c],
                         [f'/repos/{MATH}/branches?per_page=100&page={p}' for p in (1, 2, 3)])
        self.assertEqual(len([c for c in self.api.calls if '/compare/' in c]), 300)
        self.assertEqual(len([c for c in self.api.calls if '/compare/main...' in c]), 1)


if __name__ == '__main__':
    unittest.main()
