"""Trust-boundary probes for tools/public_intake_check.py. No network.

Run with INTAKE_CHECKER=<path to a copy of the checker>. Tests named
probe_F*_current_* assert the UNWANTED outcome on the current checker (they pass
on the current copy and must fail on the patched copy). Tests named
probe_R*_* are rejected hypotheses: they assert the checker already fails
closed / behaves correctly, and must pass on both copies.
"""
import base64
import contextlib
import copy
import hashlib
import http.client
import importlib.util
import io
import json
import os
import tempfile
from pathlib import Path
from unittest import mock
import unittest

CHECKER = Path(os.environ.get('INTAKE_CHECKER', Path(__file__).with_name('public_intake_check_current.py')))
spec = importlib.util.spec_from_file_location('intake', CHECKER)
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)

REPO = 'd6g8k5htny-coder/main'
HEAD = 'a' * 40
BASE = 'b' * 40
SOURCE = 'c' * 40
MAIN_TIP = 'f' * 40


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


class FakeAPI:
    """Same shape as tests/test_public_intake.py: a dict of routes, no network."""
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


class Fixture(unittest.TestCase):
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
        source_repo = sr['repository']
        self.api.routes[f'/repos/{source_repo}'] = {'private': False, 'full_name': source_repo}
        self.api.routes[f'/repos/{source_repo}/git/commits/{sr["commit"]}'] = {'sha': sr['commit']}
        self.api.routes[f'/repos/{source_repo}/git/trees/{sr["commit"]}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': sr['path'], 'mode': '100644', 'type': 'blob', 'size': len(self.source), 'sha': blob(self.source)}]}
        self.api.routes[f'/repos/{source_repo}/git/blobs/{blob(self.source)}'] = {
            'encoding': 'base64', 'size': len(self.source), 'content': base64.b64encode(self.source).decode(), 'sha': blob(self.source)}

    def run_check(self):
        return intake.check(self.event, self.api, REPO)

    def reject(self, message):
        with self.assertRaisesRegex((ValueError, OSError), message):
            self.run_check()

    def run_cli(self, env_extra=None):
        with tempfile.TemporaryDirectory() as folder:
            event_file = Path(folder) / 'event.json'
            event_file.write_text(json.dumps(self.event))
            env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': REPO, 'GH_TOKEN': 'test-only'}
            env.update(env_extra or {})
            with mock.patch.dict(os.environ, env), mock.patch.object(intake, 'API', return_value=self.api), contextlib.redirect_stdout(io.StringIO()) as output:
                code = intake.main(['--event', str(event_file)])
            return code, output.getvalue()


# ---------------------------------------------------------------------------
# FINDINGS: these assert the unwanted outcome on the current checker.
# ---------------------------------------------------------------------------

class F1_StaleBaseSha(Fixture):
    """F1: 'landed intake is immutable' is checked against pr.base.sha only.

    GitHub semantics relied on: pull.base.sha is the base tip as of the PR's
    last merge-state recomputation, not necessarily the current refs/heads/main;
    and pull_request_target never re-fires on a push to main. So a package id
    that lands on main after the PR's base snapshot is invisible to this check.
    """
    def setUp(self):
        super().setUp()
        # Current main tip already carries the very same package id.
        self.api.routes[f'/repos/{REPO}/git/ref/heads/main'] = {'ref': 'refs/heads/main', 'object': {'type': 'commit', 'sha': MAIN_TIP}}
        self.api.routes[f'/repos/{REPO}/git/trees/{MAIN_TIP}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': 'incoming', 'type': 'tree', 'mode': '040000', 'sha': 'd' * 40},
            {'path': 'incoming/example', 'type': 'tree', 'mode': '040000', 'sha': 'e' * 40},
            {'path': 'incoming/example/RESULT.md', 'type': 'blob', 'mode': '100644', 'sha': blob(b'landed'), 'size': 6}]}

    def probe_F1_current_green_when_package_id_already_on_main_tip(self):
        # GITHUB_SHA under pull_request_target is the trusted base tip at event
        # time; the current checker ignores it and pr.base.sha alone decides.
        code, out = self.run_cli({'GITHUB_SHA': MAIN_TIP})
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)['package'], 'incoming/example')
        self.assertFalse(any('heads/main' in r or MAIN_TIP in r for r in self.api.calls),
                         'checker never consulted the current main tip')


class F2_EscapingExceptions(Fixture):
    """F2: exception types outside main()'s except tuple escape as tracebacks.

    Exit status is still 1 (uncaught exception), so this is not a fail-open,
    but the documented JSON verdict is not printed and the failure is
    indistinguishable by exit code from a rejection.
    """
    def probe_F2_current_non_dict_pr_response_raises_AttributeError_out_of_main(self):
        self.api.routes[f'/repos/{REPO}/pulls/7'] = []   # e.g. a malformed/empty JSON array body
        with self.assertRaises(AttributeError):
            self.run_cli()

    def probe_F2_current_http_client_exception_escapes_main(self):
        # urllib can surface http.client.IncompleteRead (HTTPException, NOT an
        # OSError) from response.read() on a cut-off transfer.
        self.api.routes[f'/repos/{REPO}/pulls/7'] = http.client.IncompleteRead(b'', 10)
        with self.assertRaises(http.client.HTTPException):
            self.run_cli()


class F3_ProtectedNameStemOnly(Fixture):
    """F3 (INFO): protected-name refusal matches the exact stem before the first
    dot, so STATUS_v2.md / STATUSv2.md / claims.json inside a package pass.
    No repository tool consumes incoming/ at this head, so scientific effect is
    unaffected; recorded as an observation against the task's 'STATUS*' wording.
    """
    def probe_F3_current_status_variants_inside_package_accepted(self):
        for name in ['STATUS_v2.md', 'STATUSv2.md', 'claims.json', 'PROOF_INDEX-2.md']:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.raw[name] = b'{}\n'
                self.save()
                self.assertEqual(self.run_check()['lane'], 'incoming')
                self.raw = old


# ---------------------------------------------------------------------------
# REJECTED HYPOTHESES: the checker already fails closed / behaves correctly.
# ---------------------------------------------------------------------------

class R_TrustBoundary(Fixture):
    def probe_R01_fork_without_label_and_without_incoming_path_cannot_reach_engineering_lane(self):
        # intake = head_repo != repository or label or incoming path. A fork is
        # always intake. Outsiders cannot create a same-repo head.
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = [
            {'filename': 'docs/notes.md', 'status': 'added', 'sha': 'd' * 40}]
        self.pr['changed_files'] = 1
        self.reject('incoming-only')

    def probe_R02_head_repo_lookalikes_do_not_select_engineering_lane(self):
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = [
            {'filename': 'docs/notes.md', 'status': 'added', 'sha': 'd' * 40}]
        self.pr['changed_files'] = 1
        for name in ['D6G8K5HTNY-CODER/main', 'd6g8k5htny-coder/main.git', 'd6g8k5htny-coder/main/', 'd6g8k5htny-coder/main ', 'd6g8k5htny-coder/Main']:
            with self.subTest(name=name):
                self.pr['head']['repo']['full_name'] = name
                self.reject('incoming-only|invalid head repository')

    def probe_R03_missing_or_non_dict_head_repo_fails_closed(self):
        for value in [None, 'd6g8k5htny-coder/main', 7]:
            with self.subTest(value=value):
                self.pr['head']['repo'] = value
                self.reject('head repository')

    def probe_R04_event_head_must_equal_api_head(self):
        self.event['pull_request']['head']['sha'] = 'e' * 40
        self.reject('head changed since event')

    def probe_R05_label_or_count_change_between_snapshots_is_refused(self):
        original_get = self.api.get
        reads = 0
        def changing_get(route):
            nonlocal reads
            result = original_get(route)
            if route == f'/repos/{REPO}/pulls/7':
                reads += 1
                if reads == 2:
                    result['labels'] = [{'name': 'results-for-review'}]
            return result
        self.api.get = changing_get
        self.reject('changed during verification')

    def probe_R06_force_push_between_snapshot_and_listing_then_back_is_refused(self):
        # Head A (snapshotted) = package + tools/evil.py (count 4). Listing is
        # served for head B = package only. Count binding refuses it.
        head_tree = self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1']['tree']
        head_tree.append({'path': 'tools/evil.py', 'type': 'blob', 'mode': '100644', 'sha': 'd' * 40, 'size': 1})
        self.pr['changed_files'] = 4
        self.reject('incomplete file listing')
        # Variant: B lists 4 rows, all inside the package, so counts agree. The
        # head tree (A) still does not carry the 4th package file: refused.
        rows = self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1']
        rows.append({'filename': self.package + 'extra.md', 'status': 'added', 'sha': blob(b'x')})
        self.reject('package tree does not match')
        # Variant: same paths, different blob identity between listing and tree.
        rows.pop()
        self.pr['changed_files'] = 3
        head_tree.pop()
        rows[0]['sha'] = 'd' * 40
        self.reject('identity mismatch')

    def probe_R07_closed_mid_run_is_refused(self):
        original_get = self.api.get
        reads = 0
        def changing_get(route):
            nonlocal reads
            result = original_get(route)
            if route == f'/repos/{REPO}/pulls/7':
                reads += 1
                if reads == 2:
                    result['state'] = 'closed'
            return result
        self.api.get = changing_get
        self.reject('not open')

    def probe_R08_top_level_file_named_incoming_routes_to_intake_and_is_refused(self):
        self.pr['head']['repo']['full_name'] = REPO
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = [
            {'filename': 'incoming', 'status': 'added', 'sha': 'd' * 40}]
        self.pr['changed_files'] = 1
        self.reject('incoming-only')

    def probe_R09_base_carrying_package_id_as_file_or_symlink_is_refused(self):
        for entry in [{'path': 'incoming/example', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40},
                      {'path': 'incoming/example', 'mode': '120000', 'type': 'blob', 'sha': 'd' * 40},
                      {'path': 'incoming/example', 'mode': '040000', 'type': 'tree', 'sha': 'd' * 40}]:
            with self.subTest(entry=entry):
                self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1']['tree'] = [entry]
                self.reject('new package id')

    def probe_R10_sibling_prefix_on_base_is_not_a_false_collision(self):
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1']['tree'] = [
            {'path': 'incoming/example-2/RESULT.md', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40}]
        self.assertEqual(self.run_check()['package'], 'incoming/example')

    def probe_R11_zero_changed_files_fails_closed(self):
        self.pr['changed_files'] = 0
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = []
        self.reject('changed file count')

    def probe_R12_source_repo_route_injection_refused(self):
        for repo in ['d6g8k5htny-coder/Math-?x=1', 'd6g8k5htny-coder/Math-/../main', 'd6g8k5htny-coder/math-', 'd6g8k5htny-coder/Math-#']:
            with self.subTest(repo=repo):
                self.manifest['sources'][0]['repository'] = repo
                self.save()
                self.reject('source repository outside')

    def probe_R13_gitlink_inside_package_is_refused(self):
        tree_rows = self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1']['tree']
        tree_rows.append({'path': self.package + 'sub', 'type': 'commit', 'mode': '160000', 'sha': 'd' * 40})
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'].append(
            {'filename': self.package + 'sub', 'status': 'added', 'sha': 'd' * 40})
        self.pr['changed_files'] = 4
        self.reject('regular non-executable')

    def probe_R14_wrong_event_name_repository_or_missing_token_fail_closed_before_network(self):
        with mock.patch.object(intake, 'API') as api:
            code, out = self.run_cli({'GITHUB_EVENT_NAME': 'pull_request'})
            self.assertEqual(code, 1)
            api.assert_not_called()
        code, out = self.run_cli({'GITHUB_REPOSITORY': 'visitor/main'})
        self.assertEqual(code, 1)
        self.assertIn('wrong repository', out)
        self.assertEqual(self.api.calls, [])
        with mock.patch.dict(os.environ, {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': REPO, 'GH_TOKEN': ''}), contextlib.redirect_stdout(io.StringIO()) as out2:
            with tempfile.TemporaryDirectory() as folder:
                event_file = Path(folder) / 'event.json'
                event_file.write_text(json.dumps(self.event))
                self.assertEqual(intake.main(['--event', str(event_file)]), 1)
        self.assertIn('token missing', out2.getvalue())

    def probe_R15_pass_always_prints_verdict_and_exit_codes_are_distinct(self):
        code, out = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)['lane'], 'incoming')
        self.raw['RESULT.md'] = b'\n'
        self.save()
        code, out = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)['result'], 'REJECTED')


class R_Transport(unittest.TestCase):
    """Real API class with a faked opener: size cap, redirects, empty bodies."""
    def make_api(self, body, chunks=None):
        api = intake.API('t')
        class Resp:
            def __init__(self):
                self.pos = 0
            def read(self, n=-1):
                if chunks is not None:
                    # Simulate a transport that returns short reads.
                    part = chunks.pop(0) if chunks else b''
                    return part
                out = body[self.pos:self.pos + n] if n >= 0 else body[self.pos:]
                self.pos += len(out)
                return out
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        class Opener:
            def __init__(self):
                self.requests = []
            def open(self, request, timeout=None):
                self.requests.append(request)
                return Resp()
        api.opener = Opener()
        return api

    def probe_R16_response_over_cap_refused_and_exact_cap_accepted(self):
        big = b'[' + b'1,' * (intake.MAX_RESPONSE_BYTES // 2) + b'1]'
        self.assertGreater(len(big), intake.MAX_RESPONSE_BYTES)
        with self.assertRaisesRegex(ValueError, 'exceeds safe limit'):
            self.make_api(big).get('/repos/x/y')
        exact = b'"' + b'a' * (intake.MAX_RESPONSE_BYTES - 2) + b'"'
        self.assertEqual(len(exact), intake.MAX_RESPONSE_BYTES)
        self.assertEqual(len(self.make_api(exact).get('/repos/x/y')), intake.MAX_RESPONSE_BYTES - 2)

    def probe_R17_empty_or_non_json_body_refused(self):
        with self.assertRaises(ValueError):
            self.make_api(b'').get('/repos/x/y')
        with self.assertRaises(ValueError):
            self.make_api(b'<html>').get('/repos/x/y')

    def probe_R18_route_prefix_and_header_injection_refused_and_token_only_to_api_host(self):
        api = self.make_api(b'{}')
        for route in ['/nope', 'https://evil.invalid/repos/x', '/repos/x\r\nX: y', '/repos/x\n']:
            with self.subTest(route=route):
                with self.assertRaisesRegex(ValueError, 'invalid API route'):
                    api.get(route)
        api.get('/repos/x/y')
        request = api.opener.requests[-1]
        self.assertTrue(request.full_url.startswith('https://api.github.com/repos/'))
        self.assertEqual(request.get_header('Authorization'), 'Bearer t')

    def probe_R19_redirect_handler_refuses_every_code(self):
        for code in (301, 302, 303, 307, 308):
            with self.assertRaisesRegex(ValueError, 'redirects refused'):
                intake.NoRedirect().redirect_request(None, None, code, '', {}, 'https://api.github.com/x')


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for cls in (F1_StaleBaseSha, F2_EscapingExceptions, F3_ProtectedNameStemOnly, R_TrustBoundary, R_Transport):
        names = [n for n in dir(cls) if n.startswith('probe_')]
        suite.addTests(cls(n) for n in sorted(names))
    return suite


if __name__ == '__main__':
    unittest.main()
