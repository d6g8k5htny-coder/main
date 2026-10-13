"""Refutation probes for finding TB-F2 (skeptic #3, trust-boundary lens). No network.

INTAKE_CHECKER selects the checker copy (default: ./tools/public_intake_check.py,
a byte copy of origin/main). Every test here passes on the UNPATCHED copy and
records why the escaping exceptions are not a MINOR defect against a stated
intake guarantee: the process still exits 1 with nothing on stdout, the workflow
consumes only the exit status, GitHub's own error responses are OSError and
already caught, a non-object body is not something api.github.com serves for
/pulls/{n}, and the finder's patch does not change any verdict, only the log form.
"""
import base64, contextlib, copy, hashlib, http.client, importlib.util, io, json, os, subprocess, sys, tempfile, urllib.error
from pathlib import Path
from unittest import mock
import unittest

HERE = Path(__file__).resolve().parent
CHECKER = Path(os.environ.get('INTAKE_CHECKER', HERE / 'tools/public_intake_check.py'))
spec = importlib.util.spec_from_file_location('intake', CHECKER)
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)
WORKFLOW = Path('/home/user/main-review-refute-TB-F2-3/.github/workflows/public-intake.yml')
REPO, HEAD, BASE, SOURCE = 'd6g8k5htny-coder/main', 'a' * 40, 'b' * 40, 'c' * 40
MARKER = b'SUBMITTED-LOOKING-BYTES'


def sha(raw): return hashlib.sha256(raw).hexdigest()
def blob(raw): return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


class FakeAPI:
    def __init__(self): self.routes, self.calls = {}, []
    def get(self, route):
        self.calls.append(route)
        if route not in self.routes: raise AssertionError('Unexpected API request: ' + route)
        result = self.routes[route]
        if isinstance(result, Exception): raise result
        return copy.deepcopy(result)


class Fixture(unittest.TestCase):
    """A legitimate package, as tests/test_public_intake.py builds one (post-#153 routes included)."""
    def setUp(self):
        self.api = FakeAPI()
        self.pr = {'number': 7, 'state': 'open', 'changed_files': 3, 'head': {'sha': HEAD, 'repo': {'full_name': 'visitor/main'}},
                   'base': {'sha': BASE, 'ref': 'main', 'repo': {'full_name': REPO}}, 'labels': []}
        self.event = {'number': 7, 'repository': {'full_name': REPO}, 'pull_request': copy.deepcopy(self.pr)}
        raw = {'RESULT.md': b'# Result\nIllustration only; scientific effect NONE.\n', 'output.json': b'{"illustration":true}\n'}
        source = b'public source numbers\n'
        manifest = {'schema': 1, 'scientific_effect': 'NONE', 'review_status': 'REVIEW_REQUIRED',
                    'sources': [{'repository': 'd6g8k5htny-coder/Math-', 'path': 'coefficients/example.json', 'commit': SOURCE, 'sha256': sha(source)}],
                    'artifacts': [{'path': n, 'bytes': len(r), 'sha256': sha(r)} for n, r in raw.items()]}
        files = dict(raw, **{'IDENTITY.json': json.dumps(manifest).encode()})
        entries, changes = [], []
        for name, r in files.items():
            path = 'incoming/example/' + name
            entries.append({'path': path, 'type': 'blob', 'mode': '100644', 'sha': blob(r), 'size': len(r)})
            changes.append({'filename': path, 'status': 'added', 'sha': blob(r)})
            self.api.routes[f'/repos/{REPO}/git/blobs/{blob(r)}'] = {'encoding': 'base64', 'content': base64.b64encode(r).decode(), 'size': len(r), 'sha': blob(r)}
        self.api.routes[f'/repos/{REPO}/pulls/7'] = self.pr
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = changes
        self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1'] = {'truncated': False, 'tree': entries}
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1'] = {'truncated': False, 'tree': []}
        m = 'd6g8k5htny-coder/Math-'
        self.api.routes[f'/repos/{m}'] = {'private': False, 'full_name': m, 'default_branch': 'main'}
        self.api.routes[f'/repos/{m}/git/commits/{SOURCE}'] = {'sha': SOURCE}
        self.api.routes[f'/repos/{m}/compare/main...{SOURCE}?per_page=1'] = {'status': 'behind', 'ahead_by': 0, 'behind_by': 3, 'merge_base_commit': {'sha': SOURCE}}
        self.api.routes[f'/repos/{m}/git/trees/{SOURCE}?recursive=1'] = {'truncated': False, 'tree': [{'path': 'coefficients/example.json', 'mode': '100644', 'type': 'blob', 'size': len(source), 'sha': blob(source)}]}
        self.api.routes[f'/repos/{m}/git/blobs/{blob(source)}'] = {'encoding': 'base64', 'size': len(source), 'content': base64.b64encode(source).decode(), 'sha': blob(source)}

    def run_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            event_file = Path(folder) / 'event.json'
            event_file.write_text(json.dumps(self.event))
            env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': REPO, 'GH_TOKEN': 'test-only'}
            with mock.patch.dict(os.environ, env), mock.patch.object(intake, 'API', return_value=self.api), contextlib.redirect_stdout(io.StringIO()) as out:
                return intake.main(['--event', str(event_file)]), out.getvalue()


class RefuteTBF2(Fixture):
    def test_R1_real_process_exit_is_1_stdout_empty_and_partial_bytes_not_logged(self):
        # The CI step's only observable is the exit status; a traceback on stderr is a red step, not a green one.
        for mode in ('list', 'null', 'incomplete', 'badstatus'):
            with self.subTest(mode=mode):
                p = subprocess.run([sys.executable, '-B', '-I', str(HERE / 'cli_driver.py'), str(CHECKER), mode], capture_output=True,
                                   env=dict(os.environ, INTAKE_PARTIAL=MARKER.decode()))
                self.assertEqual(p.returncode, 1)  # unpatched: uncaught exception; patched: return 1 -- same red step
                self.assertNotIn(MARKER, p.stdout + p.stderr)  # IncompleteRead prints byte counts, never the partial body
                if p.stdout == b'':
                    self.assertIn(b'Traceback', p.stderr)  # unpatched copy: traceback, no JSON verdict
                else:
                    self.assertEqual(json.loads(p.stdout)['result'], 'REJECTED')  # patched copy

    def test_R2_workflow_consumes_exit_status_only(self):
        text = WORKFLOW.read_text()
        self.assertIn('run: python3 -B -I tools/public_intake_check.py\n', text)
        for token in ('continue-on-error', 'GITHUB_OUTPUT', '$(', ' | ', 'tee', 'if: always', '||'):
            self.assertNotIn(token, text)

    def test_R3_github_error_responses_are_OSError_and_already_caught(self):
        # How api.github.com actually answers a missing/forbidden PR: 4xx with a JSON *object* body,
        # surfaced by urllib as HTTPError (an OSError) -> caught -> JSON verdict. A 200 body is always an object.
        body = io.BytesIO(b'{"message":"Not Found","documentation_url":"https://docs.github.com/rest"}')
        self.api.routes[f'/repos/{REPO}/pulls/7'] = urllib.error.HTTPError('https://api.github.com/x', 404, 'Not Found', {}, body)
        code, out = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)['result'], 'REJECTED')
        # RemoteDisconnected (server closes without a status line) is ConnectionResetError -> caught too.
        self.api.routes[f'/repos/{REPO}/pulls/7'] = http.client.RemoteDisconnected('Remote end closed connection without response')
        code, out = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)['result'], 'REJECTED')

    def test_R4_non_dict_body_is_refused_everywhere_but_the_first_line(self):
        # snapshot() guards isinstance(pr, dict): the second /pulls read with a list body is a caught ValueError.
        first = self.api.get
        reads = 0
        def second_read_is_a_list(route):
            nonlocal reads
            if route == f'/repos/{REPO}/pulls/7':
                reads += 1
                if reads == 2: return []
            return first(route)
        self.api.get = second_read_is_a_list
        code, out = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)['reason'], 'PR is not open')

    def test_R5_legitimate_package_under_transport_failure_is_red_either_way(self):
        # A genuine package whose head-tree read is cut off: no copy of the checker can pass it, so the
        # patch does not rescue any legitimate submission; a rerun is the remedy with or without it.
        self.assertEqual(self.run_cli()[0], 0)  # the package itself is green
        self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1'] = http.client.IncompleteRead(b'', 10)
        try:
            code, out = self.run_cli()
        except http.client.HTTPException:
            code, out = 1, ''  # unpatched: uncaught -> interpreter exit status 1 (R1 shows this in a real process)
        self.assertEqual(code, 1)
        self.assertNotIn('"lane"', out)

    def test_R6_no_other_unguarded_attribute_access_on_API_data(self):
        # Every .get()/[] on an API-derived object is behind an isinstance check except pr.get('number').
        # An outsider controls file paths, blob bytes and the fork name; none of those reach an unguarded access.
        src = CHECKER.read_text()
        self.assertEqual(src.count("pr.get('number')"), 1)
        for value in (None, 7, 'x', ['pr']):
            with self.subTest(value=value):
                self.api.routes[f'/repos/{REPO}/pulls/7'] = value
                try:
                    code, out = self.run_cli()  # patched copy: caught, JSON verdict, exit 1
                    self.assertEqual((code, json.loads(out)['result']), (1, 'REJECTED'))
                except AttributeError as error:  # unpatched copy: escapes; message names a Python type, no response bytes
                    self.assertNotIn('pulls', str(error))


if __name__ == '__main__':
    unittest.main()
