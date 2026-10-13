"""Refutation probes for TB-F1 (skeptic #2, trust-boundary). No network.

Run with INTAKE_CHECKER=<copy>. Same fixture shape as tests/test_public_intake.py.
R1: the patched checker is exactly as green as the current one when main moves
    AFTER event time (GITHUB_SHA is frozen at event time); the patch does not
    close the 'no base push re-fires the run' half of the finding.
R2..R4: every post-conflict-resolution state a stale-green same-id PR can reach
    is already refused by the CURRENT checker (and covered by existing tests).
"""
import base64, contextlib, copy, hashlib, importlib.util, io, json, os, tempfile, unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location('intake', Path(os.environ['INTAKE_CHECKER']))
intake = importlib.util.module_from_spec(spec); spec.loader.exec_module(intake)
REPO = 'd6g8k5htny-coder/main'; HEAD = 'a'*40; BASE = 'b'*40; SOURCE = 'c'*40
EVENT_TIP = 'e'*40   # base tip when the event fired (== GITHUB_SHA)
LATER_TIP = 'f'*40   # main tip after a later merge landed the same package id
sha = lambda r: hashlib.sha256(r).hexdigest()
blob = lambda r: hashlib.sha1(b'blob ' + str(len(r)).encode() + b'\0' + r).hexdigest()

class FakeAPI:
    def __init__(self): self.routes = {}; self.calls = []
    def get(self, route):
        self.calls.append(route)
        if route not in self.routes: raise AssertionError('Unexpected API request: ' + route)
        r = self.routes[route]
        if isinstance(r, Exception): raise r
        return copy.deepcopy(r)

class Fixture(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI()
        self.pr = {'number': 7, 'state': 'open', 'changed_files': 3,
                   'head': {'sha': HEAD, 'repo': {'full_name': 'visitor/main'}},
                   'base': {'sha': BASE, 'ref': 'main', 'repo': {'full_name': REPO}}, 'labels': []}
        self.event = {'number': 7, 'repository': {'full_name': REPO}, 'pull_request': copy.deepcopy(self.pr)}
        self.package = 'incoming/example/'
        self.raw = {'RESULT.md': b'# Result\nNONE.\n', 'output.json': b'{"illustration":true}\n'}
        self.source = b'public source numbers\n'
        self.manifest = {'schema': 1, 'scientific_effect': 'NONE', 'review_status': 'REVIEW_REQUIRED',
                         'sources': [{'repository': 'd6g8k5htny-coder/Math-', 'path': 'coefficients/example.json',
                                      'commit': SOURCE, 'sha256': sha(self.source)}], 'artifacts': []}
        self.manifest['artifacts'] = [{'path': n, 'bytes': len(r), 'sha256': sha(r)} for n, r in self.raw.items()]
        self.files = dict(self.raw, **{'IDENTITY.json': json.dumps(self.manifest).encode()})
        entries, changes = [], []
        for name, raw in self.files.items():
            p = self.package + name
            entries.append({'path': p, 'type': 'blob', 'mode': '100644', 'sha': blob(raw), 'size': len(raw)})
            changes.append({'filename': p, 'status': 'added', 'sha': blob(raw)})
            self.api.routes[f'/repos/{REPO}/git/blobs/{blob(raw)}'] = {'encoding': 'base64', 'content': base64.b64encode(raw).decode(), 'size': len(raw), 'sha': blob(raw)}
        self.pr['changed_files'] = len(changes)
        self.api.routes[f'/repos/{REPO}/pulls/7'] = self.pr
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = changes
        self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1'] = {'truncated': False, 'tree': entries}
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1'] = {'truncated': False, 'tree': []}
        sr = self.manifest['sources'][0]; srepo = sr['repository']
        self.api.routes[f'/repos/{srepo}'] = {'private': False, 'full_name': srepo}
        self.api.routes[f'/repos/{srepo}/git/commits/{SOURCE}'] = {'sha': SOURCE}
        self.api.routes[f'/repos/{srepo}/git/trees/{SOURCE}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': sr['path'], 'mode': '100644', 'type': 'blob', 'size': len(self.source), 'sha': blob(self.source)}]}
        self.api.routes[f'/repos/{srepo}/git/blobs/{blob(self.source)}'] = {'encoding': 'base64', 'size': len(self.source), 'content': base64.b64encode(self.source).decode(), 'sha': blob(self.source)}

    def run_cli(self, extra):
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder) / 'event.json'; f.write_text(json.dumps(self.event))
            env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': REPO, 'GH_TOKEN': 'test-only', **extra}
            with mock.patch.dict(os.environ, env), mock.patch.object(intake, 'API', return_value=self.api), contextlib.redirect_stdout(io.StringIO()) as out:
                code = intake.main(['--event', str(f)])
            return code, out.getvalue()

    def reject(self, msg):
        with self.assertRaisesRegex((ValueError, OSError), msg):
            intake.check(self.event, self.api, REPO)

class R_Refute(Fixture):
    def test_R1_patch_is_equally_green_when_main_moves_after_event_time(self):
        # GITHUB_SHA is frozen at event time; a later push to main is invisible to
        # BOTH checkers (no trigger, and neither reads refs/heads/main).
        self.api.routes[f'/repos/{REPO}/git/trees/{EVENT_TIP}?recursive=1'] = {'truncated': False, 'tree': []}
        self.api.routes[f'/repos/{REPO}/git/ref/heads/main'] = {'object': {'sha': LATER_TIP}}
        self.api.routes[f'/repos/{REPO}/git/trees/{LATER_TIP}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': 'incoming/example/RESULT.md', 'type': 'blob', 'mode': '100644', 'sha': blob(b'landed'), 'size': 6}]}
        code, out = self.run_cli({'GITHUB_SHA': EVENT_TIP})
        self.assertEqual(code, 0); self.assertEqual(json.loads(out)['package'], 'incoming/example')
        self.assertFalse(any('heads/main' in r or LATER_TIP in r for r in self.api.calls))

    def test_R2_post_resolution_keeping_B_bytes_is_refused_by_current_checker(self):
        # After 'merge main into B' resolution, the three-dot diff reports the
        # shared root files as modified (demonstrated with git in mergedemo/).
        rows = self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1']
        rows[0]['status'] = 'modified'
        self.reject('add-only')

    def test_R3_post_resolution_keeping_A_bytes_is_refused_by_current_checker(self):
        # Resolution taking landed bytes: base tree (fresh after synchronize)
        # carries the prefix and only B-only files remain 'added'.
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1']['tree'] = [
            {'path': self.package + 'RESULT.md', 'mode': '100644', 'type': 'blob', 'sha': 'd'*40}]
        self.reject('new package id')

    def test_R4_identical_package_resolution_has_zero_changed_files_and_is_refused(self):
        self.pr['changed_files'] = 0
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = []
        self.reject('changed file count')

if __name__ == '__main__':
    unittest.main()
