"""Shared probe harness: same FakeAPI/fixture pattern as tests/test_public_intake.py; no network.
INTAKE_MODULE selects the checker copy under test (current/ or patched/)."""
import base64, copy, hashlib, importlib.util, json, os, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = Path(os.environ.get('INTAKE_MODULE', HERE / 'current' / 'public_intake_check.py'))
spec = importlib.util.spec_from_file_location('intake', MODULE)
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)

REPO = 'd6g8k5htny-coder/main'
HEAD = 'a' * 40
BASE = 'b' * 40
SOURCE = 'c' * 40


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def blob_route(raw):
    return {'encoding': 'base64', 'content': base64.b64encode(raw).decode(), 'size': len(raw), 'sha': blob(raw)}


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


class Base(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI()
        self.pr = {'number': 7, 'state': 'open', 'changed_files': 3,
                   'head': {'sha': HEAD, 'repo': {'full_name': 'visitor/main'}},
                   'base': {'sha': BASE, 'ref': 'main', 'repo': {'full_name': REPO}},
                   'labels': []}
        self.event = {'number': 7, 'repository': {'full_name': REPO}, 'pull_request': copy.deepcopy(self.pr)}
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
        self.manifest['artifacts'] = [{'path': n, 'bytes': len(r), 'sha256': sha(r)} for n, r in self.raw.items()]
        self.files = dict(self.raw, **{'IDENTITY.json': json.dumps(self.manifest).encode()})
        self.install()

    def install(self):
        entries, changes = [], []
        for name, raw in self.files.items():
            path = self.package + name
            entries.append({'path': path, 'type': 'blob', 'mode': '100644', 'sha': blob(raw), 'size': len(raw)})
            changes.append({'filename': path, 'status': 'added', 'sha': blob(raw)})
            self.api.routes[f'/repos/{REPO}/git/blobs/{blob(raw)}'] = blob_route(raw)
        self.pr['changed_files'] = len(changes)
        self.api.routes[f'/repos/{REPO}/pulls/7'] = self.pr
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = changes
        self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1'] = {'truncated': False, 'tree': entries}
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1'] = {'truncated': False, 'tree': []}
        for sr in self.manifest['sources']:
            if sr.get('commit') in (HEAD, BASE):
                continue  # the PR's own trees are installed above; a probe that cites them sets routes itself
            self.install_source(sr['repository'], sr['commit'], sr['path'], self.source)

    def install_source(self, source_repo, commit, path, raw, default_branch='main', reachable_from=('main',), branches=('main',)):
        """Serve what GitHub serves for a commit object in the repository network, plus the
        reachability routes (compare/branches) the CURRENT checker never asks for."""
        self.api.routes[f'/repos/{source_repo}'] = {'private': False, 'full_name': source_repo, 'default_branch': default_branch}
        self.api.routes[f'/repos/{source_repo}/git/commits/{commit}'] = {'sha': commit}
        self.api.routes[f'/repos/{source_repo}/git/trees/{commit}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': path, 'mode': '100644', 'type': 'blob', 'size': len(raw), 'sha': blob(raw)}]}
        self.api.routes[f'/repos/{source_repo}/git/blobs/{blob(raw)}'] = blob_route(raw)
        self.api.routes[f'/repos/{source_repo}/branches?per_page=100&page=1'] = [{'name': b, 'commit': {'sha': 'd' * 40}} for b in branches]
        for b in branches:
            if b in reachable_from:
                self.api.routes[f'/repos/{source_repo}/compare/{b}...{commit}?per_page=1'] = {
                    'status': 'behind', 'ahead_by': 0, 'behind_by': 5, 'merge_base_commit': {'sha': commit}}
            else:
                # GitHub: the commit is in the object store (fork network / refs/pull/N/head) but has
                # commits not on this branch: "This commit does not belong to any branch on this repository".
                self.api.routes[f'/repos/{source_repo}/compare/{b}...{commit}?per_page=1'] = {
                    'status': 'diverged', 'ahead_by': 2, 'behind_by': 9, 'merge_base_commit': {'sha': BASE}}

    def run_check(self):
        return intake.check(self.event, self.api, REPO)

    def reject(self, message):
        with self.assertRaisesRegex((ValueError, OSError), message):
            self.run_check()

    def reachability_routes_requested(self):
        return [c for c in self.api.calls if '/compare/' in c or '/branches' in c]
