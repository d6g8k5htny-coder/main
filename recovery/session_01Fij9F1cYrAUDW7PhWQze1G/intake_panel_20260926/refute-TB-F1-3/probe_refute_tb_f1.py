"""Refutation probes for TB-F1 (skeptic #3, trust-boundary). No network.

Run with INTAKE_CHECKER=<copy of tools/public_intake_check.py>. Works against
both the e495837 (296-line) and the origin/main 1b0d4ee (357-line) checkers;
the fixture serves the PR #153 reachability routes, which are simply unused by
the older checker.

Every probe here asserts that the checker ALREADY fails closed in each state a
same-id PR can be in when GitHub would let a maintainer press Merge, so the
'landed intake is immutable' guarantee has no reachable gap. The one state in
which the current checker prints green (finder's F1: head based on a pre-landing
main, base.sha stale) is either unmergeable on GitHub (add/add conflict) or a
byte-identical no-op; probe_R4 reproduces the finder's fact for the record.
"""
import base64
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import tempfile
from pathlib import Path
from unittest import mock
import unittest

CHECKER = Path(os.environ['INTAKE_CHECKER'])
spec = importlib.util.spec_from_file_location('intake', CHECKER)
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)

REPO = 'd6g8k5htny-coder/main'
MATH = 'd6g8k5htny-coder/Math-'
HEAD = 'a' * 40
BASE = 'b' * 40       # pr.base.sha as the checker reads it (possibly stale, pre-landing)
SOURCE = 'c' * 40
MAIN_TIP = 'f' * 40   # refs/heads/main after the same-id package landed


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def compare(commit):
    return {'status': 'behind', 'ahead_by': 0, 'behind_by': 3, 'merge_base_commit': {'sha': commit}}


class FakeAPI:
    def __init__(self):
        self.routes = {}
        self.calls = []

    def get(self, route):
        self.calls.append(route)
        if route not in self.routes:
            raise AssertionError('Unexpected API request: ' + route)
        return copy.deepcopy(self.routes[route])


class Fixture(unittest.TestCase):
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
                         'sources': [{'repository': MATH, 'path': 'coefficients/example.json',
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
            self.serve_blob(REPO, raw)
        self.pr['changed_files'] = len(changes)
        self.api.routes[f'/repos/{REPO}/pulls/7'] = self.pr
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = changes
        self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1'] = {'truncated': False, 'tree': entries}
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1'] = {'truncated': False, 'tree': []}
        # Source routes, including the PR #153 reachability routes (unused by e495837).
        self.api.routes[f'/repos/{MATH}'] = {'private': False, 'full_name': MATH, 'default_branch': 'main'}
        self.api.routes[f'/repos/{MATH}/git/commits/{SOURCE}'] = {'sha': SOURCE}
        self.api.routes[f'/repos/{MATH}/compare/main...{SOURCE}?per_page=1'] = compare(SOURCE)
        self.api.routes[f'/repos/{MATH}/git/trees/{SOURCE}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': 'coefficients/example.json', 'mode': '100644', 'type': 'blob', 'size': len(self.source), 'sha': blob(self.source)}]}
        self.serve_blob(MATH, self.source)

    def serve_blob(self, repo, raw):
        self.api.routes[f'/repos/{repo}/git/blobs/{blob(raw)}'] = {
            'encoding': 'base64', 'content': base64.b64encode(raw).decode(), 'size': len(raw), 'sha': blob(raw)}

    def landed_entries(self):
        # What refs/heads/main carries once a package with the SAME id has landed:
        # a different RESULT.md, its own IDENTITY.json, and an artifact a.json.
        landed = {'RESULT.md': b'# Landed result\n', 'a.json': b'{"landed":true}\n'}
        manifest = dict(self.manifest, artifacts=[{'path': n, 'bytes': len(r), 'sha256': sha(r)} for n, r in landed.items()])
        landed['IDENTITY.json'] = json.dumps(manifest).encode()
        return {self.package + n: r for n, r in landed.items()}

    def run_check(self):
        return intake.check(self.event, self.api, REPO)

    def reject(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.run_check()

    def run_cli(self, env_extra=None):
        with tempfile.TemporaryDirectory() as folder:
            event_file = Path(folder) / 'event.json'
            event_file.write_text(json.dumps(self.event))
            env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': REPO, 'GH_TOKEN': 'test-only'}
            env.update(env_extra or {})
            with mock.patch.dict(os.environ, env), mock.patch.object(intake, 'API', return_value=self.api), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                code = intake.main(['--event', str(event_file)])
            return code, output.getvalue()


class RefuteTBF1(Fixture):
    def probe_R1_head_updated_with_landed_files_is_refused_even_with_stale_base_tree(self):
        """After 'Update branch' / a conflict-resolution push (the only way GitHub lets an
        out-of-date or conflicting same-id PR reach Merge), the head tree carries the
        landed files. Even if pr.base.sha were still the stale pre-landing commit (tree
        empty, so line 239 is silent), line 242's head-tree/changed-files equality refuses
        the package. This defense does not depend on base freshness at all."""
        landed = self.landed_entries()
        new = self.package + 'b.json'
        raw = b'{"new":true}\n'
        self.serve_blob(REPO, raw)
        head_tree = [{'path': p, 'type': 'blob', 'mode': '100644', 'sha': blob(r), 'size': len(r)} for p, r in landed.items()]
        head_tree.append({'path': new, 'type': 'blob', 'mode': '100644', 'sha': blob(raw), 'size': len(raw)})
        self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1'] = {'truncated': False, 'tree': head_tree}
        # Contributor resolved by keeping main's files and only adding b.json: the
        # 3-dot diff against the new merge base lists exactly one 'added' row.
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = [{'filename': new, 'status': 'added', 'sha': blob(raw)}]
        self.pr['changed_files'] = 1
        # Stale base: tree at pr.base.sha still empty.
        self.assertEqual(self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1']['tree'], [])
        self.reject('package tree does not match changed files')
        self.assertIn(f'/repos/{REPO}/git/trees/{BASE}?recursive=1', self.api.calls)  # base check ran and was silent

    def probe_R2_head_updated_and_keeping_own_bytes_is_refused_as_not_add_only(self):
        """Contributor resolved the add/add conflict in favour of their own bytes: the
        diff against the new merge base reports RESULT.md / IDENTITY.json as modified."""
        rows = self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1']
        rows[0]['status'] = 'modified'
        self.reject('add-only intake')

    def probe_R3_fresh_base_sha_refuses_prefix_including_dir_file_and_symlink_forms(self):
        """When GitHub's merge-state recomputation has refreshed pr.base.sha to the landed
        tip, line 239 refuses; already covered by tests/test_public_intake.py
        test_existing_package_is_immutable, repeated here in the 'landed tip' framing."""
        landed = self.landed_entries()
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1']['tree'] = [
            {'path': p, 'type': 'blob', 'mode': '100644', 'sha': blob(r), 'size': len(r)} for p, r in landed.items()]
        self.reject('landed intake is immutable')

    def probe_R4_finder_fact_reproduced_and_patch_has_the_same_window(self):
        """For the record: with the head based on a pre-landing main and pr.base.sha
        stale, the CURRENT checker prints green and never reads GITHUB_SHA (finder's F1).
        The same fixture also shows the finder's remedy is a snapshot of the same kind:
        a GITHUB_SHA whose tree is empty (landing happened after the event) is green under
        the patched checker too, and nothing re-fires on a base push either way."""
        self.api.routes[f'/repos/{REPO}/git/trees/{MAIN_TIP}?recursive=1'] = {'truncated': False, 'tree': []}
        code, out = self.run_cli({'GITHUB_SHA': MAIN_TIP})
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)['package'], 'incoming/example')
        self.assertEqual(json.loads(out)['scientific_effect'], 'NONE')

    def probe_R5_git_case_C_extra_file_under_landed_id_cannot_have_been_green(self):
        """The only same-id shape that git merges cleanly while changing main's tree is
        'every coinciding path byte-identical plus an extra file' (case C of the git
        experiment). Such a PR cannot pass this checker: if it carries the landed
        IDENTITY.json verbatim, that manifest names the landed a.json it does not ship
        ('duplicate or unknown artifact') or omits the b.json it does ship ('manifest
        must exactly cover'); any manifest that fixes either differs in bytes from the
        landed one and is an add/add conflict (case D). Hence a green same-id PR that
        GitHub will merge is byte-identical to the landed package (case B, a no-op)."""
        landed = self.landed_entries()
        landed_files = {p[len(self.package):]: r for p, r in landed.items()}
        extra = b'{"new":true}\n'
        # C1: identical RESULT.md + identical IDENTITY.json + b.json, without a.json.
        self.files = {'RESULT.md': landed_files['RESULT.md'], 'IDENTITY.json': landed_files['IDENTITY.json'], 'b.json': extra}
        self.install()
        self.reject('duplicate or unknown artifact')
        # C2: the full landed package verbatim plus b.json.
        self.files = dict(landed_files, **{'b.json': extra})
        self.install()
        self.reject('manifest must exactly cover every artifact except itself')
        # B: the landed package verbatim is the only same-id shape that is green, and
        # merging it cannot change main's tree.
        self.files = dict(landed_files)
        self.install()
        self.assertEqual(self.run_check()['package'], 'incoming/example')

def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(RefuteTBF1(n) for n in sorted(n for n in dir(RefuteTBF1) if n.startswith('probe_')))
    return suite


if __name__ == '__main__':
    unittest.main()
