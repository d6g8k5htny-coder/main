"""Skeptic #3 probe for PT-3 (case-only path collisions). No network.

Run:  ROOT=<orig|patched dir> python3 -m pytest -q probe_pt3_refute.py
Independent fixture in the style of tests/test_public_intake.py (own FakeAPI).
"""
import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest

ROOT = Path(os.environ.get('ROOT', Path(__file__).resolve().parent / 'orig'))
spec = importlib.util.spec_from_file_location('intake', ROOT / 'tools/public_intake_check.py')
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)
PATCHED = 'differing only by case' in (ROOT / 'tools/public_intake_check.py').read_text()

REPO = 'd6g8k5htny-coder/main'
HEAD, BASE, SOURCE = 'a' * 40, 'b' * 40, 'c' * 40


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


class FakeAPI:
    def __init__(self):
        self.routes, self.calls = {}, []

    def get(self, route):
        self.calls.append(route)
        if route not in self.routes:
            raise AssertionError('Unexpected API request: ' + route)
        return copy.deepcopy(self.routes[route])


class CaseProbe(unittest.TestCase):
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
        self.base_entries = []
        self.install()

    def install(self):
        manifest = {'schema': 1, 'scientific_effect': 'NONE', 'review_status': 'REVIEW_REQUIRED',
                    'sources': [{'repository': 'd6g8k5htny-coder/Math-', 'path': 'coefficients/example.json',
                                 'commit': SOURCE, 'sha256': sha(self.source)}],
                    'artifacts': [{'path': n, 'bytes': len(r), 'sha256': sha(r)} for n, r in self.raw.items()]}
        self.files = dict(self.raw, **{'IDENTITY.json': json.dumps(manifest).encode()})
        entries, changes = [], []
        for name, raw in self.files.items():
            path = self.package + name
            entries.append({'path': path, 'type': 'blob', 'mode': '100644', 'sha': blob(raw), 'size': len(raw)})
            changes.append({'filename': path, 'status': 'added', 'sha': blob(raw)})
            self.api.routes[f'/repos/{REPO}/git/blobs/{blob(raw)}'] = {'encoding': 'base64', 'content': base64.b64encode(raw).decode(), 'size': len(raw), 'sha': blob(raw)}
        self.pr['changed_files'] = len(changes)
        self.api.routes[f'/repos/{REPO}/pulls/7'] = self.pr
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = changes
        self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1'] = {'truncated': False, 'tree': entries}
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1'] = {'truncated': False, 'tree': list(self.base_entries)}
        sr = 'd6g8k5htny-coder/Math-'
        self.api.routes[f'/repos/{sr}'] = {'private': False, 'full_name': sr}
        self.api.routes[f'/repos/{sr}/git/commits/{SOURCE}'] = {'sha': SOURCE}
        self.api.routes[f'/repos/{sr}/git/trees/{SOURCE}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': 'coefficients/example.json', 'mode': '100644', 'type': 'blob', 'size': len(self.source), 'sha': blob(self.source)}]}
        self.api.routes[f'/repos/{sr}/git/blobs/{blob(self.source)}'] = {
            'encoding': 'base64', 'size': len(self.source), 'content': base64.b64encode(self.source).decode(), 'sha': blob(self.source)}

    def run_check(self):
        return intake.check(self.event, self.api, REPO)

    def reject(self, message):
        with self.assertRaisesRegex((ValueError, OSError), message):
            self.run_check()

    # --- (a) within-package case-only collision -------------------------------------
    def test_a_result_md_and_result_lower_md_in_one_package(self):
        self.raw['result.md'] = b'shadow\n'
        self.install()
        if PATCHED:
            self.reject('differing only by case')
        else:
            result = self.run_check()
            self.assertEqual(result['lane'], 'incoming')
            self.assertEqual(result['artifact_files'], 3)  # both files manifest-covered and digest-checked

    def test_a_exact_bytes_replay_on_case_folded_view_detects_divergence(self):
        # Simulate a case-insensitive checkout: git writes index entries in byte order,
        # so the later entry ('result.md' > 'RESULT.md') overwrites the same on-disk file.
        # CONTRIBUTING step 6: the reviewer checks 'the content and exact bytes'. Replaying
        # the manifest on that view fails for one of the two entries whenever bytes differ.
        self.raw['result.md'] = b'shadow\n'
        self.install()
        manifest = json.loads(self.files['IDENTITY.json'])
        disk = {}
        for name in sorted(self.files):  # git index order (bytewise)
            disk[name.lower()] = self.files[name]
        mismatches = [a['path'] for a in manifest['artifacts'] if sha(disk[a['path'].lower()]) != a['sha256']]
        self.assertEqual(mismatches, ['RESULT.md'])
        # Identical bytes under both names: nothing is hidden from the on-disk view at all.
        self.raw['result.md'] = self.raw['RESULT.md']
        self.install()
        manifest = json.loads(self.files['IDENTITY.json'])
        disk = {name.lower(): self.files[name] for name in sorted(self.files)}
        self.assertFalse([a['path'] for a in manifest['artifacts'] if sha(disk[a['path'].lower()]) != a['sha256']])

    # --- (b) landed 'incoming/Example/' versus new 'incoming/example/' --------------------
    def test_b_uppercase_landed_dir_not_treated_as_existing_package(self):
        self.base_entries = [{'path': 'incoming/Example/RESULT.md', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}]
        self.install()
        if PATCHED:
            self.reject('landed intake is immutable')
        else:
            self.assertEqual(self.run_check()['lane'], 'incoming')

    def test_b_uppercase_id_can_never_pass_intake_itself(self):
        # 'incoming/Example/' is not reachable through the guarded lane from a fork ...
        self.package = 'incoming/Example/'
        self.install()
        self.reject('invalid package id')
        # ... nor through the same-repository engineering lane: any incoming/ path routes to intake.
        self.pr['head']['repo']['full_name'] = REPO
        self.install()
        self.reject('invalid package id')
        # So a landed 'incoming/Example/' presupposes a bypass of the check (direct push / merge-on-red).

    def test_b_exact_case_landed_package_is_already_covered(self):
        self.base_entries = [{'path': 'incoming/example/old.txt', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}]
        self.install()
        self.reject('landed intake is immutable')

    # --- patch side effects: legitimate sibling ids still accepted ------------------------
    def test_patch_keeps_sibling_ids_accepted(self):
        for landed in ['incoming/example-2/RESULT.md', 'incoming/exampl/RESULT.md', 'incoming/EXAMPLE-2/RESULT.md']:
            with self.subTest(landed=landed):
                self.base_entries = [{'path': landed, 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}]
                self.install()
                self.assertEqual(self.run_check()['lane'], 'incoming')

    def test_patch_keeps_mixed_case_distinct_names_accepted(self):
        self.raw['Plot-Notes.txt'] = b'notes\n'
        self.raw['plot-notes-2.txt'] = b'notes\n'
        self.install()
        self.assertEqual(self.run_check()['lane'], 'incoming')


if __name__ == '__main__':
    unittest.main()
