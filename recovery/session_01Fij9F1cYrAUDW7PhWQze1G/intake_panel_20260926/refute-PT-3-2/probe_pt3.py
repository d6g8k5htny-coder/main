"""Skeptic #2 (lens paths-tree) re-probe of finding PT-3: case-only path collisions.

Run:  ROOT=<orig|patched dir> python3 -m pytest -q probe_pt3.py
Style follows tests/test_public_intake.py: a FakeAPI serving a dict of routes; no network.
Tests marked 'accepted' PASS against the current checker and document the acceptance the
finder reports; tests marked 'control' document what the checker already refuses and must
stay green under the finder's patch too.
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

REPO = 'd6g8k5htny-coder/main'
HEAD = 'a' * 40
BASE = 'b' * 40
SOURCE = 'c' * 40


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


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


class CaseCollisionProbe(unittest.TestCase):
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
        self.api.routes.setdefault(f'/repos/{REPO}/git/trees/{BASE}?recursive=1', {'truncated': False, 'tree': []})
        sr = self.manifest['sources'][0]
        src = sr['repository']
        self.api.routes[f'/repos/{src}'] = {'private': False, 'full_name': src}
        self.api.routes[f'/repos/{src}/git/commits/{sr["commit"]}'] = {'sha': sr['commit']}
        self.api.routes[f'/repos/{src}/git/trees/{sr["commit"]}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': sr['path'], 'mode': '100644', 'type': 'blob', 'size': len(self.source), 'sha': blob(self.source)}]}
        self.api.routes[f'/repos/{src}/git/blobs/{blob(self.source)}'] = {
            'encoding': 'base64', 'size': len(self.source), 'content': base64.b64encode(self.source).decode(), 'sha': blob(self.source)}

    def run_check(self):
        return intake.check(self.event, self.api, REPO)

    def reject(self, message):
        with self.assertRaisesRegex((ValueError, OSError), message):
            self.run_check()

    # --- (a) accepted: two files in one package differing only by case ---------------------
    def test_accepted_result_md_and_result_lower_md_in_one_package(self):
        self.raw['result.md'] = b'shadow content that differs from RESULT.md\n'
        self.save()
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertEqual(result['artifact_files'], 3)
        # Both blobs were actually fetched and byte-verified; nothing escaped coverage.
        fetched = {c.rsplit('/', 1)[1] for c in self.api.calls if f'/repos/{REPO}/git/blobs/' in c}
        self.assertEqual(fetched, {blob(r) for r in self.files.values()})

    def test_accepted_case_variant_directory_inside_package(self):
        self.raw['Plots/a.png'] = b'\x89PNG\r\n\x1a\n' + b'0' * 8
        self.raw['plots/a.png'] = b'\x89PNG\r\n\x1a\n' + b'1' * 8
        self.save()
        self.assertEqual(self.run_check()['lane'], 'incoming')

    # --- (b) accepted: base holds incoming/Example/, PR adds incoming/example/ -----------
    def test_accepted_new_lowercase_id_when_base_holds_case_variant_directory(self):
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': 'incoming/Example/RESULT.md', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}]}
        self.assertEqual(self.run_check()['lane'], 'incoming')

    # --- controls: what the checker already refuses, independent of case ----------------
    def test_control_exact_case_landed_package_is_immutable(self):
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': 'incoming/example/old.txt', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}]}
        self.reject('new package')

    def test_control_uppercase_package_id_cannot_be_created_through_the_lane(self):
        self.package = 'incoming/Example/'
        self.save()
        self.reject('invalid package id')

    def test_control_protected_names_are_already_case_insensitive(self):
        for name in ['status.md', 'Status.txt', 'proof_index.md', 'landing_claims.json', 'Claims/graph.json', 'CLAIMS/x.json']:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.raw[name] = b'x'
                self.save()
                self.reject('protected')
                self.raw = old

    def test_control_case_variant_of_required_result_file_does_not_satisfy_requirement(self):
        self.raw['result.md'] = self.raw.pop('RESULT.md')
        self.save()
        self.reject('RESULT.md and IDENTITY.json required')

    def test_control_sibling_ids_that_share_a_prefix_still_accepted(self):
        for landed in ['incoming/example-2/RESULT.md', 'incoming/exampl/RESULT.md', 'incoming/examples/RESULT.md']:
            with self.subTest(landed=landed):
                self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1'] = {'truncated': False, 'tree': [
                    {'path': landed, 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}]}
                self.assertEqual(self.run_check()['lane'], 'incoming')


if __name__ == '__main__':
    unittest.main()
