"""Probe harness: reuse the repository's own FakeAPI/fixture from tests/test_public_intake.py.

ROOT env var selects which copy (orig/ or patched/) of tools/public_intake_check.py
the imported test module binds to. No network.
"""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('ROOT', Path(__file__).resolve().parent / 'orig'))
_spec = importlib.util.spec_from_file_location('test_public_intake', ROOT / 'tests/test_public_intake.py')
T = importlib.util.module_from_spec(_spec)
sys.modules['test_public_intake'] = T
_spec.loader.exec_module(T)
intake = T.intake
REPO, HEAD, BASE, SOURCE, sha, blob, FakeAPI, IntakeTests = T.REPO, T.HEAD, T.BASE, T.SOURCE, T.sha, T.blob, T.FakeAPI, T.IntakeTests


class Probe(IntakeTests):
    """Same fixture as the repository tests; helper to relocate/rename files."""

    def accept(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        return result

    def add_file(self, relpath, raw=b'x\n'):
        self.raw[relpath] = raw
        self.save()

    def route_files(self):
        return self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1']

    def head_tree(self):
        return self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1']

    def base_tree(self):
        return self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1']
