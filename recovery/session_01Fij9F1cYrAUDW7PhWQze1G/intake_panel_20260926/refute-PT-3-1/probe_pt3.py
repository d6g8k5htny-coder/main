"""Skeptic re-probe of PT-3 (case-only path collisions). No network.

Run:  ROOT=<orig|patched dir> python3 -m pytest -q probe_pt3.py
Expected on orig/: the two 'accepted' probes PASS (defect reproduced), the
boundary probes PASS (they show what an outsider CANNOT do).
Expected on patched/: the two 'accepted' probes FAIL closed; boundary probes still pass.
"""
import unittest
from harness import Probe, REPO, HEAD, BASE, blob


class PT3_Reproduce(Probe):
    def test_a_two_files_differing_only_by_case_accepted(self):
        # Outsider-producible: git stores both; GitHub is case-sensitive.
        self.add_file('result.md', b'shadow\n')
        r = self.accept()
        self.assertEqual(r['artifact_files'], 3)  # RESULT.md, result.md, output.json

    def test_a2_nested_directory_case_variants_accepted(self):
        self.add_file('Plots/a.png', b'\x89PNG\r\n\x1a\n' + b'x')
        self.add_file('plots/a.png', b'\x89PNG\r\n\x1a\n' + b'y')
        self.accept()

    def test_b_landed_uppercase_package_not_seen_as_existing(self):
        # Precondition: 'incoming/Example/' in main -- only reachable by bypassing the
        # checker (id regex is lowercase-only), i.e. a direct push / admin override.
        self.base_tree()['tree'] = [
            {'path': 'incoming/Example/RESULT.md', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}]
        self.accept()


class PT3_Boundaries(Probe):
    """What the current checker already prevents; these bound the severity."""

    def test_outsider_cannot_case_vary_a_legitimately_landed_package(self):
        # A landed package is lowercase by construction; any case variant of its id is
        # refused by the id regex before the base tree is even consulted.
        self.base_tree()['tree'] = [
            {'path': 'incoming/example/RESULT.md', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}]
        for pid in ['Example', 'EXAMPLE', 'exAmple']:
            with self.subTest(pid=pid):
                self.package = f'incoming/{pid}/'
                self.save()
                self.reject('invalid package id')

    def test_lowercase_result_alone_does_not_satisfy_RESULT_md(self):
        # The required-file check is exact-case; a lone 'result.md' is refused.
        self.raw.pop('RESULT.md')
        self.raw['result.md'] = b'# r\n'
        self.save()
        self.reject('RESULT.md and IDENTITY.json required')

    def test_protected_names_are_already_case_folded(self):
        for name in ['status.md', 'Status.MD', 'Claims/x.json', 'proof_index.txt']:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.raw[name] = b'x'
                self.save()
                self.reject('protected')
                self.raw = old

    def test_landed_bytes_never_touched_in_case_variant_scenario(self):
        # In scenario (b) the landed blobs are not in the PR's changed files at all;
        # git-level immutability of 'incoming/Example/RESULT.md' holds.
        self.base_tree()['tree'] = [
            {'path': 'incoming/Example/RESULT.md', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}]
        self.accept()
        self.assertFalse(any(r['filename'].startswith('incoming/Example/') for r in self.route_files()))


if __name__ == '__main__':
    unittest.main()
