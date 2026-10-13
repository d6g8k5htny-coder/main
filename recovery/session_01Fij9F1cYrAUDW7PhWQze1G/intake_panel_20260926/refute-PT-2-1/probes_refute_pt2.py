"""Skeptic probes for PT-2 (MAX_FILES cap). Run: ROOT=<orig|patched_minimal> python3 -m pytest -q probes_refute_pt2.py
Boundary behaviour of the changed_files cap; no network. FakeAPI serves a dict of routes."""
import os
import unittest
from harness import Probe, REPO

CAP = 3000 if 'patched' in os.environ.get('ROOT', '') else 500


class PT2Boundary(Probe):
    def serve_engineering(self, count):
        self.pr['head']['repo']['full_name'] = REPO
        self.pr['changed_files'] = count
        rows = [{'filename': f'registers/json/item{i}.json', 'status': 'modified', 'sha': 'd' * 40} for i in range(count)]
        for page in range(1, (count + 99) // 100 + 1):
            self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page={page}'] = rows[(page-1)*100:page*100]

    def test_exactly_cap_is_green_maintainer_lane(self):
        self.serve_engineering(CAP)
        self.assertEqual(self.run_check()['lane'], 'maintainer-engineering')

    def test_cap_plus_one_is_rejected(self):
        self.serve_engineering(CAP + 1)
        self.reject('unsupported changed file count')

    def test_finders_own_3879_example_rejected_under_both_caps(self):
        # The finder cites a 3879-file commit as motivating practice; MAX_FILES=3000 does not admit it,
        # and the API's 3000-file listing ceiling could not serve it anyway.
        self.serve_engineering(3879)
        self.reject('unsupported changed file count')

    def test_fork_pr_with_501_files_is_rejected_either_way_so_no_legitimate_package_is_lost(self):
        # Outsider path: a fork PR is always the intake lane, bounded by MAX_PACKAGE_FILES=50.
        count = 501
        self.pr['changed_files'] = count
        rows = [{'filename': f'incoming/big/f{i}.md', 'status': 'added', 'sha': 'd' * 40} for i in range(count)]
        for page in range(1, (count + 99) // 100 + 1):
            self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page={page}'] = rows[(page-1)*100:page*100]
        self.reject('unsupported changed file count|too many package files')

    def test_rejection_is_before_any_file_listing_request(self):
        self.serve_engineering(CAP + 1)
        self.reject('unsupported changed file count')
        self.assertFalse(any('/files?' in c for c in self.api.calls))
        self.assertEqual(self.api.calls, [f'/repos/{REPO}/pulls/7'])


if __name__ == '__main__':
    unittest.main()
