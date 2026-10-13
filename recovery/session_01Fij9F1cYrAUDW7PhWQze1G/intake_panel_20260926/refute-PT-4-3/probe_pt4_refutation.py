"""Skeptic #3 (lens paths-tree) probes for PT-4 against the fresh checker. No network.

Grant the finder's unverified premise -- pr.base.sha lags refs/heads/main and the tip
already holds incoming/example -- and show that every head an outsider could push, and
every merge GitHub could perform, is refused or is a byte-for-byte no-op.
"""
import copy
import json
import unittest
from harness import IntakeTests, REPO, HEAD, BASE, blob, sha

TIP = 'f' * 40  # tip of main holding a landed incoming/example (never consulted by design)


class PT4_StaleBaseShaIsNotAMutationPath(IntakeTests):
    def landed(self):
        return [{'path': self.package + n, 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}
                for n in ('RESULT.md', 'IDENTITY.json', 'output.json')]

    def test_1_finder_observation_reproduces(self):
        # Stale base tree lacks the id, tip has it: the checker is green and never reads TIP.
        self.api.routes[f'/repos/{REPO}/git/trees/{TIP}?recursive=1'] = {'truncated': False, 'tree': self.landed()}
        self.assertEqual(self.run_check()['lane'], 'incoming')
        self.assertNotIn(f'/repos/{REPO}/git/trees/{TIP}?recursive=1', self.api.calls)
        # Such a head is, by git semantics (git_semantics_demo.out), an add/add conflict on
        # IDENTITY.json and RESULT.md against the tip: GitHub reports mergeable=false.

    def test_2_head_carrying_landed_files_beyond_its_diff_is_refused_without_base(self):
        # A head that merged/rebased onto the tip carries the landed files. Even with the
        # base tree stale (empty), the head-tree == changed-rows equality refuses it.
        head = self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1']['tree']
        head.append({'path': self.package + 'landed_by_v1.md', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3})
        self.reject('package tree does not match changed files')
        self.assertEqual(self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1']['tree'], [])

    def test_3_conflict_resolution_head_reports_modified_and_is_refused(self):
        # After 'merge main into branch' or a rebase, merge-base is the tip; the files API
        # reports the package files as modified (git_semantics_demo.out). Refused add-only.
        for row in self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1']:
            row['status'] = 'modified'
        self.reject('add-only')

    def test_4_only_conflict_free_overlap_is_a_byte_identical_noop(self):
        # For git to merge cleanly against a landed package, IDENTITY.json and RESULT.md must
        # be byte-identical. Identical IDENTITY.json pins the artifact set and every digest, so
        # any additional or altered file is refused by the manifest coverage checks.
        self.files['extra.md'] = b'x\n'
        self.install()
        self.reject('exactly cover')
        self.files.pop('extra.md')
        self.files['output.json'] = b'{"illustration":false}\n'  # altered bytes, same manifest
        self.install()
        self.reject('artifact identity mismatch')

    def test_5_base_sha_moving_during_the_run_is_caught_by_snapshot(self):
        original_get = self.api.get
        reads = 0
        def changing_get(route):
            nonlocal reads
            result = original_get(route)
            if route == f'/repos/{REPO}/pulls/7':
                reads += 1
                if reads == 2:
                    result['base']['sha'] = TIP
            return result
        self.api.get = changing_get
        self.reject('changed during verification')

    def test_6_fresh_base_sha_is_refused_by_existing_immutability_test(self):
        # If base.sha does track the tip (or a synchronize refreshes it), the existing
        # test_existing_package_is_immutable path fires.
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1']['tree'] = self.landed()
        self.reject('landed intake is immutable')

    def test_7_reading_the_branch_tip_would_not_close_the_window(self):
        # Sanity for the 'no patch' verdict: a checker that read /git/trees/main at t0 is
        # exactly as stale at merge time t1 as one that read base.sha. The FakeAPI serves a
        # fixed tip; a second landing after the run is invisible to either design. The remedy
        # is 'require branches to be up to date' branch protection (docs/PUBLIC_SHOP_SETUP.md
        # already lists installing protection on main as pending), which forces a synchronize
        # and a re-run whose head then fails test_2 or test_3 above.
        self.api.routes[f'/repos/{REPO}/git/trees/main?recursive=1'] = {'truncated': False, 'tree': []}
        self.assertEqual(self.run_check()['lane'], 'incoming')


if __name__ == '__main__':
    unittest.main()
