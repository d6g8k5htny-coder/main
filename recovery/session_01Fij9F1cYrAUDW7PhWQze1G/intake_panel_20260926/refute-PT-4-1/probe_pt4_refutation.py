"""Skeptic probes for PT-4 (lens paths-tree, refute-PT-4-1). No network; fresh checker copy.

Models the finder's assumed GitHub state -- pr.base.sha lags the tip of main and the tip
already holds incoming/example -- and shows what the checker does at each point where an
outsider could actually reach a merge.
"""
import copy
import unittest
from harness import IntakeTests, REPO, HEAD, BASE, blob

TIP = 'f' * 40  # tip of main, containing a landed incoming/example package v1 (never consulted)


class PT4_StaleBaseCannotReachMerge(IntakeTests):
    def landed_v1_rows(self):
        return [{'path': self.package + n, 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}
                for n in ('RESULT.md', 'IDENTITY.json', 'output.json')]

    def test_A_stale_base_green_is_reproducible_only_while_head_untouched(self):
        # Finder's premise: stale base tree lacks the id; tip has it. Checker passes (green).
        self.api.routes[f'/repos/{REPO}/git/trees/{TIP}?recursive=1'] = {'truncated': False, 'tree': self.landed_v1_rows()}
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertNotIn(f'/repos/{REPO}/git/trees/{TIP}?recursive=1', self.api.calls)
        # But the same head is, by git semantics (see git_conflict_demo*.sh), an add/add conflict
        # against the tip on IDENTITY.json and RESULT.md: GitHub refuses to merge it.

    def test_B_any_conflict_resolution_head_is_refused_add_only(self):
        # After merging main into the branch or rebasing onto it, the merge-base holds v1, so the
        # files API reports the package files as 'modified' (git demo output). Checker refuses.
        rows = self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1']
        for row in rows:
            row['status'] = 'modified'
        self.reject('add-only')

    def test_C_fresh_base_sha_is_refused_by_existing_immutability_check(self):
        # If base.sha does NOT lag (or a synchronize refreshes it), the existing check catches it.
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1']['tree'] = self.landed_v1_rows()
        self.reject('landed intake is immutable')

    def test_D_head_containing_landed_files_outside_diff_is_refused(self):
        # A head that merged main (so its tree holds v1 files not in the PR diff) fails the
        # head-tree == changed-files equality even if the base tree were stale.
        head = self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1']['tree']
        head.append({'path': self.package + 'landed_extra.md', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3})
        self.reject('package tree does not match changed files')

    def test_E_superset_with_identical_manifest_is_refused(self):
        # The only merge-clean overlap is byte-identical IDENTITY.json; then any extra file is uncovered.
        self.files['extra.md'] = b'x\n'
        self.install()
        self.reject('exactly cover')

    def test_F_base_sha_change_during_run_is_caught_by_snapshot(self):
        # The stated 'PR snapshot unchanged across the run' guarantee includes base.sha.
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


if __name__ == '__main__':
    unittest.main()
