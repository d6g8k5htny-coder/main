"""Skeptic #2 (refute-PT-4-2) probes for PT-4. No network; fresh checker from origin/main.

Model: v1 of incoming/example landed on main at TIP after PR 7's base.sha (BASE) was recorded.
The finder assumes base.sha lags; every probe below grants that assumption and shows the
checker's verdict at each state an outsider can actually put the PR into.
"""
import copy, hashlib, json, unittest
from harness import IntakeTests, intake, REPO, HEAD, BASE, blob, sha

TIP = 'f' * 40


def v1_entries(package):
    return [{'path': package + n, 'mode': '100644', 'type': 'blob', 'sha': blob(b'v1:' + n.encode()), 'size': 3 + len(n)}
            for n in ('RESULT.md', 'IDENTITY.json', 'output.json')]


class PT4(IntakeTests):
    def setUp(self):
        super().setUp()
        # Tip of main holds v1; the recorded base.sha (BASE) does not. The checker never asks for TIP.
        self.api.routes[f'/repos/{REPO}/git/trees/{TIP}?recursive=1'] = {'truncated': False, 'tree': v1_entries(self.package)}

    def test_1_stale_base_is_green_but_head_is_add_add_conflicting_with_tip(self):
        # Granting the finder's premise: green. But every path in the PR also exists at TIP with
        # a different blob (see git_addadd_demo.sh): GitHub reports the PR as not mergeable, and
        # the merge API answers 405 for a conflicting PR. Nothing lands from this state.
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        tip = {e['path']: e['sha'] for e in v1_entries(self.package)}
        head = {e['path']: e['sha'] for e in self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1']['tree']}
        colliding = {p for p in head if p in tip and tip[p] != head[p]}
        self.assertIn(self.package + 'IDENTITY.json', colliding)
        self.assertIn(self.package + 'RESULT.md', colliding)

    def test_2_resolution_keeping_pr_content_is_refused_before_any_tree_is_read(self):
        # Resolving the conflict (merge main into head, rebase, or a fresh branch from TIP) makes
        # the merge-base hold v1, so the files API reports the package as 'modified'. The add-only
        # check at verify_package:174 fires inside the row loop, BEFORE tree(api, repo, base) at
        # :179 -- so a stale base tree cannot even be consulted in this state.
        for row in self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1']:
            row['status'] = 'modified'
        self.reject('add-only')
        self.assertFalse(any('/git/trees/' in c for c in self.api.calls))

    def test_3_resolution_keeping_landed_content_plus_extra_is_refused(self):
        # Resolution takes v1 for the shared paths (so they drop out of the diff) and keeps one
        # new file. Head tree holds v1's files that are not in the rows: equality check at :183.
        entries = self.api.routes[f'/repos/{REPO}/git/trees/{HEAD}?recursive=1']['tree']
        extra = b'extra\n'
        entries[:] = v1_entries(self.package) + [{'path': self.package + 'extra.md', 'mode': '100644', 'type': 'blob', 'sha': blob(extra), 'size': len(extra)}]
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = [{'filename': self.package + 'extra.md', 'status': 'added', 'sha': blob(extra)}]
        self.pr['changed_files'] = 1
        self.reject('package tree does not match changed files')

    def test_4_resolution_taking_landed_content_entirely_leaves_no_rows(self):
        # Diff empty after taking v1 everywhere: changed_files 0 is refused by snapshot().
        self.pr['changed_files'] = 0
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = []
        self.reject('unsupported changed file count')

    def test_5_superset_package_cannot_share_landed_manifest(self):
        # The only merge-clean overlap with v1 is a byte-identical IDENTITY.json, and that manifest
        # cannot cover an extra file: exact-coverage check at :224.
        self.files['extra.md'] = b'x\n'
        self.install()
        self.reject('exactly cover')

    def test_6_fresh_base_sha_is_caught_by_existing_immutability_check(self):
        # When base.sha is refreshed (synchronize) the existing control test_existing_package_is_immutable applies.
        self.api.routes[f'/repos/{REPO}/git/trees/{BASE}?recursive=1']['tree'] = v1_entries(self.package)
        self.reject('landed intake is immutable')

    def test_7_base_sha_moving_during_the_run_is_caught_by_snapshot_equality(self):
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

    def test_8_consulting_refs_heads_main_would_not_close_the_window(self):
        # Hypothetical variant that reads the live tip instead of base.sha: still a check at event
        # time. A package landing AFTER the run (no re-trigger on base push in public-intake.yml)
        # is invisible to either variant; only 'require branches to be up to date' at merge closes it.
        # Demonstrated by running the checker with TIP lacking v1, then 'landing' v1 at TIP: the
        # earlier green is unchanged because nothing re-runs.
        self.api.routes[f'/repos/{REPO}/git/trees/{TIP}?recursive=1'] = {'truncated': False, 'tree': []}
        self.pr['base']['sha'] = TIP; self.event['pull_request']['base']['sha'] = TIP
        self.assertEqual(self.run_check()['lane'], 'incoming')
        self.api.routes[f'/repos/{REPO}/git/trees/{TIP}?recursive=1']['tree'] = v1_entries(self.package)
        # No event fires; the recorded green stands for both variants. Same property, not a defect of :179.


if __name__ == '__main__':
    unittest.main()
