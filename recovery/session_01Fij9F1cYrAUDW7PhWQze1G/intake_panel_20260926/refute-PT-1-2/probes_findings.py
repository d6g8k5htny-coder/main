"""Probes that PASS against the current checker while demonstrating an unwanted outcome.

Run:  ROOT=<orig|patched dir> python3 -m pytest -q probes_findings.py
Expected: all pass on orig/ (defect demonstrated); the patched copy makes them fail
(the checker now fails closed / accepts the legitimate PR).
"""
import unittest
from harness import Probe, REPO, HEAD, BASE, blob


class F1_ChecklistHostileComponentNamesAccepted(Probe):
    """safe_path accepts components that cannot be checked out on common filesystems."""

    def test_component_longer_than_NAME_MAX_accepted(self):
        # ext4/APFS/NTFS: NAME_MAX is 255 bytes; git checkout fails with ENAMETOOLONG.
        self.add_file('a' * 300 + '.md')
        self.accept()

    def test_trailing_space_directory_accepted(self):
        self.add_file('sub /x.md')          # NTFS cannot create 'sub '
        self.accept()

    def test_space_only_directory_accepted(self):
        # A top-level ' ' is refused by the manifest-path first-character rule; nested it is not.
        self.add_file('sub/ /x.md')
        self.accept()

    def test_leading_space_filename_accepted(self):
        self.add_file('sub/ x.md')
        self.accept()

    def test_trailing_dot_directory_accepted(self):
        self.add_file('dir./x.md')          # NTFS strips trailing dots: 'dir.' == 'dir'
        self.accept()

    def test_leading_hyphen_filename_accepted(self):
        # '-rf.md' at top level is refused by the manifest-path first-character rule; nested it is not.
        self.add_file('sub/-rf.md')         # CLI option-injection hazard for reviewers
        self.accept()

    def test_windows_reserved_device_names_accepted(self):
        for name in ['CON.md', 'nul.txt', 'com1.csv', 'Aux.json']:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.add_file(name, b'{"illustration":true}\n')
                self.accept()
                self.raw = old


class F2_MaxFilesCapRejectsLargeEngineeringPR(Probe):
    """MAX_FILES=500 fails closed on a same-repo engineering PR with 501 changed files.
    Repository history contains commits touching 541 and 3879 files (registers regeneration)."""

    def test_501_file_maintainer_pr_rejected(self):
        self.pr['head']['repo']['full_name'] = REPO
        self.pr['changed_files'] = 501
        rows = [{'filename': f'registers/json/item{i}.json', 'status': 'modified', 'sha': 'd' * 40} for i in range(501)]
        for page in range(1, 7):
            self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page={page}'] = rows[(page-1)*100:page*100]
        self.reject('unsupported changed file count')


class F3_CaseFoldCollisionsAccepted(Probe):
    """Exact-case comparisons only; case-insensitive checkouts (macOS/Windows) merge the entries."""

    def test_landed_package_differing_only_by_case_not_treated_as_existing(self):
        # 'incoming/Example/' can only exist in main via a bypass (id regex is lowercase),
        # i.e. a maintainer mistake; branch protection is documented as not installed.
        self.base_tree()['tree'] = [
            {'path': 'incoming/Example/RESULT.md', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}]
        self.accept()

    def test_two_package_files_differing_only_by_case_accepted(self):
        self.add_file('result.md', b'shadow\n')
        self.accept()


class F4_BaseShaNotBranchTip(Probe):
    """INFO only: the immutability check reads the tree of pr.base.sha and never the tip of main.
    GitHub semantics assumption (not verifiable offline): base.sha can lag the branch tip."""

    def test_branch_tip_never_consulted(self):
        self.accept()
        self.assertFalse(any('/git/trees/main' in c or '/git/ref' in c or '/branches/' in c for c in self.api.calls))


if __name__ == '__main__':
    unittest.main()
