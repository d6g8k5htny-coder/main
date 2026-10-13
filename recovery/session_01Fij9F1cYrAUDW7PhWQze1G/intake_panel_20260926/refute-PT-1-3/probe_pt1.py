"""PT-1 re-run: F1 only, against a fresh origin/main copy (ROOT=orig) and an isolated safe_path patch (ROOT=iso)."""
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



if __name__ == "__main__":
    unittest.main()
