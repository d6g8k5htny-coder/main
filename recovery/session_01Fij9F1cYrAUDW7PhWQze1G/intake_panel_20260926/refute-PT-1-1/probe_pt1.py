"""PT-1 re-run: the seven F1 probes, copied verbatim in substance from the finder's probes_findings.py."""
import unittest
from harness import Probe

class F1_ChecklistHostileComponentNamesAccepted(Probe):
    def test_component_longer_than_NAME_MAX_accepted(self):
        self.add_file('a' * 300 + '.md'); self.accept()
    def test_trailing_space_directory_accepted(self):
        self.add_file('sub /x.md'); self.accept()
    def test_space_only_directory_accepted(self):
        self.add_file('sub/ /x.md'); self.accept()
    def test_leading_space_filename_accepted(self):
        self.add_file('sub/ x.md'); self.accept()
    def test_trailing_dot_directory_accepted(self):
        self.add_file('dir./x.md'); self.accept()
    def test_leading_hyphen_filename_accepted(self):
        self.add_file('sub/-rf.md'); self.accept()
    def test_windows_reserved_device_names_accepted(self):
        for name in ['CON.md', 'nul.txt', 'com1.csv', 'Aux.json']:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.add_file(name, b'{"illustration":true}\n')
                self.accept()
                self.raw = old

class ExtraControls(Probe):
    """My own additions: the top-level variants the finder says are already refused, to confirm the boundary of the gap."""
    def test_top_level_leading_hyphen_is_refused_via_manifest_path(self):
        self.add_file('-rf.md'); self.reject('unsafe path')
    def test_top_level_leading_space_is_refused_via_manifest_path(self):
        self.add_file(' x.md'); self.reject('unsafe path')
    def test_255_byte_component_accepted_on_both(self):
        self.add_file('a' * 252 + '.md'); self.accept()

if __name__ == '__main__':
    unittest.main()
