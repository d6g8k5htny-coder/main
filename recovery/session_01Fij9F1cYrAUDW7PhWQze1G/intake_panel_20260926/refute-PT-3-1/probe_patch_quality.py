import unittest
from harness import Probe

LANDED = [{'path': 'incoming/example/RESULT.md', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}]

class PatchQuality(Probe):
    def test_sibling_ids_accepted_when_example_landed(self):
        for pid in ['example-2', 'exampl', 'examplex', 'ex']:
            with self.subTest(pid=pid):
                self.package = f'incoming/{pid}/'
                self.save()                      # install() resets the base tree; set it AFTER
                self.base_tree()['tree'] = LANDED
                self.assertEqual(self.run_check()['lane'], 'incoming')

    def test_exact_landed_id_still_refused(self):
        self.package = 'incoming/example/'
        self.save()
        self.base_tree()['tree'] = LANDED
        self.reject('new package id')

    def test_kelvin_sign_in_base_tree_folds_to_ascii(self):
        # INFO on the patch: U+212A KELVIN SIGN .lower() == 'k'; a base path containing it
        # would block 'incoming/key/'. Only reachable via a bypass push.
        self.package = 'incoming/key/'
        self.save()
        self.base_tree()['tree'] = [
            {'path': 'incoming/Key/RESULT.md', 'mode': '100644', 'type': 'blob', 'sha': 'd' * 40, 'size': 3}]
        self.assertEqual(self.run_check()['lane'], 'incoming')   # orig: passes; patched: raises
