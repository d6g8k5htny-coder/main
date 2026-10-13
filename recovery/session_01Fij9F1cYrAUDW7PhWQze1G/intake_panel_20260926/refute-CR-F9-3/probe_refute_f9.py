"""Skeptic re-run of CR-F9 against a fresh copy of tools/public_intake_check.py from origin/main.

INTAKE_MODE=current -> asserts the current checker returns a green 'incoming' result for a
                       RESULT.md made only of invisible characters (the finder's observation).
INTAKE_MODE=patched -> asserts the isolated F9 hunk refuses it and the normal fixture still passes.
The house fixture (FakeAPI, setUp/save/install/run_check/reject) is reused verbatim. No network.
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

MODE = os.environ.get('INTAKE_MODE', 'current')
TREE = Path(__file__).resolve().parent / MODE
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec)
spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve()

INVISIBLE = {
    'bom_only': '﻿',
    'two_zwsp': '​​',
    'nbsp_newline': ' \n',
    'word_joiner': '⁠',
    'ideographic_space': '　',
}


class F9(unittest.TestCase):
    setUp = house.IntakeTests.setUp
    save = house.IntakeTests.save
    install = house.IntakeTests.install
    install_source = house.IntakeTests.install_source
    run_check = house.IntakeTests.run_check
    reject = house.IntakeTests.reject

    def test_result_md_of_only_invisible_characters(self):
        for label, text in INVISIBLE.items():
            with self.subTest(label=label):
                self.raw['RESULT.md'] = text.encode('utf-8')
                self.save()
                if MODE == 'current':
                    result = self.run_check()
                    self.assertEqual(result['lane'], 'incoming')
                    self.assertEqual(result['scientific_effect'], 'NONE')
                else:
                    self.reject('RESULT.md has no visible content')

    def test_single_visible_character_passes_in_both_modes(self):
        # The line the patch draws: 'x' is as useless to a reviewer as a BOM and is accepted either way.
        self.raw['RESULT.md'] = b'x'
        self.save()
        self.assertEqual(self.run_check()['lane'], 'incoming')

    def test_ascii_blank_refused_in_both_modes(self):
        self.raw['RESULT.md'] = b' \n\t'
        self.save()
        self.reject('RESULT.md')

    def test_real_result_md_control(self):
        self.assertEqual(self.run_check()['lane'], 'incoming')


if __name__ == '__main__':
    print(f'[probe] mode={MODE} checker={intake.__file__}', file=sys.stderr)
    unittest.main()
