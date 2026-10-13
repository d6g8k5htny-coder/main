"""Refutation probe for CR-F9. TREE=<abs dir with tools/ and tests/>; PATCHED=1 when that tree
carries the F9 hunk. Subclasses the house fixture so it runs at e495837 and at dbb9dcf alike."""
import importlib.util, os, sys, unicodedata, unittest
from pathlib import Path
TREE = Path(os.environ['TREE']); PATCHED = os.environ.get('PATCHED') == '1'
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec); spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve()
LANDED = Path('/home/user/main-review-refute-CR-F9-2/incoming')

FINDER_CASES = ['﻿', '​​', '\xa0\n', '⁠']          # the four CR-F9 subtests
UNCAUGHT = {  # visually empty to a reader/renderer, yet NOT Cf and NOT str.isspace(): the F9 hunk keeps these green
    'U+2800 braille blank (So)': '⠀', 'U+3164 hangul filler (Lo)': 'ㅤ', 'U+115F choseong filler (Lo)': 'ᅟ',
    'U+FFA0 halfwidth filler (Lo)': 'ﾠ', 'U+034F CGJ (Mn)': '͏', 'U+0301 lone combining acute (Mn)': '́',
    'U+FE0F lone variation selector (Mn)': '️', 'markdown html comment': '<!-- -->\n',
    'markdown link-label comment': '[//]: # (.)\n', 'html nbsp entity': '&nbsp;\n', 'single period': '.\n'}

class F9(house.IntakeTests):
    def result_md(self, text):
        self.raw['RESULT.md'] = text.encode(); self.save()
    def test_finder_cases(self):
        for t in FINDER_CASES:
            with self.subTest(text=repr(t)):
                self.result_md(t)
                if PATCHED: self.reject('no visible content')
                else: self.assertEqual(self.run_check()['lane'], 'incoming')
    def test_uncaught_by_f9_hunk_green_in_both_modes(self):
        for label, t in UNCAUGHT.items():
            with self.subTest(case=label, cats=[unicodedata.category(c) for c in t]):
                self.result_md(t)
                self.assertEqual(self.run_check()['lane'], 'incoming')
    def test_landed_result_md_green_in_both_modes(self):
        for p in sorted(LANDED.glob('*/RESULT.md')):
            with self.subTest(file=p.parent.name):
                self.raw['RESULT.md'] = p.read_bytes(); self.save()
                self.assertEqual(self.run_check()['lane'], 'incoming')

if __name__ == '__main__':
    print(f'[probe] tree={TREE.name} patched={PATCHED} checker={intake.__file__}', file=sys.stderr)
    unittest.main(argv=[sys.argv[0], 'F9'])
