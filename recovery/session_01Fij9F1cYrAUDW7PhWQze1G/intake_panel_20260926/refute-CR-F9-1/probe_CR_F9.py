"""Skeptic re-run of CR-F9 against a fresh origin/main checker. No network.

INTAKE_MODE=current  -> asserts the outcome the unpatched checker produces.
INTAKE_MODE=patched  -> asserts the outcome of the minimal F9-only patch
                        (refute-CR-F9-1-minimal.patch) applied to that same copy.
Methods ending in _control must pass identically in BOTH modes.

The fixture (FakeAPI, setUp/save/install/run_check/reject) is the house one from
<MODE>/tests/test_public_intake.py, a byte-exact copy of origin/main's file.
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

MODE = os.environ.get('INTAKE_MODE', 'current')
HERE = Path(__file__).resolve().parent
TREE = HERE / MODE
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec)
spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve()

# The finder's four cases, spelled out so no invisible character hides in this source.
INVISIBLE_ONLY = {
    'U+FEFF BOM only': '\ufeff',
    'two U+200B zero-width spaces': '\u200b\u200b',
    'U+00A0 NBSP then LF': '\u00a0\n',
    'U+2060 word joiner': '\u2060',
}


class SkepticF9(unittest.TestCase):
    # Copy the house fixture by name; origin/main's fixture added install_source.
    for _name in ('setUp', 'save', 'install', 'install_source', 'run_check', 'reject'):
        if hasattr(house.IntakeTests, _name):
            locals()[_name] = getattr(house.IntakeTests, _name)
    del _name

    def green(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertEqual(result['scientific_effect'], 'NONE')
        return result

    def wrong_green_now_refused(self, message):
        if MODE == 'current':
            return self.green()
        self.reject(message)

    # --- the claimed defect -----------------------------------------------------------
    def test_F9_result_md_of_only_invisible_characters(self):
        for label, text in INVISIBLE_ONLY.items():
            with self.subTest(label=label):
                raw = text.encode('utf-8')
                self.assertTrue(raw.strip(), 'precondition: bytes.strip() leaves it non-empty')
                self.raw['RESULT.md'] = raw
                self.save()
                self.wrong_green_now_refused('no visible content')

    # --- what the existing check actually does (both modes refuse) --------------------
    def test_F9_ascii_empty_or_whitespace_only_refused_control(self):
        for raw in [b'', b' \n\t\r\n']:
            with self.subTest(raw=raw):
                self.raw['RESULT.md'] = raw
                self.save()
                self.reject('RESULT.md')

    # --- content adequacy is not a checker guarantee in either mode -------------------
    def test_F9_one_character_result_md_passes_in_both_modes_control(self):
        self.raw['RESULT.md'] = b'x'
        self.save()
        self.green()

    # --- no wrong rejection of legitimate packages ------------------------------------
    def test_F9_landed_result_md_bytes_pass_control(self):
        for name in ['landed_RESULT.md', 'landed_RESULT_chart.md']:
            with self.subTest(name=name):
                self.raw['RESULT.md'] = (HERE / name).read_bytes()
                self.save()
                self.green()

    def test_F9_bom_prefixed_and_non_ascii_content_pass_control(self):
        for label, text in {'BOM + content': '\ufeff# Result\nscientific effect NONE\n',
                            'CJK and emoji only': '数学 🙂\n',
                            'NBSP-indented content': '\u00a0\u00a0Result: NONE\n'}.items():
            with self.subTest(label=label):
                self.raw['RESULT.md'] = text.encode('utf-8')
                self.save()
                self.green()


if __name__ == '__main__':
    print(f'[skeptic probe] mode={MODE} checker={intake.__file__}', file=sys.stderr)
    unittest.main()
