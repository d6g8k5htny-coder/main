"""Skeptic probe for CR-F5 (lens content-rules). No network. House fixture reused by subclassing.

INTAKE_MODE=current -> fresh copy of origin/main's tools/public_intake_check.py
INTAKE_MODE=patched -> the same file plus ONLY the F5 fragment of the finder's patch (F5-only.patch)

Group A reproduces the finder's observation. Group B shows legitimate submissions the F5
patch would wrongly refuse. Group C is a control over the two packages actually landed
under incoming/.
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
LANDED = Path('/home/user/main-review-refute-CR-F5-1/incoming')


class RefuteF5(house.IntakeTests):
    # Hide the inherited house tests from this runner; only the probes below run.
    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def green(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertEqual(result['scientific_effect'], 'NONE')
        return result

    def expect(self, message):
        """current: green (the finder's observation). patched: refused."""
        if MODE == 'current':
            self.green()
        else:
            self.reject(message)

    # ---- A: the finder's observation, reproduced independently ----------------------------
    def test_A1_bidi_override_in_result_md(self):
        self.raw['RESULT.md'] = 'Scientific effect: \u202eTPECCA\u202c NONE.\n'.encode()
        self.save()
        self.expect('control or bidi')

    def test_A2_terminal_escapes_in_txt(self):
        self.raw['log.txt'] = b'\x1b[2J\x1b[H\x1b]0;pwned\x07plain looking log\n'
        self.save()
        self.expect('control or bidi')

    # ---- B: legitimate content the F5 patch wrongly refuses --------------------------------
    def test_B1_ansi_colored_pytest_capture_txt(self):
        # A contributor pipes `pytest --color=yes` into regeneration/test_run.txt.
        self.raw['regeneration/test_run.txt'] = (
            b'tests/test_side24.py \x1b[32m.\x1b[0m\x1b[32m.\x1b[0m\n'
            b'\x1b[32m============ \x1b[32m\x1b[1m2 passed\x1b[0m\x1b[32m in 0.10s\x1b[0m\x1b[32m ============\x1b[0m\n')
        self.save()
        self.expect('control or bidi')

    def test_B2_rtl_prose_with_right_to_left_mark_md(self):
        # Hebrew sentence ending in a Latin token: editors insert U+200F so the period stays put.
        self.raw['RESULT.md'] = '# \u05ea\u05d5\u05e6\u05d0\u05d4\n\u05d4\u05d7\u05e1\u05dd \u05d4\u05d5\u05d0 SIDE24\u200f.\nScientific effect NONE.\n'.encode()
        self.save()
        self.expect('control or bidi')

    def test_B3_form_feed_page_break_txt(self):
        self.raw['run.txt'] = b'page one\n\x0cpage two\n'
        self.save()
        self.expect('control or bidi')

    # ---- C: both landed packages pass the text branch in both modes (control) --------------
    def test_C_landed_packages_text_branch(self):
        count = 0
        for path in sorted(LANDED.rglob('*')):
            if path.suffix.lower() in intake.TEXT_SUFFIXES:
                raw = path.read_bytes()
                text = raw.decode('utf-8')
                self.assertNotIn('\0', text)
                if MODE == 'patched':
                    self.assertFalse(any((ord(ch) < 32 and ch not in '\t\n\r') or 0x7f <= ord(ch) <= 0x9f
                                         or ch in intake.BIDI_CONTROLS for ch in text), path)
                count += 1
        self.assertGreaterEqual(count, 10)


if __name__ == '__main__':
    print(f'[probe] mode={MODE} checker={intake.__file__}', file=sys.stderr)
    unittest.main(argv=[sys.argv[0], '-v'])
