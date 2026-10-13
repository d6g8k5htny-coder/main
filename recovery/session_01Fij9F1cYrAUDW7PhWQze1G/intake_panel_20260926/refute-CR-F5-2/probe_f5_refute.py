"""Skeptic #2 independent probe for CR-F5. INTAKE_MODE selects <dir>/tools copy.
Section A: direct reproduction (no finder harness) -- bidi and ESC in text -> outcome.
Section B: over-breadth of the proposed patch against plausible legitimate inputs."""
import importlib.util, os, sys, unittest
from pathlib import Path
MODE = os.environ.get('INTAKE_MODE', 'current')
TREE = Path(__file__).resolve().parent / MODE
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec); spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve()

LEGIT = {
    'rtl_marks_lrm_rlm': 'Author: ‏מחבר‎ (Hebrew name with LRM/RLM marks)\nscientific effect NONE\n'.encode(),
    'ansi_colored_pytest_log': b'\x1b[32m3 passed\x1b[0m in 0.12s\n',
    'form_feed_page_break': b'section 1\n\x0csection 2\n',
    'vertical_tab': b'a\x0bb\n',
    'del_char': b'a\x7fb\n',
    'nbsp_and_zwj_emoji': 'x y \U0001F469‍\U0001F4BB\n'.encode(),
    'bom_prefixed_md': '﻿# Result\nNONE\n'.encode(),
}

class Direct(unittest.TestCase):
    setUp = house.IntakeTests.setUp; save = house.IntakeTests.save; install = house.IntakeTests.install
    run_check = house.IntakeTests.run_check; reject = house.IntakeTests.reject

    def outcome(self):
        try:
            r = self.run_check(); return 'GREEN lane=%s' % r['lane']
        except (ValueError, OSError) as e:
            return 'REFUSED: %s' % e

    def test_A_bidi_in_result_md(self):
        self.raw['RESULT.md'] = 'Scientific effect: ‮TPECCA‬ NONE.\n'.encode(); self.save()
        print('\n[A1 bidi RESULT.md]', MODE, '->', self.outcome())

    def test_A_bidi_in_json_string(self):
        self.raw['output.json'] = '{"note": "⁧ACCEPT⁩"}\n'.encode(); self.save()
        print('\n[A2 bidi in JSON string]', MODE, '->', self.outcome())

    def test_A_esc_in_txt(self):
        self.raw['log.txt'] = b'\x1b[2J\x1b[H\x1b]0;pwned\x07plain looking log\n'; self.save()
        print('\n[A3 ESC/OSC log.txt]', MODE, '->', self.outcome())

    def test_A_c1_control_in_csv(self):
        self.raw['t.csv'] = 'a,b\n1,\u0085x\n'.encode(); self.save()
        print('\n[A4 C1 NEL in csv]', MODE, '->', self.outcome())

    def test_B_legit_inputs(self):
        for label, raw in LEGIT.items():
            old = dict(self.raw); self.raw['notes.txt'] = raw; self.save()
            print('\n[B %s]' % label, MODE, '->', self.outcome()); self.raw = old

if __name__ == '__main__':
    unittest.main(verbosity=0)
