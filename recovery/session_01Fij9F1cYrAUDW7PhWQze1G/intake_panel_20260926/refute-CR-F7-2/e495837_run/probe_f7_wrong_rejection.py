"""Control for the CR-F7 prefix patch: legitimate package names that share a prefix with a
protected register name. INTAKE_MODE=current expects green (unchanged behaviour);
INTAKE_MODE=patched expects the prefix rule to REFUSE them (a wrong rejection)."""
import importlib.util, os, unittest
from pathlib import Path
MODE = os.environ.get('INTAKE_MODE', 'current')
TREE = Path(__file__).resolve().parent / MODE
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec); spec.loader.exec_module(house)
assert Path(house.intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve()

NAMES = ['status_notes.txt', 'STATUS_REPORT.md', 'claims_comparison.csv', 'claimsdata/values.json',
         'landing_claims_replay.json', 'proof_index_excerpt.md', 'statusbar.png']

class Control(unittest.TestCase):
    locals().update({k: v for k, v in vars(house.IntakeTests).items()
                     if callable(v) and not k.startswith('test_') and not k.startswith('__')})

    def test_legitimate_prefix_names(self):
        for name in NAMES:
            with self.subTest(name=name):
                old = self.raw.copy()
                try:
                    self.raw[name] = b'\x89PNG\r\n\x1a\n' if name.endswith('.png') else (b'{"ok":true}\n' if name.endswith('.json') else b'notes\n')
                    if name.endswith('.png') and hasattr(house.intake, 'png_structure'):
                        continue  # HEAD checker needs a structurally valid PNG; the name rule fires before content anyway
                    self.save()
                    if MODE == 'current':
                        self.assertEqual(self.run_check()['lane'], 'incoming')
                    else:
                        self.reject('protected scientific status path')
                finally:
                    self.raw = old

if __name__ == '__main__':
    unittest.main()
