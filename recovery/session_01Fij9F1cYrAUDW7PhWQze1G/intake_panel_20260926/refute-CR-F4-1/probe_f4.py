"""CR-F4 re-run. INTAKE_TREE selects e495837 (finding's base) or main (origin/main).
Reuses the house fixture from that tree's tests/test_public_intake.py; no network."""
import importlib.util, os, sys, unittest
from pathlib import Path
TREE = Path(__file__).resolve().parent / os.environ.get('INTAKE_TREE', 'e495837')
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec); spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve(), intake.__file__

class F4(house.IntakeTests):
    # Subclass the whole house fixture so both trees' helpers (install_source on main) are present.

    def test_F4a_out_of_range_float_literal_observed(self):
        # Reports what the checker does rather than asserting a mode; result printed.
        self.raw['output.json'] = b'{"bound": 1e400, "neg": -1e999}\n'
        self.save()
        try:
            result = self.run_check()
            print(f'\n[{TREE.name}] F4a: GREEN lane={result["lane"]}', file=sys.stderr)
        except SystemExit:
            pass
        except Exception as e:
            print(f'\n[{TREE.name}] F4a: exception {e!r}', file=sys.stderr)

    def test_F4a_out_of_range_float_literal_is_refused(self):
        self.raw['output.json'] = b'{"bound": 1e400, "neg": -1e999}\n'
        self.save()
        self.reject('nonfinite')

    def test_F4_literal_infinity_refused_control(self):
        self.raw['output.json'] = b'{"bound": Infinity}\n'
        self.save()
        self.reject('nonfinite')

    def test_F4_ordinary_floats_control(self):
        self.raw['output.json'] = b'{"x": 1.5e-300, "y": 1.7976931348623157e308, "z": 1e308, "n": 123456789012345678901234567890}\n'
        self.save()
        self.assertEqual(self.run_check()['lane'], 'incoming')

    def test_direct_strict_json(self):
        import math
        try:
            v = intake.strict_json(b'{"x": 1e400}')
            print(f'\n[{TREE.name}] strict_json(1e400) -> {v!r}', file=sys.stderr)
        except ValueError as e:
            print(f'\n[{TREE.name}] strict_json(1e400) -> ValueError({e})', file=sys.stderr)

if __name__ == '__main__':
    unittest.main()
