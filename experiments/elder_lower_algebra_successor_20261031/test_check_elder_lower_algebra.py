"""Negative controls for check_elder_lower_algebra.py, invoked through its CLI.

Run: python3 -m unittest -v test_check_elder_lower_algebra (from this directory).
Credit 0 · sci NONE. A pass shows the checker accepts the displayed algebra and
rejects each named weakening; it says nothing about the probabilistic steps.
"""
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK = HERE / "check_elder_lower_algebra.py"
MUTANTS = {
    "cubic_coeff": "e6_g1_identity",
    "e6_g3_linear": "e6_g3_identity",
    "g3_prime_bound": "g3_prime_lower_bound",
    "margin_eps": "margin_end_above_1/4",
    "e7_target_beta": "e7_targets_recover_G",
    "overlap_coeff": "overlap_integral_8_1",
}


def run(*args):
    return subprocess.run([sys.executable, "-I", str(CHECK), *args],
                          capture_output=True, text=True)


class ElderLowerAlgebra(unittest.TestCase):
    def test_baseline_passes(self):
        r = run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(": OK", r.stdout)

    def test_every_mutant_is_rejected_on_its_intended_check(self):
        for name, expected in MUTANTS.items():
            with self.subTest(mutant=name):
                r = run("--mutate", name)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertIn(expected, r.stdout)

    def test_mutant_list_matches_checker(self):
        sys.path.insert(0, str(HERE))
        try:
            import check_elder_lower_algebra as m
        finally:
            sys.path.pop(0)
        self.assertEqual(set(m.MUTANTS), set(MUTANTS))


if __name__ == "__main__":
    unittest.main()
