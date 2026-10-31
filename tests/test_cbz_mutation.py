"""Tests for the CBZ-0 gate G0 mutation engine. Scientific effect: NONE.

The engine must not be able to report "all killed" by construction: for every
certifying tool, a mutant that changes no behavior has to survive and a real
logic mutant has to die. The full gate (every mutant of every tool) runs as
`python3 -B tools/cbz/mutation.py` in the landing checks; it is too slow for
the unit suite.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cbz_mutation', ROOT / 'tools/cbz/mutation.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

# A behavior-neutral edit and a real logic mutant per target.
CASES = {
    'layer_a_scan': ('Exit status: 0 only when', 'Exit status: zero only when',
                     "        if actual != row['count']:", "        if False:"),
    'fuzz_bounds': ('This layer falsifies; it never certifies.', 'This layer only falsifies; it never certifies.',
                    "        return max(a - b, 0)", "        return a - b"),
    'drift_crawler': ('Exit status 0 only when nothing blocking is found.', 'Exit status zero only when nothing blocks.',
                      "    if len(mathlib) > 1:", "    if False:"),
}


class Engine(unittest.TestCase):
    def source(self, target):
        return (engine.TOOLS / engine.TARGETS[target]['file']).read_text(encoding='utf-8')

    def run_text(self, target, text):
        with tempfile.TemporaryDirectory() as tmp:
            return engine.run_suite(target, engine.stage(tmp, target, text))[0]

    def test_every_target_resolves_to_real_changes(self):
        for target in engine.TARGETS:
            with self.subTest(target=target):
                source = self.source(target)
                cases = engine.mutants(target, source)
                self.assertGreaterEqual(len(cases), len(engine.TARGETS[target]['logic']))
                self.assertTrue(all(mutated != source for _, mutated in cases))

    def test_scanner_rule_deletions_are_generated(self):
        names = [n for n, _ in engine.mutants('layer_a_scan', self.source('layer_a_scan'))]
        self.assertGreaterEqual(sum(n.startswith('delete-rule:') for n in names), 20)

    def test_missing_target_is_an_error_not_a_kill(self):
        for target, (_, _, real_old, _) in CASES.items():
            with self.subTest(target=target):
                with self.assertRaises(RuntimeError):
                    engine.mutants(target, self.source(target).replace(real_old, '    pass  # gone'))

    def test_neutral_mutants_survive_and_real_mutants_die(self):
        for target, (neutral_old, neutral_new, real_old, real_new) in CASES.items():
            with self.subTest(target=target):
                source = self.source(target)
                self.assertEqual(source.count(neutral_old), 1)
                self.assertEqual(self.run_text(target, source.replace(neutral_old, neutral_new)), 0)
                self.assertEqual(source.count(real_old), 1)
                self.assertNotEqual(self.run_text(target, source.replace(real_old, real_new)), 0)


if __name__ == '__main__':
    unittest.main()
