"""Tests for the CBZ-0 gate G0 mutation harness. Scientific effect: NONE.

The harness must not be able to report "all killed" by construction: a mutant
that changes no behavior has to survive, and a real logic mutant has to die.
"""
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cbz_scanner_mutation', ROOT / 'tools/cbz/scanner_mutation.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class Harness(unittest.TestCase):
    def setUp(self):
        self.source = harness.SCANNER.read_text(encoding='utf-8')
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text):
        path = Path(self.tmp.name) / 'mutant.py'
        path.write_text(text, encoding='utf-8')
        return path

    def test_every_target_resolves_to_a_real_change(self):
        cases = harness.mutants(self.source)
        names = [name for name, _ in cases]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(sum(n.startswith('delete-rule:') for n in names), 20)
        self.assertEqual(len(cases) - sum(n.startswith('delete-rule:') for n in names), len(harness.LOGIC))
        self.assertTrue(all(mutated != self.source for _, mutated in cases))

    def test_missing_target_is_an_error_not_a_kill(self):
        with self.assertRaises(RuntimeError):
            harness.mutants(self.source.replace("    return 0 if report['verdict'] == 'PASS' else 1", "    return 9"))

    def test_behavior_neutral_mutant_survives(self):
        neutral = self.source.replace('Exit status: 0 only when', 'Exit status: zero only when', 1)
        self.assertNotEqual(neutral, self.source)
        code, _ = harness.run_suite(self.write(neutral))
        self.assertEqual(code, 0)

    def test_real_mutant_is_killed(self):
        mutated = self.source.replace("        if actual != row['count']:", "        if False:", 1)
        self.assertNotEqual(mutated, self.source)
        code, _ = harness.run_suite(self.write(mutated))
        self.assertNotEqual(code, 0)

    def test_full_gate_kills_every_mutant(self):
        out = Path(self.tmp.name) / 'g0.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code = harness.main(['--json', str(out), '--jobs', '4'])
        self.assertEqual(code, 0)
        self.assertIn('"survived": []', out.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
