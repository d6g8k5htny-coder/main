"""The runner must execute source mutants, and cannot infer a kill from broken syntax."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
try:
    import mutation_test
except ImportError:
    mutation_test=None


class MutationHarnessRules(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(mutation_test,'mutation runner implementation is missing')

    def test_unchanged_checker_survives_and_assertion_failure_kills_real_mutant(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            survived=mutation_test.run_one('identity','source_checks.py',lambda s:s,out)
            self.assertEqual(survived['outcome'],'SURVIVED',survived)
            killed=mutation_test.run_one('drop-placeholders','source_checks.py',
                lambda s:s.replace("placeholders = {'sorry', 'admit', 'sorryAx', 'admitGoal', 'mkSorry'}","placeholders = set()"),out)
            self.assertEqual(killed['outcome'],'KILLED',killed)
            self.assertGreater(killed['assertion_failures'],0)
            self.assertEqual(killed['errors'],0)
            self.assertTrue((out/'drop-placeholders.log').is_file())

    def test_invalid_checker_syntax_is_invalid_mutant_not_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=mutation_test.run_one('broken-syntax','source_checks.py',lambda s:'def :',Path(tmp))
            self.assertEqual(r['outcome'],'INVALID',r)

    def test_mutant_that_only_crashes_is_not_counted_as_behavioral_kill(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=mutation_test.run_one('crash','source_checks.py',lambda s:'raise RuntimeError("crash")\n'+s,Path(tmp))
            self.assertEqual(r['outcome'],'ERROR',r)


if __name__=='__main__':
    unittest.main()
