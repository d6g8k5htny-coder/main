import importlib.util, tempfile
from pathlib import Path
WT = Path("/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/r789")
spec = importlib.util.spec_from_file_location("g", WT / "cbz/evidence/node-status-freshness/run_mutations.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
TID = 'tests.fixture.Fixture.test_selected'
def run(src):
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as t:
        r = Path(t); (r/'tests').mkdir(); (r/'tests/__init__.py').write_text(''); (r/'tests/fixture.py').write_text(src)
        return [g.qualified_kill(g.run_selected(r, TID, o), TID) for o in (False, True)]
DUNDER = '''import unittest
class _X:
    failureException = AssertionError
    __eq__ = unittest.TestCase.fail
    __hash__ = None
def laundered():
    return _X()
class Fixture(unittest.TestCase):
    def test_selected(self):
        self.assertEqual(laundered(), 1)
'''
SUBJ = '''import unittest
def subject():
    raise AssertionError('subject assertion')
class Fixture(unittest.TestCase):
    def test_selected(self):
        self.assertEqual(subject(), 1)
'''
REENTER = '''import unittest
class Fixture(unittest.TestCase):
    def setUp(self):
        self.helper2 = lambda: self.test_selected()
    def test_selected(self):
        self.n = getattr(self, 'n', 0) + 1
        if self.n == 1: self.helper2()
        else: self.fail('inner')
'''
print('subject raises directly (should refuse):', run(SUBJ))
print('oracle-dunder laundering:', run(DUNDER))
print('re-entry via helper (head should refuse):', run(REENTER))
