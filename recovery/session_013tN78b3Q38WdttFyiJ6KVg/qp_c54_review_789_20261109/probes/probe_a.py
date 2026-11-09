import importlib.util, sys, tempfile, textwrap
from pathlib import Path
WT = Path("/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/r789")
spec = importlib.util.spec_from_file_location("g", WT / "cbz/evidence/node-status-freshness/run_mutations.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
SRC = '''import unittest
class Fixture(unittest.TestCase):
    def setUp(self):
        {setup}
    def tearDown(self):
        {teardown}
    def test_selected(self):
        self.calls += 1
        if self.calls == 2: self.fail('called from teardown')
'''
def run(setup, teardown):
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as t:
        r = Path(t); (r/'tests').mkdir(); (r/'tests/__init__.py').write_text('')
        (r/'tests/fixture.py').write_text(SRC.format(setup=setup, teardown=teardown))
        return [g.qualified_kill(g.run_selected(r, 'tests.fixture.Fixture.test_selected', o), 'tests.fixture.Fixture.test_selected') for o in (False, True)]
print('control (teardown calls body via wrapped hook):', run('self.calls = 0', 'self.test_selected()'))
print('A (setUp rebinds _callTearDown):', run('self.calls = 0; self._callTearDown = self.test_selected', 'pass'))
