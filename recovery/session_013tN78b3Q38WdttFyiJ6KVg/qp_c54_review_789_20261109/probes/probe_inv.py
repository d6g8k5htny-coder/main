import importlib.util, sys, tempfile
from pathlib import Path
W = Path("/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/r789")
spec = importlib.util.spec_from_file_location("origin", W / "tests/cbz_inventory_origin.py")
o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)
FX = {
 'setup_via_callTestMethod': '''import unittest
class F(unittest.TestCase):
    def setUp(self):
        self._callTestMethod(getattr(self, self._testMethodName))
    def test_it(self):
        self.fail('body via setUp _callTestMethod')
''',
 'genuine_body_failure': '''import unittest
class F(unittest.TestCase):
    def test_it(self):
        self.assertEqual(1, 2)
''',
}
for name, src in FX.items():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as t:
        r = Path(t); (r/'fx.py').write_text(src)
        tid = 'fx.F.test_it'
        res = o.run(r, tid, [tid], tid)
        print(name, res["classification"], [ (e["authorized"], [f["name"] for f in e["frames"]][-4:]) for e in res.get("events",[])])
