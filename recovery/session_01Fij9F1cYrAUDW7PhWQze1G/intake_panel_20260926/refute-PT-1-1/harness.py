"""Re-run harness: binds tests/test_public_intake.py (untouched) to the copy of the checker under ROOT."""
import importlib.util, os, sys
from pathlib import Path
ROOT = Path(os.environ.get('ROOT', Path(__file__).resolve().parent / 'orig'))
_spec = importlib.util.spec_from_file_location('test_public_intake', ROOT / 'tests/test_public_intake.py')
T = importlib.util.module_from_spec(_spec)
sys.modules['test_public_intake'] = T
_spec.loader.exec_module(T)
intake = T.intake
REPO, HEAD, BASE, SOURCE, sha, blob, FakeAPI, IntakeTests = T.REPO, T.HEAD, T.BASE, T.SOURCE, T.sha, T.blob, T.FakeAPI, T.IntakeTests

class Probe(IntakeTests):
    def accept(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        return result
    def add_file(self, relpath, raw=b'x\n'):
        self.raw[relpath] = raw
        self.save()
