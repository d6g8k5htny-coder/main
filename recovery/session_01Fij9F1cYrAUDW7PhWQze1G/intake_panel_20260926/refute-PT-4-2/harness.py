"""Bind the repository's own test fixture to a FRESH copy of the checker from origin/main (fresh/)."""
import importlib.util, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent / 'fresh'
_spec = importlib.util.spec_from_file_location('test_public_intake', ROOT / 'tests/test_public_intake.py')
T = importlib.util.module_from_spec(_spec); sys.modules['test_public_intake'] = T; _spec.loader.exec_module(T)
intake = T.intake
REPO, HEAD, BASE, SOURCE, sha, blob, FakeAPI, IntakeTests = T.REPO, T.HEAD, T.BASE, T.SOURCE, T.sha, T.blob, T.FakeAPI, T.IntakeTests
