"""Re-run harness (skeptic refute-PT-2-2). ROOT selects orig/ or patched/. No network."""
import importlib.util, os, sys
from pathlib import Path
ROOT = Path(os.environ.get('ROOT', Path(__file__).resolve().parent / 'orig'))
_spec = importlib.util.spec_from_file_location('test_public_intake', ROOT / 'tests/test_public_intake.py')
T = importlib.util.module_from_spec(_spec)
sys.modules['test_public_intake'] = T
_spec.loader.exec_module(T)
intake = T.intake
REPO, HEAD, BASE, IntakeTests = T.REPO, T.HEAD, T.BASE, T.IntakeTests
