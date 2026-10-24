"""S2 (main#475): raw OSError on output scandir / upload-stage subdir mkdir must exit 74, not 1."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tools import periodic_pilot_custody as custody
from tests.test_periodic_pilot_custody import SUCCESS

REPO = Path(__file__).resolve().parents[1]
# Child runs the real CLI `seal` with one injected OSError; an uncaught OSError exits 1.
FAULT = r"""
import os, pathlib, sys
sys.path.insert(0, sys.argv[2])
from tools import periodic_pilot_custody as custody
mode = sys.argv[1]
if mode == 'scandir':
    def bad_scandir(*a, **k):
        raise PermissionError(13, 'injected scandir failure')
    custody.os.scandir = bad_scandir
else:
    real = pathlib.Path.mkdir
    def bad_mkdir(self, *a, **k):
        if self.parent.name == 'upload' and self.name in ('logs', 'outputs'):
            raise OSError(28, 'injected stage mkdir failure')
        return real(self, *a, **k)
    pathlib.Path.mkdir = bad_mkdir
sys.argv = ['periodic_pilot_custody.py', 'seal', '--job-status', 'success', '--pilot-outcome', 'success']
raise SystemExit(custody.main())
"""


class RawOSErrorExitTests(unittest.TestCase):
    def setUp(self):
        sandbox = tempfile.TemporaryDirectory(); self.addCleanup(sandbox.cleanup)
        base = Path(sandbox.name)
        self.checkout = base / 'checkout'; self.checkout.mkdir()
        self.temp = base / 'runner-temp'; self.temp.mkdir()
        self.env = {'GITHUB_ACTIONS': 'true', 'RUNNER_ENVIRONMENT': 'github-hosted',
                    'GITHUB_REPOSITORY': 'd6g8k5htny-coder/main', 'GITHUB_SHA': 'a' * 40,
                    'GITHUB_RUN_ID': '12345', 'GITHUB_RUN_ATTEMPT': '2', 'GITHUB_JOB': 'pilot',
                    'GITHUB_SERVER_URL': 'https://github.com',
                    'GITHUB_WORKFLOW_REF': 'd6g8k5htny-coder/main/.github/workflows/periodic-h0-pilot.yml@refs/heads/main',
                    'GITHUB_WORKFLOW_SHA': 'b' * 40,
                    'GITHUB_WORKSPACE': str(self.checkout), 'RUNNER_TEMP': str(self.temp)}
        for name in custody.SOURCE_PATHS:
            p = self.checkout / name; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'# synthetic fixture\n')
        (self.checkout / 'experiments/periodic_h0/run_pilot.py').write_text(SUCCESS)
        root = custody.initialize(self.checkout, self.temp, self.env)
        self.assertEqual(custody.run_pilot(root, self.checkout, self.temp, self.env), 0)

    def seal_with_fault(self, mode):
        env = dict(os.environ); env.update(self.env)
        return subprocess.run([sys.executable, '-c', FAULT, mode, str(REPO)], env=env,
                              capture_output=True, text=True, timeout=60)

    def test_output_scandir_oserror_exits_74(self):
        p = self.seal_with_fault('scandir')
        self.assertEqual(p.returncode, custody.CUSTODY_FAILURE, p.stderr)
        self.assertIn('periodic pilot custody refused', p.stderr)
        self.assertNotIn('Traceback', p.stderr)

    def test_upload_stage_mkdir_oserror_exits_74(self):
        p = self.seal_with_fault('mkdir')
        self.assertEqual(p.returncode, custody.CUSTODY_FAILURE, p.stderr)
        self.assertIn('periodic pilot custody refused', p.stderr)
        self.assertNotIn('Traceback', p.stderr)


if __name__ == '__main__':
    unittest.main()
