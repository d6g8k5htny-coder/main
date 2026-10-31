"""Run the frozen ST pilot in isolated children, not the discovery process.

This is software/algebra evidence only. No field theorem is accepted here.
Command-only check scripts deliberately do not match test_*.py. All three
commands remain exercised below; no existing suite or workflow is excluded.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

PILOT = Path(__file__).resolve().parents[1] / 'experiments/st_contract_pilot'
PINNED = {'check_object_overlay.py': '3fa1ad97c8265ce9bceadbff0ccaa6037f5c3f996e4e4ac4c7668d2c12ce7383', 'check_scaling_diagnostic.py': '3cd0b0890e3971cc8034f08f4e5a0a53a7cfbc0f5cc22193be146b6421caf570', 'deps/fold_controls.py': 'b4c1c230517df55e35254aa810898799a850573d9967649eef7161b0918f0f9a', 'deps/st_object_overlay.py': '343af2feddf9e49798da09c222a19ae2e9f548e89fcf9882e2ed2ff6b146fdd2', 'fixtures/st_diagnostic_cases.json': 'a24b17ff803cb0cf72531f460c757f7c4c65534792a98912ccb0bde131ac322e', 'fixtures/st_measure_contract.json': '5a521b035d257349530cef33721e1aaa45eb2280e009f4cea641dcef66461e6d', 'run_cases.py': '34713219bfcfb58f2d2f939444e5415526c27a47264d7ce1c44627ba83737f24', 'st_scaling_diagnostic.py': 'ccdaf9e56179db91de2213d5ffe264e90df3304219afdb1d9622de01182d100c'}


class STContractPilotTests(unittest.TestCase):
    def verify_inputs(self):
        for name, digest in PINNED.items():
            # Do not wrap this pre-execution guard in subTest: subTest records
            # a failure and continues, which would still launch the child.
            self.assertEqual(hashlib.sha256((PILOT / name).read_bytes()).hexdigest(),
                             digest, 'Frozen pilot input changed: ' + name)

    def run_child(self, filename):
        # Verify before any project/pilot code executes, including check scripts.
        self.verify_inputs()
        names = ('fold_controls', 'st_object_overlay', 'st_scaling_diagnostic')
        absent = object()
        before = {name: sys.modules.get(name, absent) for name in names}
        command = [sys.executable, '-I', '-B', '-S']
        if sys.flags.optimize:
            command.append('-O')
        run = subprocess.run(command + [str(PILOT / filename)], cwd=PILOT,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        for name, previous in before.items():
            self.assertIs(sys.modules.get(name, absent), previous,
                          'Child execution contaminated parent module: ' + name)
        self.verify_inputs()
        return run

    def check_suite(self, filename, count):
        run = self.run_child(filename)
        self.assertEqual(run.stdout, '')
        self.assertRegex(run.stderr, r'\nRan ' + str(count) +
                         r' tests in [^\n]+\n\nOK\n\Z')

    def test_frozen_dependency_and_fixture_identities(self):
        self.assertEqual(len(PINNED), 8)
        self.verify_inputs()
        self.assertEqual(list(PILOT.rglob('test_*.py')), [],
                         'Command-only scripts must not run at discovery')

    def test_overlay_controls(self):
        self.check_suite('check_object_overlay.py', 28)

    def test_scaling_controls(self):
        self.check_suite('check_scaling_diagnostic.py', 32)

    def test_eight_case_observations(self):
        run = self.run_child('run_cases.py')
        self.assertEqual(run.stderr, '')
        report = json.loads(run.stdout)
        self.assertEqual(report['scope'], 'BOUNDED_ST_ALGEBRA_DIAGNOSTIC_ONLY')
        self.assertEqual(report['optimized'], sys.flags.optimize)
        self.assertEqual(report['cases_executed'], 8)
        self.assertIs(report['scientific_acceptance'], False)
        rows = report['results']
        self.assertEqual([r['case_id'] for r in rows],
                         ['ST-%02d' % i for i in range(1, 9)])
        self.assertEqual([r['observed']['status'] for r in rows],
                         ['ALGEBRA_ONLY','ALGEBRA_ONLY','REFUSED','REFUSED',
                          'REFUSED','ALGEBRA_ONLY','ALGEBRA_ONLY','REFUSED'])
        self.assertEqual(rows[0]['observed']['density_power'], '-1/3')
        self.assertEqual(rows[1]['observed']['density_power'], '0')
        for row in rows:
            for flag in ('scientific_acceptance', 'field_admission',
                         'complete_downstream_coverage', 'remote_source_authenticated'):
                self.assertIs(row['observed'][flag], False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
