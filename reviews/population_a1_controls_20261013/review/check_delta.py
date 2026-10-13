"""Read-only bounded replay of the reviewed A1 campaign regression and mutant."""
import hashlib
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

repo = Path(sys.argv[1]).resolve()
folder = repo / 'experiments/universality/two_parameter_cusp_population'
expected = {
    'population_certificate.py': 'be1d135b838733e4549b2519bae30d02377c97561a8876a16d9b7ea198305edf',
    'test_population_certificate.py': '3d653a0646a8121b44982a65c14d8a2a82c61343522e4b4ab76fd64166810d4e',
    'run_population_certificate.py': '3799f53e2066ae3f47a6b65ca93a130f2531ee78df1e0f4a094077cd8eabfb4e',
}
actual = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in expected}
if actual != expected:
    raise SystemExit('Unexpected source identity')
sys.path.insert(0, str(folder))
import test_population_certificate as tests
import run_population_certificate as runner

def check(module):
    suite = unittest.TestSuite([tests.MassControlRouteTests(
        'test_real_campaign_withholds_acceptance_after_mass_control_fault')])
    with mock.patch.object(tests, 'runner', module):
        result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    return {'tests': result.testsRun, 'failures': len(result.failures),
            'errors': len(result.errors), 'skipped': len(result.skipped),
            'passed': result.wasSuccessful()}

candidate = check(runner)
source = (folder/'run_population_certificate.py').read_text()
removed = "                elif not sound:\n                    row['status'] = 'FAIL_IMPLEMENTATION'\n"
if source.count(removed) != 1:
    raise SystemExit('Unexpected mutation location count')
mutated = source.replace(removed, '')
module = types.ModuleType('review_runner_sound_gate_mutant')
module.__file__ = runner.__file__
exec(compile(mutated, 'review_runner_sound_gate_mutant', 'exec'), module.__dict__)
mutant = check(module)
if candidate != {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0, 'passed': True}:
    raise SystemExit('Candidate regression did not pass')
if mutant != {'tests': 1, 'failures': 1, 'errors': 0, 'skipped': 0, 'passed': False}:
    raise SystemExit('Sound-gate mutant was not killed by an assertion')
print(json.dumps({
    'reviewer': 'OpenAI/Codex review_population_control_repair',
    'reviewed_commit': '10de976214f9cfc36d50fc4bbee7ceba3683cafb',
    'reviewed_tree': '0e0108007ee1cb893739ab32c80f350cd8be0acc',
    'source_sha256': actual,
    'optimized': bool(sys.flags.optimize),
    'candidate': candidate,
    'mutation': {'removed': removed, 'mutant_sha256': hashlib.sha256(mutated.encode()).hexdigest(),
                 'test_result': mutant},
    'full_numerical_campaign': False,
    'scope': 'Synthetic certified-route acceptance wiring with real failed A1 controls',
}, indent=2, sort_keys=True))
