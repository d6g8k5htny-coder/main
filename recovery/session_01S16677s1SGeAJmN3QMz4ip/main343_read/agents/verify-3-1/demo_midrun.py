"""Mid-run source drift demo for run_population_certificate.run().

Usage: python3.12 -I demo_midrun.py <repo_copy_root>

Mirrors test_preexisting_frozen_source_drift_prevents_population_evaluation
(test lines 229-248): copy SOURCE_PATHS into a tempdir, load the copied runner,
then replace its _campaign with a stub that appends one byte to the copied
PROOF.md during evaluation and returns a PASS-shaped certificate.
"""
from fractions import Fraction as Q
from pathlib import Path
import importlib.util
import json
import sys
import tempfile

repo = Path(sys.argv[1]).resolve()
pop = repo / 'experiments/universality/two_parameter_cusp_population'
sys.path.insert(0, str(pop))
spec = importlib.util.spec_from_file_location('orig_runner', pop / 'run_population_certificate.py')
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)

with tempfile.TemporaryDirectory() as temporary:
    base = Path(temporary)
    for source in orig.SOURCE_PATHS:
        target = base / source.relative_to(orig.REPOSITORY)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    copy_path = base / orig.POPULATION.relative_to(orig.REPOSITORY) / 'run_population_certificate.py'
    spec = importlib.util.spec_from_file_location('copied_runner_midrun', copy_path)
    copied = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copied)
    proof = base / 'experiments/universality/two_parameter_cusp_population/PROOF.md'

    def stub_campaign(pc, max_evaluations, deadline):
        proof.write_bytes(proof.read_bytes() + b'\n')  # source edited during evaluation
        rows = [{'j': j, 'tau': Q(1, 1 << (3 * j)), 'n': 4 ** j, 'status': 'PASS',
                 'coefficient_status': 'PASS',
                 'direct': {'status': 'CERTIFIED', 'normalized_interval': (Q(0), Q(1)),
                            'evaluations': 1},
                 'cdf_normalized': (Q(0), Q(1))} for j in range(1, 9)]
        return {'schema_version': 1, 'status': 'PASS', 'rows': rows,
                'scope_statuses': {'exact_non_gaussian_population': 'PASS'}}

    copied._campaign = stub_campaign
    run = copied.run(output_dir=base / 'attempt', max_evaluations=80000, time_budget=900)
    cert = copied.load_json(base / 'attempt/CERTIFICATE.json')
    print(json.dumps({'runner': str(copy_path.relative_to(base)),
                      'repo_copy': str(repo),
                      'status': run['status'], 'exit_code': run['exit_code'],
                      'source_drift': run['source_drift'],
                      'full_declared_campaign': run['full_declared_campaign'],
                      'certificate_status': cert['status'],
                      'proof_hash_changed': run['sources_before'][
                          'experiments/universality/two_parameter_cusp_population/PROOF.md']['sha256']
                          != run['sources_after'][
                          'experiments/universality/two_parameter_cusp_population/PROOF.md']['sha256']},
                     indent=1))
