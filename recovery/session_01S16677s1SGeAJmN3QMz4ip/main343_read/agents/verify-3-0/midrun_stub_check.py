"""Independent check (a): stub _campaign edits PROOF.md during evaluation, returns a full PASS certificate.
Usage: python3.12 -B -E -S midrun_stub_check.py <copy-root>"""
import importlib.util, json, shutil, sys, tempfile
from fractions import Fraction as Q
from pathlib import Path
POP = 'experiments/universality/two_parameter_cusp_population'
root = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp) / 'copy'
    shutil.copytree(root, base)
    sys.path.insert(0, str(base / POP))           # so `import population_certificate` resolves
    spec = importlib.util.spec_from_file_location('runner_under_test', base / POP / 'run_population_certificate.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    assert runner.REPOSITORY == base, runner.REPOSITORY
    proof = base / POP / 'PROOF.md'
    def stub(pc, max_evaluations, deadline):
        proof.write_bytes(proof.read_bytes() + b'X')          # a source changes DURING evaluation
        rows = []
        for j in range(1, 9):
            rows.append({'j': j, 'tau': Q(1, 1 << (3 * j)), 'n': 4 ** j, 'status': 'PASS',
                         'coefficient_status': 'PASS',
                         'direct': {'status': 'CERTIFIED', 'evaluations': 1,
                                    'normalized_interval': (Q(0), Q(1))},
                         'cdf_normalized': (Q(0), Q(1)),
                         **({'finite_void': {'status': 'PASS', 'endpoint_distance': Q(0)},
                             'void_interval': (Q(0), Q(1)), 'target_interval': (Q(0), Q(1))}
                            if j == 8 else {})})
        return {'schema_version': 1, 'status': 'PASS', 'rows': rows,
                'scope_statuses': {'exact_non_gaussian_population': 'PASS'}}
    runner._campaign = stub
    rec = runner.run(output_dir=base / 'attempt', max_evaluations=80000, time_budget=900)
    cert = runner.load_json(base / 'attempt/CERTIFICATE.json')
    disk = runner.load_json(base / 'attempt/RUN.json')
    print('returned: status=%s exit_code=%s source_drift=%s full_declared_campaign=%s' %
          (rec['status'], rec['exit_code'], rec['source_drift'], rec['full_declared_campaign']))
    print('on-disk CERTIFICATE.json status=%s source_drift=%s row8 finite_void=%s' %
          (cert['status'], cert.get('source_drift'), cert['rows'][7].get('finite_void')))
    key = POP + '/PROOF.md'
    print('on-disk RUN.json status=%s source_drift=%s full_declared_campaign=%s' %
          (disk['status'], disk['source_drift'], disk['full_declared_campaign']))
    print('  sources_before[PROOF.md]=%s' % disk['sources_before'][key])
    print('  sources_after [PROOF.md]=%s' % disk['sources_after'][key])
