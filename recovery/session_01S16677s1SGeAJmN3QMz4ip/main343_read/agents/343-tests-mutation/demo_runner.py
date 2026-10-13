"""Runner-level demonstrations of what surviving mutants change in a full-campaign
certificate. Usage: demo_runner.py <root-or-mutant-dir> <scenario>
Scenarios:
  wrongB        - pristine direct rows (cached) but coefficient interval B shifted by +1/1000:
                  rows 7,8 must be FALSIFIED (E(tau_7)=4.1e-4, E(tau_8)=7.8e-5 < 1e-3).
  disjoint_cdf  - pristine direct rows but the CDF route shifted by +1/100: the two routes
                  are disjoint, so every row must be FAIL_ROUTE_CONSISTENCY.
  midrun_drift  - a throwaway copy of the dir; _campaign is replaced by a stub that appends a
                  byte to PROOF.md during evaluation and returns a PASS certificate: the
                  post-run hash check must demote the attempt to INCONCLUSIVE_SOURCE_DRIFT.
The scenarios never modify the given dir (midrun_drift works on a tempfile copy)."""
import importlib.util, pickle, shutil, sys, tempfile, time, types
from fractions import Fraction as Q
from pathlib import Path
HERE = Path(__file__).resolve().parent
POP = 'experiments/universality/two_parameter_cusp_population'

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def proxy_pc(pc, cache, b_shift=Q(0), cdf_shift=Q(0)):
    p = types.SimpleNamespace(**{k: getattr(pc, k) for k in dir(pc) if not k.startswith('__')})
    p.direct_mass = lambda j, max_evaluations=80000, deadline=None: cache[j]
    base_b = pc.coefficient_interval()
    p.coefficient_interval = lambda: (base_b[0] + b_shift, base_b[1] + b_shift)
    p.poisson_target = lambda b=None: pc.poisson_target(p.coefficient_interval() if b is None else b)
    p.cdf_mass = lambda tau: tuple(x + cdf_shift for x in pc.cdf_mass(tau))
    return p

def summarize(cert):
    print('overall status:', cert['status'])
    for row in cert['rows']:
        fv = row.get('finite_void', {}).get('status')
        print(f"  j={row['j']} status={row['status']:24} coefficient_status={row['coefficient_status']:24} "
              f"finite_void={fv} error={row.get('error')}")

if __name__ == '__main__':
    target, scenario = Path(sys.argv[1]).resolve(), sys.argv[2]
    sys.path.insert(0, str(target / POP))
    if scenario in ('wrongB', 'wrongB_small', 'disjoint_cdf', 'plain'):
        pc = load(target / POP / 'population_certificate.py', 'pc_demo')
        runner = load(target / POP / 'run_population_certificate.py', 'runner_demo')
        cache = pickle.load((HERE / 'direct_cache.pkl').open('rb'))
        kw = {'wrongB': {'b_shift': Q(1, 1000)}, 'wrongB_small': {'b_shift': Q(3, 25000)}, 'disjoint_cdf': {'cdf_shift': Q(1, 100)}, 'plain': {}}[scenario]
        cert = runner._campaign(proxy_pc(pc, cache, **kw), 80000, time.monotonic() + 900)
        summarize(cert)
    elif scenario == 'midrun_drift':
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'copy'
            shutil.copytree(target, base)
            runner = load(base / POP / 'run_population_certificate.py', 'runner_demo')
            proof = base / POP / 'PROOF.md'
            def stub_campaign(pc, max_evaluations, deadline):
                proof.write_bytes(proof.read_bytes() + b'\n')   # source changes DURING evaluation
                return {'schema_version': 1, 'status': 'PASS', 'rows': [
                    {'j': 1, 'tau': Q(1, 8), 'status': 'PASS', 'coefficient_status': 'PASS',
                     'direct': {'status': 'CERTIFIED', 'evaluations': 1, 'normalized_interval': (Q(0), Q(1))}}],
                    'scope_statuses': {'exact_non_gaussian_population': 'PASS'}}
            runner._campaign = stub_campaign
            rec = runner.run(output_dir=base / 'attempt', max_evaluations=80000, time_budget=900)
            print('RUN status:', rec['status'], 'exit_code:', rec['exit_code'], 'source_drift:', rec['source_drift'],
                  'full_declared_campaign:', rec['full_declared_campaign'])
