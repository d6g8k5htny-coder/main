"""Own stub-driven _campaign demo. Usage: demo.py <root> ; prints overall + row statuses per scenario."""
import importlib.util, sys, time, types
from fractions import Fraction as Q
from pathlib import Path
root = Path(sys.argv[1]).resolve()
pop = root / 'experiments/universality/two_parameter_cusp_population'
sys.path.insert(0, str(pop))
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pc = load(pop / 'population_certificate.py', 'pc_v')
runner = load(pop / 'run_population_certificate.py', 'runner_v')
def stub(b_shift=Q(0), cdf_shift=Q(0)):
    p = types.SimpleNamespace(**{k: getattr(pc, k) for k in dir(pc) if not k.startswith('__')})
    p.cdf_mass = lambda tau: tuple(x + cdf_shift for x in pc.cdf_mass(tau))
    def direct_mass(j, max_evaluations=80000, deadline=None):
        tau, n = pc._schedule(j); m = pc.cdf_mass(tau)
        lo, hi = n * m[0] - Q(1, 1 << 20), n * m[1] + Q(1, 1 << 20)
        return {'j': j, 'tau': tau, 'n': n, 'status': 'CERTIFIED', 'evaluations': 1,
                'normalized_interval': (lo, hi), 'interval': (lo / n, hi / n)}
    p.direct_mass = direct_mass
    b0 = pc.coefficient_interval()
    p.coefficient_interval = lambda: (b0[0] + b_shift, b0[1] + b_shift)
    p.poisson_target = lambda b=None: pc.poisson_target(p.coefficient_interval() if b is None else b)
    return p
for name, kw in (('consistent', {}), ('wrongB_small(+3/25000)', {'b_shift': Q(3, 25000)}), ('disjoint_cdf(+1/100)', {'cdf_shift': Q(1, 100)})):
    c = runner._campaign(stub(**kw), 80000, time.monotonic() + 900)
    rows = ' '.join(f"{r['j']}:{r['status']}/{r['coefficient_status']}" + (f"/void={r['finite_void']['status']}" if r.get('finite_void') else '') for r in c['rows'])
    print(f"  {name:24} overall={c['status']:24} rows: {rows}")
