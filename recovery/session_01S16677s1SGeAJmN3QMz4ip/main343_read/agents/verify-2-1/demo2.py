"""M25/M28 scenarios. Usage: demo2.py <root>"""
import importlib.util, sys, time, types
from fractions import Fraction as Q
from pathlib import Path
root = Path(sys.argv[1]).resolve(); pop = root / 'experiments/universality/two_parameter_cusp_population'
sys.path.insert(0, str(pop))
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pc = load(pop / 'population_certificate.py', 'pc_v2'); runner = load(pop / 'run_population_certificate.py', 'runner_v2')
def stub(override=None, void=None):
    override = override or {}
    p = types.SimpleNamespace(**{k: getattr(pc, k) for k in dir(pc) if not k.startswith('__')})
    def cdf_mass(tau):
        for j, (lo, hi) in override.items():
            t, n = pc._schedule(j)
            if tau == t: return lo / n, hi / n
        return pc.cdf_mass(tau)
    def direct_mass(j, max_evaluations=80000, deadline=None):
        tau, n = pc._schedule(j)
        if j in override: lo, hi = override[j]
        else:
            m = pc.cdf_mass(tau); lo, hi = n * m[0] - Q(1, 1 << 20), n * m[1] + Q(1, 1 << 20)
        return {'j': j, 'tau': tau, 'n': n, 'status': 'CERTIFIED', 'evaluations': 1, 'normalized_interval': (lo, hi), 'interval': (lo / n, hi / n)}
    p.cdf_mass, p.direct_mass = cdf_mass, direct_mass
    if void is not None: p.void_interval = lambda mass, n: void
    return p
c = runner._campaign(stub(override={8: (Q(34, 100), Q(34, 100) + Q(1, 1 << 20))}), 80000, time.monotonic() + 900)
r8 = c['rows'][7]
print(f"  row8 mass 0.34 : overall={c['status']} row8.status={r8['status']} coeff={r8['coefficient_status']} mass_gate={r8['finite_void']['gates']['mass_gate']} void={r8['finite_void']['status']}")
c = runner._campaign(stub(void=(Q(0), Q(1, 4))), 80000, time.monotonic() + 900)
r1 = c['rows'][0]
print(f"  void (0,1/4)   : overall={c['status']} row1.status={r1['status']} coeff={r1['coefficient_status']} void_width_gate={r1['void_width_gate']}")
