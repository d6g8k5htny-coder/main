"""Feed the real (independently computed) CERTIFIED direct rows through _campaign of a given
copy root, with the CDF route or B interval perturbed, and print the certificate outcome.
Usage: impact.py <copy-root> <scenario> ; scenario in plain|disjoint_cdf|wrongB_small"""
import importlib.util, pickle, sys, time, types
from fractions import Fraction as Q
from pathlib import Path
V = Path(__file__).resolve().parents[1]
root = Path(sys.argv[1]).resolve(); scenario = sys.argv[2]
pop = root / 'experiments/universality/two_parameter_cusp_population'
sys.path.insert(0, str(pop))
def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
pc = load(pop / 'population_certificate.py', 'pc_x'); runner = load(pop / 'run_population_certificate.py', 'runner_x')
rows = pickle.load(open(V / 'direct_rows.pkl', 'rb'))
proxy = types.SimpleNamespace(**{k: getattr(pc, k) for k in dir(pc) if not k.startswith('__')})
proxy.direct_mass = lambda j, max_evaluations=80000, deadline=None: rows[j]
b0 = pc.coefficient_interval()
bshift = {'plain': Q(0), 'disjoint_cdf': Q(0), 'wrongB_small': Q(3, 25000)}[scenario]
cshift = {'plain': Q(0), 'disjoint_cdf': Q(1, 100), 'wrongB_small': Q(0)}[scenario]
proxy.coefficient_interval = lambda: (b0[0] + bshift, b0[1] + bshift)
proxy.poisson_target = lambda b=None: pc.poisson_target(proxy.coefficient_interval() if b is None else b)
proxy.cdf_mass = lambda tau: tuple(x + cshift for x in pc.cdf_mass(tau))
cert = runner._campaign(proxy, 80000, time.monotonic() + 900)
print(f'[{root.name} / {scenario}] overall={cert["status"]}')
for r in cert['rows']:
    fv = r.get('finite_void', {}).get('status'); g = r.get('finite_void', {}).get('gates')
    print(f"   j={r['j']} status={r['status']:24} coeff={r['coefficient_status']:24} void={fv} err={r.get('error')}" + (f" gates={g}" if g else ''))
