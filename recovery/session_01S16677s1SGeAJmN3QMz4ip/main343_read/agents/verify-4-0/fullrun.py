import sys, json
from pathlib import Path
root = Path(sys.argv[1]); out = Path(sys.argv[2])
sys.path.insert(0, str(root / 'experiments/universality/two_parameter_cusp_population'))
import run_population_certificate as runner
res = runner.run(output_dir=out)  # defaults: max_evaluations=80000, time_budget=900
cert = runner.load_json(out / 'CERTIFICATE.json')
print('exit_code:', res['exit_code'], '| overall status:', cert['status'], '| full_declared_campaign:', res['full_declared_campaign'], '| drift:', res['source_drift'], '| evaluations:', res['actual_route_evaluations'])
nc = cert['negative_controls']
print('negative_controls status:', nc['status'], '| all gates true:', all(nc['gates'].values()), '| whole_rectangle_mass:', nc['whole_rectangle_mass'])
print('scope exact_non_gaussian_population:', cert['scope_statuses']['exact_non_gaussian_population'])
from fractions import Fraction as Q
for r in cert['rows']:
    d = r['direct']
    print(f"  j={r['j']} direct={d['status']} coeff={r['coefficient_status']:22} row={r['status']:22} intersection=[{float(Q(r['intersection_normalized'][0])):.6f},{float(Q(r['intersection_normalized'][1])):.6f}] outer_dist={float(Q(r['outer_envelope_distance'])):.5f}" if r.get('intersection_normalized') else f"  j={r['j']} direct={d['status'] if d else None} row={r['status']}")
print('B:', [float(Q(x)) for x in cert['coefficient_interval']])
