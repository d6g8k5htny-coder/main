import sys, time
from fractions import Fraction as Q
V, variant = sys.argv[1], sys.argv[2]
p = f'{V}/{variant}/experiments/universality/two_parameter_cusp_population'
sys.path.insert(0, p)
import population_certificate as pc
import run_population_certificate as run
t0 = time.monotonic()
cert = run._campaign(pc, 80000, time.monotonic() + 900)
print(f'[{variant}] elapsed {time.monotonic()-t0:.1f}s  overall status: {cert["status"]}  negative_controls: {cert["negative_controls"]["status"]}  guards: {cert["rational_guards"]["status"]}')
for row in cert['rows']:
    d = row['direct']
    print(f'  j={row["j"]} status={row["status"]} coeff={row["coefficient_status"]} direct={d["status"] if d else None} '
          f'cdf_norm={[float(x) for x in row["cdf_normalized"]] if row["cdf_normalized"] else None} '
          f'outer_dist={float(row.get("outer_envelope_distance", 0)) if row.get("outer_envelope_distance") is not None else None} '
          f'err={row.get("error","")[:60]}')
