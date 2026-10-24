"""Check 16-point Gauss-Legendre lobe integrals of the vectorized Riemann-Siegel Z against
mpmath.quad of mpmath.siegelz on a few lobes of window B (smallest lobe + random ones)."""
import numpy as np, mpmath, json
from stats_paths import out_path, raw_path
d = np.load(raw_path("zeta_B.npz")); z = d["zeros"]; raw = d["lobe_raw"]
rng = np.random.default_rng(3)
idx = [int(np.argmin(np.abs(raw)))] + [int(i) for i in rng.integers(0, raw.size, 3)]
mpmath.mp.dps = 20
rows = []
for k in idx:
    a, b = mpmath.mpf(float(z[k])), mpmath.mpf(float(z[k + 1]))
    ref = float(mpmath.quad(mpmath.siegelz, [a, b]))
    rows.append(dict(lobe=k, t=float(z[k]), gap=float(z[k+1]-z[k]), ours=float(raw[k]), mpmath=ref, rel_err=abs(raw[k]-ref)/abs(ref)))
print(json.dumps(rows, indent=1))
json.dump(rows, open(out_path("zeta_quad_check.json"), "w"), indent=1)
