"""Sample Hardy's Z(t) and log of the positive weight w(t) with Xi(t) = -w(t) Z(t)
on a grid; save to results/raw/Z_grid_<T0>_<T1>.npz.
Usage: python3 stats/sample_Z.py 600 900 0.02; python3 stats/sample_Z.py 5000 5200 0.015
(several minutes each; stdout saved as results/stats/sample_<T0>.log)."""
import sys, time
import numpy as np
from mpmath import mp

from stats_paths import raw_path

mp.dps = 20
T0, T1, h = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
ts = np.arange(T0, T1 + h / 2, h)
Z = np.empty_like(ts)
logw = np.empty_like(ts)
t1 = time.time()
for i, t in enumerate(ts):
    Z[i] = float(mp.siegelz(t))
    # Xi(t) = -(1/2)(t^2+1/4) pi^{-1/4} |Gamma(1/4 + i t/2)| Z(t)
    logw[i] = float(mp.log((t * t + 0.25) / 2) - mp.log(mp.pi) / 4 + mp.re(mp.loggamma(0.25 + 0.5j * t)))
np.savez(raw_path(f"Z_grid_{int(T0)}_{int(T1)}.npz"), t=ts, Z=Z, logw=logw)
print(f"{len(ts)} samples in {time.time()-t1:.1f}s")
# spot check the Xi relation at one point
t = mp.mpf(T0) + 0.123
lhs = mp.re(mp.mpf(1) / 2 * (0.5 + 1j * t) * (-0.5 + 1j * t) * mp.pi ** (-(0.5 + 1j * t) / 2) * mp.gamma((0.5 + 1j * t) / 2) * mp.zeta(0.5 + 1j * t))
rhs = -mp.exp(mp.log((t * t + 0.25) / 2) - mp.log(mp.pi) / 4 + mp.re(mp.loggamma(0.25 + 0.5j * t))) * mp.siegelz(t)
print("Xi(t) check:", mp.nstr(lhs, 12), mp.nstr(rhs, 12))
