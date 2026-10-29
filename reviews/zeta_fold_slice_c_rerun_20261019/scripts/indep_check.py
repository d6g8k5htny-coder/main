#!/usr/bin/env python3
"""Independent slice-C check, written by the reviewer and NOT part of the packet.

1. Refit the Riemann-Siegel Chebyshev coefficients into a private raw dir and hash them.
2. Hardy Z from stats/rs_zeta.py at 200 uniform random points per window with a DIFFERENT seed
   from the packet's (20261012), against mpmath.siegelz at 30 digits.
3. Zero counts by sign change of Z on fine grids over four windows, each at two grid steps,
   against the exact mpmath.nzeros(T1) - nzeros(T0).
4. The H_0 window [1, 250] used by dbn/validate_H0.py: exact nzeros(125) and the bracketing
   zeta zeros, for comparison with the scan's count.
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import mpmath

PKT = Path(os.environ["ZF_PKT"]).resolve()
RAW = Path(os.environ["ZF_INDEP_RAW"]).resolve()
RAW.mkdir(parents=True, exist_ok=True)
os.environ["ZETA_FOLD_RAW"] = str(RAW)
sys.path.insert(0, str(PKT / "stats"))
import rs_zeta as rs  # noqa: E402  (fits coefficients into RAW on import)

SEED = 8675309
out = dict(
    seed=SEED,
    python=sys.version.split()[0], numpy=np.__version__, mpmath=mpmath.__version__,
    platform=platform.platform(),
)
cf = RAW / "rs_cheb_coeffs.npz"
out["rs_cheb_coeffs"] = dict(bytes=cf.stat().st_size, sha256=hashlib.sha256(cf.read_bytes()).hexdigest())

rng = np.random.default_rng(SEED)
mpmath.mp.dps = 30
val = {}
for lo, hi in [(1e4, 1e5), (1e5, 1e6)]:
    ts = np.sort(rng.uniform(lo, hi, 200))
    t0 = time.time()
    ref = np.array([float(mpmath.siegelz(mpmath.mpf(float(x)))) for x in ts])
    row = {f"maxabs_err_corr{k}": float(np.max(np.abs(rs.Z(ts, nterms_corr=k) - ref))) for k in (1, 2, 3)}
    row.update(n_points=200, max_abs_Z=float(np.max(np.abs(ref))), siegelz_seconds=round(time.time() - t0, 1))
    val[f"{lo:g}-{hi:g}"] = row
out["rs_vs_siegelz_other_seed"] = val


def count_sign_changes(T0, T1, step):
    t = np.arange(T0, T1 + step / 2, step)
    z = rs.Z(t)
    return int(np.count_nonzero(np.sign(z[1:]) != np.sign(z[:-1]))), len(t)


counts = []
for T0, T1, step in [(10000.0, 12000.0, 0.01), (99000.0, 101000.0, 0.005),
                     (500000.0, 500500.0, 0.005), (990000.0, 991000.0, 0.004)]:
    t0 = time.time()
    exact = int(mpmath.nzeros(T1)) - int(mpmath.nzeros(T0))
    nz_s = round(time.time() - t0, 1)
    c1, n1 = count_sign_changes(T0, T1, step)
    c2, n2 = count_sign_changes(T0, T1, step / 2)
    counts.append(dict(window=[T0, T1], step=step, grid_points=n1, sign_changes=c1,
                       sign_changes_half_step=c2, exact_nzeros_diff=exact, nzeros_seconds=nz_s,
                       match=(c1 == exact == c2)))
out["zero_counts_vs_nzeros"] = counts

mpmath.mp.dps = 20
out["H0_window_1_250"] = dict(
    nzeros_125=int(mpmath.nzeros(125)),
    two_gamma_41=float(2 * mpmath.zetazero(41).imag),
    two_gamma_42=float(2 * mpmath.zetazero(42).imag),
)
out["all_counts_match"] = all(c["match"] for c in counts)
Path(os.environ["ZF_LOG"], "indep_check.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
