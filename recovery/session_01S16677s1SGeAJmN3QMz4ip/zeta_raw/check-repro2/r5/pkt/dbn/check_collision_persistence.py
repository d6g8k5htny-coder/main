"""Superlevel H0 persistence of F_t(x) = int H_t near the collision found by
check_collision.py: the newborn pair is a persistence pair whose bar equals the
area between the two new zeros, ~ |A| r^3/6; and before the collision the bar is absent.
Input: results/dbn/collision_point.json (written by check_collision.py).
Output: results/dbn/check_collision_persistence.out
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "dbn"

import numpy as np
from mpmath import mp
from dbn_common import HFlow
from pers1d import superlevel_h0

mp.dps = 40
pt = json.loads((OUT / "collision_point.json").read_text())
z0, t0, A = mp.mpf(pt["z0"]), mp.mpf(pt["t0"]), mp.mpf(pt["A"])
F = HFlow(U=1.3, pieces=26, degree=4)
out = []
def rep(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

for tau in ["0.01", "-0.01"]:
    tt = t0 + mp.mpf(tau)
    xs = np.linspace(float(z0) - 9, float(z0) + 9, 3601)
    H = np.array([float(F.H(tt, mp.mpf(x))) for x in xs])
    # cumulative trapezoid antiderivative
    Fv = np.concatenate([[0.0], np.cumsum((H[1:] + H[:-1]) / 2 * np.diff(xs))])
    bars, ess = superlevel_h0(list(Fv))
    bars = sorted(bars, key=lambda b: b[2] - b[3])
    zeros = xs[:-1][np.sign(H[1:]) != np.sign(H[:-1])]
    rep(f"tau={tau}: real zeros of H_t in window: {np.round(zeros,4).tolist()}")
    for (bi, di, bv, dv) in bars:
        rep(f"   bar: birth at x={xs[bi]:.4f}, death at x={xs[di]:.4f}, length={bv-dv:.6e}")
    if tau == "0.01":
        r = 2 * mp.sqrt(2 * mp.mpf(tau))
        rep(f"   prediction for newborn bar |A| r^3/6 with r~2 sqrt(2 tau): {mp.nstr(abs(A)*r**3/6,6)} (sampled-grid antiderivative)")

(OUT / "check_collision_persistence.out").write_text("\n".join(out) + "\n")
