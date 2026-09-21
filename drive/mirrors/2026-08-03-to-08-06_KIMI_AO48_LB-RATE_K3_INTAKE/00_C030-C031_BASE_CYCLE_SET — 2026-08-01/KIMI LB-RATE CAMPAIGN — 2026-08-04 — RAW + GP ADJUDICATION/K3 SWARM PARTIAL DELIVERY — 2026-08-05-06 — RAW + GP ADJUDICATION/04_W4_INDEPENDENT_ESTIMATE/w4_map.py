#!/usr/bin/env python3
"""
W4 INDEPENDENT - spatial map of rho(y) over the rigidity zone.

Polar grid about the cluster centroid (0,0): rho evaluated exactly at every
node (mpmath conditional law -> float64 nested quadrature). Writes a TSV
table for the integrator. Deterministic, fail-closed, no asserts.

Usage: python3 w4_map.py NR NTH RMAX OUT.tsv
"""
import sys
import numpy as np
from mpmath import mp, mpf

import w4_condlaw as CL
import w4_rho as RW

mp.dps = 50


def eval_point(y):
    mu6, Sig6 = CL.conditional_jet(y)
    m4, S4 = CL.cond_fH_given_grad0(y)
    pg = CL.grad_density0(y)
    Sg = np.array([[float(Sig6[1, 1]), float(Sig6[1, 2])],
                   [float(Sig6[1, 2]), float(Sig6[2, 2])]])
    mg = np.array([float(mu6[1]), float(mu6[2])])
    detSg = Sg[0, 0] * Sg[1, 1] - Sg[0, 1] ** 2
    expo2 = 0.5 * mg @ np.linalg.solve(Sg, mg)
    m4f = [float(v) for v in m4]
    S4f = [[float(S4[i, j]) for j in range(4)] for i in range(4)]
    pgf = float(pg)
    core = RW.RhoCore(m4f, S4f, 1.2, float(CL.ELL))
    ew, t1 = core.E_win()
    pw, t2 = core.P_detneg_win()
    rho = pgf * ew
    cs = pgf * np.sqrt(max(core.E_det2(), 0.0)) * np.sqrt(max(pw, 0.0))
    return dict(m=float(mu6[0]), gradn=float(np.linalg.norm(mg)),
                detSg=detSg, expo2=expo2, pg=pgf, E_win=ew, Pwin=pw,
                E_det2=core.E_det2(), rho=rho, rho_CS=cs,
                vf=float(Sig6[0, 0]), tail=t1 + t2)


def main():
    NR = int(sys.argv[1])
    NTH = int(sys.argv[2])
    RMAX = float(sys.argv[3])
    out = sys.argv[4]
    RW.ck(NR >= 2 and NTH >= 4, "grid sizes")
    rs = np.linspace(0.05, RMAX, NR)
    ths = np.linspace(0.0, 2.0 * np.pi, NTH, endpoint=False)
    CL.pin_values()  # initialize mu_t once (printed by caller)
    lines = ["# y1 y2 m |gradm| sqrt(detSg) expo/2 p_grad0 vf E_win Pwin E_det2 rho rho_CS tail"]
    for th in ths:
        c, s = np.cos(th), np.sin(th)
        for rr in rs:
            y = (mpf(float(c * rr)), mpf(float(s * rr)))
            r = eval_point(y)
            lines.append("%.8f %.8f %.10f %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.3e" % (
                c * rr, s * rr, r["m"], r["gradn"], np.sqrt(r["detSg"]),
                r["expo2"], r["pg"], r["vf"], r["E_win"], r["Pwin"],
                r["E_det2"], r["rho"], r["rho_CS"], r["tail"]))
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("WROTE %s rows=%d" % (out, NR * NTH))


if __name__ == "__main__":
    main()
