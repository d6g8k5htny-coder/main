#!/usr/bin/env python3
"""
W4 INDEPENDENT - spatial integration of rho(y) over the rigidity zone.

Grids (CLI mode):
  wedge TH0 TH1 R0 R1 DTH_DR  : fine polar grid over the hot wedge, trapezoid
  full  NR NTH R0 R1          : medium polar grid over full annulus
  inner                        : certified kill bound for r in (0, 0.05]
  outer NR NTH R0 R1           : shell beyond 1.5 (context / truncation report)

All rho values computed exactly per node (mpmath conditional law ->
float64 nested GL quadrature). Deterministic ordering. Fail-closed.
"""
import sys
import os
import numpy as np
from mpmath import mp, mpf

mp.dps = 50

import w4_kernel as WK
import w4_condlaw as CL
import w4_rho as RW

B = 1.2
ELLf = float(CL.ELL)

_WRAP_CERT = WK.set_production_wrap(4)  # certified n=0-only mode, |s|<=4


def eval_point_idx(args):
    idx, y1, y2 = args
    y = (mpf(y1), mpf(y2))
    mu6, Sig6 = CL.conditional_jet(y)
    m4, S4 = CL.cond_fH_given_grad0_from(mu6, Sig6)
    pg = CL.grad_density0_from(mu6, Sig6)
    Sg = np.array([[float(Sig6[1, 1]), float(Sig6[1, 2])],
                   [float(Sig6[1, 2]), float(Sig6[2, 2])]])
    mg = np.array([float(mu6[1]), float(mu6[2])])
    detSg = Sg[0, 0] * Sg[1, 1] - Sg[0, 1] ** 2
    expo2 = 0.5 * mg @ np.linalg.solve(Sg, mg)
    core = RW.RhoCore([float(v) for v in m4],
                      [[float(S4[i, j]) for j in range(4)] for i in range(4)],
                      B, ELLf)
    ew, t1 = core.E_win()
    pw, t2 = core.P_detneg_win()
    pgf = float(pg)
    rho = pgf * ew
    cs = pgf * np.sqrt(max(core.E_det2(), 0.0)) * np.sqrt(max(pw, 0.0))
    csb = pgf * np.sqrt(max(core.E_det2(), 0.0))  # cheap CS bound (P<=1)
    return (idx, y1, y2, float(mu6[0]), float(np.linalg.norm(mg)),
            np.sqrt(detSg), expo2, pgf, float(Sig6[0, 0]), ew, pw,
            core.E_det2(), rho, cs, csb, t1 + t2)


COLS = ["y1", "y2", "m", "gradn", "sqrtdetSg", "expo2", "pg", "vf",
        "E_win", "Pwin", "E_det2", "rho", "rho_CS", "rho_CSb", "tail"]


def run_grid(points, out):
    """points: list of (y1,y2). Evaluate in pool, write TSV, return array."""
    n = len(points)
    args = [(i, p[0], p[1]) for i, p in enumerate(points)]
    results = [None] * n
    import multiprocessing as mpp
    with mpp.Pool(2) as pool:
        for res in pool.imap_unordered(eval_point_idx, args, chunksize=16):
            results[res[0]] = res
    RW.ck(all(r is not None for r in results), "all grid points evaluated")
    arr = np.array([r[1:] for r in results], dtype=np.float64)
    hdr = "# " + " ".join(COLS)
    np.savetxt(out, arr, header=hdr, fmt="%.10e")
    return arr


def trap2d(A, ths, rs):
    """Integrate A(theta,r) * r over polar grid; trapezoid both directions."""
    I1 = np.trapezoid(A * rs[None, :], ths, axis=0)
    return float(np.trapezoid(I1, rs))


def main():
    mode = sys.argv[1]
    CL.pin_values()
    if mode == "wedge":
        th0, th1, r0, r1 = map(float, sys.argv[2:6])
        dth, dr = map(float, sys.argv[6:8])
        out = sys.argv[8]
        ths = np.deg2rad(np.arange(th0, th1 + 0.5 * dth, dth))
        rs = np.arange(r0, r1 + 0.5 * dr, dr)
        pts = [(float(np.cos(t) * r), float(np.sin(t) * r)) for t in ths for r in rs]
        arr = run_grid(pts, out)
        nth, nr = len(ths), len(rs)
        A = arr[:, 11].reshape(nth, nr)
        CS = arr[:, 12].reshape(nth, nr)
        I = trap2d(A, ths, rs)
        Ics = trap2d(CS, ths, rs)
        print("WEDGE th=[%.3f,%.3f]deg r=[%.4f,%.4f] nth=%d nr=%d" % (
            th0, th1, r0, r1, nth, nr))
        print("I_wedge = %.12e" % I)
        print("ICS_wedge = %.12e" % Ics)
        print("max boundary rho (should be << peak): %.3e" % max(
            A[0, :].max(), A[-1, :].max(), A[:, 0].max(), A[:, -1].max()))
    elif mode == "full":
        NR, NTH, r0, r1 = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
        out = sys.argv[6]
        ths = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
        rs = np.linspace(r0, r1, NR)
        pts = [(float(np.cos(t) * r), float(np.sin(t) * r)) for t in ths for r in rs]
        arr = run_grid(pts, out)
        A = arr[:, 11].reshape(NTH, NR)
        CS = arr[:, 12].reshape(NTH, NR)
        print("FULL r=[%.3f,%.3f] NR=%d NTH=%d" % (r0, r1, NR, NTH))
        print("I_full = %.12e" % trap2d(A, ths, rs))
        print("ICS_full = %.12e" % trap2d(CS, ths, rs))
    elif mode == "inner":
        out = sys.argv[2]
        rr = [0.002, 0.005, 0.01, 0.02, 0.035, 0.05]
        ths = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        pts = [(float(np.cos(t) * r), float(np.sin(t) * r)) for t in ths for r in rr]
        arr = run_grid(pts, out)
        # certified bound: rho <= rho_CSb = pg * sqrt(E_det2)  (P <= 1)
        bnd = arr[:, 13]
        expo2 = arr[:, 6]
        print("INNER scan r in {0.002..0.05}: min expo/2 = %.3f" % expo2.min())
        print("max rho_CSb = %.3e" % bnd.max())
        # bound integral over disk r<=0.05 by max rho_CSb * area
        print("disk-area bound: %.3e" % (bnd.max() * np.pi * 0.05 ** 2))
    elif mode == "outer":
        NR, NTH, r0, r1 = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
        out = sys.argv[6]
        ths = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
        rs = np.linspace(r0, r1, NR)
        pts = [(float(np.cos(t) * r), float(np.sin(t) * r)) for t in ths for r in rs]
        arr = run_grid(pts, out)
        A = arr[:, 11].reshape(NTH, NR)
        print("OUTER r=[%.3f,%.3f]" % (r0, r1))
        print("I_outer = %.12e" % trap2d(A, ths, rs))
    else:
        sys.stderr.write("unknown mode\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
