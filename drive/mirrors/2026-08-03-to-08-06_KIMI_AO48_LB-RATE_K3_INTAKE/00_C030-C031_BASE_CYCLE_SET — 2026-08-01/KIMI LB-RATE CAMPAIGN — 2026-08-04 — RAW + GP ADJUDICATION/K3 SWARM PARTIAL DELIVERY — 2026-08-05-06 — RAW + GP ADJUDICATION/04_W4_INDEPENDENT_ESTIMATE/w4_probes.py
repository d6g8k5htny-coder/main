#!/usr/bin/env python3
"""
W4 INDEPENDENT - probe conditional laws at 15+ digits (deliverable 4).
Reports the law of (f, grad f, H) at y given the 9 pins: mean mu6 and
covariance Sigma6 (full 6x6), plus the further-conditioned law of
(f, H11, H12, H22) given grad f(y) = 0, p_grad(0), and the rho components.
All values printed at 25 significant digits from the dps=60 pipeline.

Usage: python3 w4_probes.py y1a y2a y1b y2b OUT.txt
"""
import sys
from mpmath import mp, mpf

mp.dps = 60
import w4_kernel as WK
import w4_condlaw as CL

WK.set_production_wrap(4)


def fmt(v):
    return mp.nstr(v, 25)


def report(y, label, fh):
    mu6, Sig6 = CL.conditional_jet(y)
    m4, S4 = CL.cond_fH_given_grad0_from(mu6, Sig6)
    pg = CL.grad_density0_from(mu6, Sig6)
    fh.write("=== PROBE %s : y = (%s, %s) ===\n" % (label, fmt(y[0]), fmt(y[1])))
    fh.write("mu6 = (f, f1, f2, f11, f12, f22) | 9 pins:\n")
    for i in range(6):
        fh.write("  mu6[%d] = %s\n" % (i, fmt(mu6[i])))
    fh.write("Sigma6 (rows i=0..5, cols j=0..i):\n")
    for i in range(6):
        for j in range(i + 1):
            fh.write("  S6[%d][%d] = %s\n" % (i, j, fmt(Sig6[i, j])))
    fh.write("p_grad(0) = %s\n" % fmt(pg))
    fh.write("(f, H11, H12, H22) | pins, grad(y)=0:\n")
    for i in range(4):
        fh.write("  m4[%d] = %s\n" % (i, fmt(m4[i])))
    for i in range(4):
        for j in range(i + 1):
            fh.write("  S4[%d][%d] = %s\n" % (i, j, fmt(S4[i, j])))
    fh.write("\n")


def main():
    y1a, y2a, y1b, y2b = [mpf(a) for a in sys.argv[1:5]]
    out = sys.argv[5]
    mu_t, pv = CL.pin_values()
    with open(out, "w") as fh:
        fh.write("W4 PROBE CONDITIONAL LAWS (dps=60, 25 sig digits)\n")
        fh.write("r = %s\n" % fmt(CL.R))
        fh.write("b = %s\n" % fmt(CL.B))
        fh.write("ell = %s\n" % fmt(CL.ELL))
        fh.write("M = (%s, %s)\n" % (fmt(CL.M[0]), fmt(CL.M[1])))
        fh.write("S = (%s, %s)\n" % (fmt(CL.S[0]), fmt(CL.S[1])))
        fh.write("Y = (%s, %s)\n" % (fmt(CL.Y[0]), fmt(CL.Y[1])))
        fh.write("mu_t = %s\n" % fmt(mu_t))
        fh.write("\n")
        report((y1a, y2a), "P1", fh)
        report((y1b, y2b), "P2", fh)
    print("WROTE %s" % out)


if __name__ == "__main__":
    main()
