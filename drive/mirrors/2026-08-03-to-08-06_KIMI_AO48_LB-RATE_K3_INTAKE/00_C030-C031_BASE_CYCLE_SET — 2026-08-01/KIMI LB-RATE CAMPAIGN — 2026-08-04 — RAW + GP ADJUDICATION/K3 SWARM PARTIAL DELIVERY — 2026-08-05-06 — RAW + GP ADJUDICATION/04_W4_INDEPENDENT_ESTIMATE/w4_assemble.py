#!/usr/bin/env python3
"""
W4 INDEPENDENT - final assembly: integrate I_WP(0.025) over the rigidity zone
Z = union of disks of radius 1.5 about the pins {M,S,Y}, with error budget.

Inputs: maps/wedge_fine.tsv, wedge_mid.tsv, wedge_finer.tsv, full_med.tsv,
        arc_a.tsv, arc_b.tsv, inner_scan.tsv, outer_shell.tsv
Columns: y1 y2 m gradn sqrtdetSg expo2 pg vf E_win Pwin E_det2 rho rho_CS rho_CSb tail

Deterministic, fail-closed, no asserts.
"""
import sys
import numpy as np

CKFAIL = False


def ck(cond, msg):
    if not cond:
        sys.stderr.write("CK-FAIL: %s\n" % msg)
        raise SystemExit(2)
    return True


def load(path):
    return np.loadtxt(path)


PINS = [(-0.0125, 0.0), (0.0125, 0.0), (-0.0315, 0.006)]
RZ_RADIUS = 1.5


def R_Z(theta):
    """Outer boundary radius of Z along ray theta (union of pin disks)."""
    e = np.array([np.cos(theta), np.sin(theta)])
    best = 0.0
    for p in PINS:
        pv = np.array(p)
        proj = e @ pv
        disc = RZ_RADIUS ** 2 - (pv @ pv) + proj ** 2
        ck(disc > 0, "pin inside Z")
        best = max(best, proj + np.sqrt(disc))
    return best


def grid_trap(A, ths, rs):
    return float(np.trapezoid(np.trapezoid(A * rs[None, :], ths, axis=0), rs))


def load_grid(path, nth, nr, periodic=False, dth=None, dr=None,
              th0=None, th1=None, r0=None, r1=None):
    d = load(path)
    A = d[:, 11].reshape(nth, nr)
    CS = d[:, 12].reshape(nth, nr)
    if periodic:
        ths = np.linspace(th0, th1, nth, endpoint=False)
        rs = np.linspace(r0, r1, nr)
    else:
        ck(dth is not None and dr is not None, "arange grids need steps")
        ths = np.deg2rad(th0 + dth * np.arange(nth))
        rs = r0 + dr * np.arange(nr)
    return d, A, CS, ths, rs


def main():
    # ---- main wedge (arange grids; fine: 45..135 step .25, 0.10..1.55 step .01)
    _, Af, CSf, thf, rf = load_grid("maps/wedge_fine.tsv", 361, 146, dth=0.25, dr=0.01, th0=45, r0=0.10)
    _, Am, CSm, thm, rm = load_grid("maps/wedge_mid.tsv", 181, 73, dth=0.5, dr=0.02, th0=45, r0=0.10)
    # compare over the common domain r <= 1.54
    jf = np.searchsorted(rf, 1.54)
    I_fine = grid_trap(Af, thf, rf)
    I_mid = grid_trap(Am, thm, rm)
    I_fine_c = grid_trap(Af[:, :jf + 1], thf, rf[:jf + 1])
    ICS_fine = grid_trap(CSf, thf, rf)
    ICS_mid = grid_trap(CSm, thm, rm)
    # finer subdomain check [70,110] x [0.30,1.10]
    _, Ax, _, thx, rx = load_grid("maps/wedge_finer.tsv", 321, 161, dth=0.125, dr=0.005, th0=70, r0=0.30)
    I_x = grid_trap(Ax, thx, rx)
    i0 = int(round((70 - 45) / 0.25))
    i1 = int(round((110 - 45) / 0.25))
    j0 = int(round((0.30 - 0.10) / 0.01))
    j1 = int(round((1.10 - 0.10) / 0.01))
    I_x_coarse = grid_trap(Af[i0:i1 + 1, j0:j1 + 1], thf[i0:i1 + 1], rf[j0:j1 + 1])

    # ---- arcs (arange: th step .5, r 0.80..1.54 step .02) ----
    _, Aa, CSa, tha, ra = load_grid("maps/arc_a.tsv", 121, 38, dth=0.5, dr=0.02, th0=140, r0=0.80)
    _, Ab, CSb, thb, rb = load_grid("maps/arc_b.tsv", 161, 38, dth=0.5, dr=0.02, th0=220, r0=0.80)
    I_aa = grid_trap(Aa, tha, ra)
    I_ab = grid_trap(Ab, thb, rb)
    ICS_aa = grid_trap(CSa, tha, ra)
    ICS_ab = grid_trap(CSb, thb, rb)

    # ---- full medium grid for the remainder ----
    dfull, AF, CSF, thF, rF = load_grid("maps/full_med.tsv", 72, 60, periodic=True, th0=0, th1=2*np.pi, r0=0.05, r1=1.55)
    I_full = grid_trap(AF, thF, rF)
    ICS_full = grid_trap(CSF, thF, rF)
    # full-grid integral over the wedge rectangle [45,135]x[0.10,1.55] (mask)
    thFd = np.degrees(thF)
    wedge_mask = (thFd >= 45) & (thFd <= 135)
    r_mask = rF >= 0.10
    I_full_wedge = grid_trap(AF[np.ix_(wedge_mask, r_mask)], thF[wedge_mask], rF[r_mask])
    ICS_full_wedge = grid_trap(CSF[np.ix_(wedge_mask, r_mask)], thF[wedge_mask], rF[r_mask])
    arc_a_mask = (thFd >= 140) & (thFd <= 200) & True
    arc_b_mask = (thFd >= 220) & (thFd <= 300) & True
    r_arc_mask = rF >= 0.80
    I_full_aa = grid_trap(AF[np.ix_(arc_a_mask, r_arc_mask)], thF[arc_a_mask], rF[r_arc_mask])
    I_full_ab = grid_trap(AF[np.ix_(arc_b_mask, r_arc_mask)], thF[arc_b_mask], rF[r_arc_mask])
    ICS_full_aa = grid_trap(CSF[np.ix_(arc_a_mask, r_arc_mask)], thF[arc_a_mask], rF[r_arc_mask])
    ICS_full_ab = grid_trap(CSF[np.ix_(arc_b_mask, r_arc_mask)], thF[arc_b_mask], rF[r_arc_mask])

    # remainder = full minus (wedge + arcs rectangles), all on the coarse grid
    I_rem = I_full - I_full_wedge - I_full_aa - I_full_ab
    ICS_rem = ICS_full - ICS_full_wedge - ICS_full_aa - ICS_full_ab
    # also need r in [0.05,0.10) sliver of the full grid: included in I_full.
    # but wedge rectangle r from 0.10; remainder thus includes r in [0.05,0.10].

    # ---- Z-boundary sliver subtraction ----
    # subtract integral over r in [R_Z(theta), 1.55]
    th_s = np.deg2rad(np.arange(0, 360, 1.0))
    sliver = 0.0
    sliver_CS = 0.0
    for th in th_s:
        RZ = R_Z(th)
        # pick source grid
        thd = np.degrees(th) % 360
        if 45 <= thd <= 135:
            ths_g, rs_g, Ag, CSg = thf, rf, Af, CSf
            ti = int(round((thd - 45) / 0.25))
        elif 140 <= thd <= 200:
            ths_g, rs_g, Ag, CSg = tha, ra, Aa, CSa
            ti = int(round((thd - 140) / 0.5))
        elif 220 <= thd <= 300:
            ths_g, rs_g, Ag, CSg = thb, rb, Ab, CSb
            ti = int(round((thd - 220) / 0.5))
        else:
            ths_g, rs_g, Ag, CSg = thF, rF, AF, CSF
            ti = int(round(thd / 5.0)) % 72
        prof = Ag[ti, :]
        profCS = CSg[ti, :]
        sel = rs_g >= RZ - 1e-12
        if sel.sum() < 2:
            continue
        rr = rs_g[sel]
        # include RZ as left endpoint via linear interpolation
        rho_rz = np.interp(RZ, rs_g, prof)
        rhoCS_rz = np.interp(RZ, rs_g, profCS)
        rr2 = np.concatenate([[RZ], rr])
        pr2 = np.concatenate([[rho_rz], prof[sel]])
        pc2 = np.concatenate([[rhoCS_rz], profCS[sel]])
        sliver += np.trapezoid(pr2 * rr2, rr2) * np.deg2rad(1.0)
        sliver_CS += np.trapezoid(pc2 * rr2, rr2) * np.deg2rad(1.0)

    # ---- inner disk r <= 0.05: exact node evals ----
    di = load("maps/inner_scan.tsv")
    inner_max_rho = float(np.abs(di[:, 11]).max())
    inner_bound = inner_max_rho * np.pi * 0.05 ** 2

    I_WP = I_fine + I_aa + I_ab + I_rem - sliver
    I_CS = ICS_fine + ICS_aa + ICS_ab + ICS_rem - sliver_CS

    print("================ W4 ASSEMBLY ================")
    print("I_wedge_fine  = %.12e" % I_fine)
    print("I_wedge_mid   = %.12e (domain r<=1.54)" % I_mid)
    print("I_wedge_fine_common = %.12e (r<=1.54)  diff mid-fine = %.3e (rel %.2e)" % (
        I_fine_c, I_fine_c - I_mid, abs(I_fine_c - I_mid) / I_fine))
    print("I_finer_sub   = %.12e" % I_x)
    print("I_fine_sub    = %.12e   (diff %.3e, rel %.3e)" % (
        I_x_coarse, I_x - I_x_coarse, abs(I_x - I_x_coarse) / max(I_x, 1e-300)))
    print("I_arc_a (140-200deg, r .8-1.55) = %.12e" % I_aa)
    print("I_arc_b (220-300deg, r .8-1.55) = %.12e" % I_ab)
    print("I_full_med (all angles, r .05-1.55) = %.12e" % I_full)
    print("I_remainder (coarse, outside fine domains) = %.12e" % I_rem)
    print("Z-boundary sliver subtracted = %.12e" % sliver)
    print("inner disk max rho = %.3e -> bound %.3e" % (inner_max_rho, inner_bound))
    print("---------------------------------------------")
    print("I_WP(0.025) = %.12e" % I_WP)
    print("I_CS corrected = %.12e" % I_CS)
    print("ICS pieces: wedge %.6e arcs %.6e rem %.6e sliver %.3e" % (
        ICS_fine, ICS_aa + ICS_ab, ICS_rem, sliver_CS))
    # error budget numbers
    print("---------------------------------------------")
    print("spatial quadrature: |fine-mid| (common domain) = %.3e (rel %.2e)" % (
        abs(I_fine_c - I_mid), abs(I_fine_c - I_mid) / I_WP))
    print("subdomain finer check: |finer-fine| = %.3e (rel %.2e of sub)" % (
        abs(I_x - I_x_coarse), abs(I_x - I_x_coarse) / I_x))
    print("remainder share: %.2e of total" % (I_rem / I_WP))
    print("sliver share: %.2e of total" % (sliver / I_WP))


if __name__ == "__main__":
    main()
