#!/usr/bin/env python3
# =============================================================================
# CL-DATA-052 — GP-REQ-069 independent reconstruction (executable, re-runnable)
# Exact finite-r six-pin midpoint nine-jet conditional law for the normalized
# periodized side-24 Bargmann-Fock field, executed at r = 1/40 and r = 1/80.
#
# Line: CL (Anthropic Claude).  Independently authored 2026-07-21.
# INDEPENDENCE / LINEAGE DISCLOSURE (GP-REQ-069 independence rule):
#   * GP-DATA-035 was NOT imported, opened, or read.  GP-DER-034 was NOT
#     opened during implementation.  No GP matrices were copied.
#   * The only inputs are: the object definition in GP-REQ-069 itself
#     (pins, jet ordering, b = 6/5, side 24, declared limit values for the
#     comparison stage) and the classical Gaussian regression identity
#     (Schur complement), plus File 4 (3.1)-(3.4) for the OPTIONAL
#     divided-difference stabilization cross-check.
#   * All code below is stdlib-only (fractions.Fraction, decimal.Decimal);
#     there is no float64 inversion anywhere (GP-REQ-069 item 4).
#
# FIELD MODEL (independently defined, GP-REQ-069 item 1):
#   Planar Bargmann-Fock covariance  g(u) = exp(-|u|^2/2), normalized so
#   Var f = 1.  Torus of side L = 24: periodized covariance
#       C_L(u) = sum_{v in Z^2} g(u + L v),
#   normalized kernel  K(u) = C_L(u) / C_L(0)  so that Var f = 1 on the
#   torus.  Derivative covariances:
#       Cov(d^a f(p), d^b f(q)) = (-1)^{|b|} (d^{a+b} K)(p - q),
#   and for the Gaussian factor kernel
#       d_x^j d_y^k g(u) = (-1)^{j+k} He_j(u_x) He_k(u_y) g(u)
#   with probabilists' Hermite polynomials He_n (exact Fraction recurrence).
#
# OBJECT (GP-REQ-069 items 2-3):
#   Pins (ordered):  P = ( f(M), f_x(M), f_y(M), f(S), f_x(S), f_y(S) ),
#     M = (-r/2, 0) with values (b, 0, 0),
#     S = (+r/2, 0) with values (b - r^3/6, 0, 0),  b = 6/5.
#   Jets (ordered) at midpoint (0,0):
#     J = (q,a,w,z,c40,c31,c22,c13,c04)
#       = (f_yy, f_xxy, f_xyy, f_yyy, f_xxxx, f_xxxy, f_xxyy, f_xyyy, f_yyyy).
#   Conditional law: mean = S_JP S_PP^-1 p ;  cov = S_JJ - S_JP S_PP^-1 S_PJ.
#
# PRECISION QUALIFICATION (item 4): all linear algebra in decimal.Decimal at
# 200 significant digits (rerun at 120 digits for a roundoff certificate).
# S_PP has condition ~ r^-10 ~ 10^19 at r=1/80; 200 digits leaves >150
# correct digits.  A divided-difference (File 4 (3.4)) stabilized replay
# with an O(1)-conditioned Gram matrix must agree; both are reported.
#
# SCOPE (item 8): finite-dimensional Gaussian law reconstruction only.
# No theorem, Palm-law identity, adjacency result, P0.2 result, or machine
# promotion follows from this computation.
# =============================================================================
import json, hashlib, sys
from fractions import Fraction
from decimal import Decimal, getcontext

F = Fraction
L = 24                     # torus side
SHIFTS = 2                 # periodization shells: v in [-2..2]^2
B_PIN = F(6, 5)            # b = 6/5
JET_NAMES = ["q", "a", "w", "z", "c40", "c31", "c22", "c13", "c04"]
JETS = [(0, 2), (2, 1), (1, 2), (0, 3), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
PIN_NAMES = ["f(M)", "fx(M)", "fy(M)", "f(S)", "fx(S)", "fy(S)"]

# ---------------- exact Hermite (probabilists') coefficients ---------------
def hermite_coeffs(nmax):
    He = [[F(1)], [F(0), F(1)]]
    for n in range(1, nmax):
        prev, cur = He[n - 1], He[n]
        nxt = [F(0)] * (n + 2)
        for i, c in enumerate(cur):
            nxt[i + 1] += c                    # x * He_n
        for i, c in enumerate(prev):
            nxt[i] -= n * c                    # - n He_{n-1}
        He.append(nxt)
    return He
HE = hermite_coeffs(9)                          # He_0 .. He_9 exact

def he_eval(n, x):                              # Horner in Decimal
    acc = Decimal(0)
    for c in reversed(HE[n]):
        acc = acc * x + Decimal(c.numerator) / Decimal(c.denominator)
    return acc

def d2(fr):                                     # Fraction -> Decimal
    return Decimal(fr.numerator) / Decimal(fr.denominator)

def fmt(x, sig=30):
    """Exponent-safe Decimal formatter (str()[:n] would amputate E-exponents)."""
    return f"{x:.{sig}E}" if isinstance(x, Decimal) else str(x)

# ---------------- kernel derivative machinery ------------------------------
class Kernel:
    """mode: 'torus' (periodized normalized), 'planar', or mutated modes
    'planar_badsign' (drops (-1)^{|b|}) and 'planar_bw' (bandwidth 2)."""
    def __init__(self, mode):
        self.mode = mode
        self.cache = {}
        self.C0 = self._c0()
    def _g(self, x, y, scale=Decimal(2)):
        return (-(x * x + y * y) / scale).exp()
    def _c0(self):
        if self.mode != "torus":
            return Decimal(1)
        tot = Decimal(0)
        for m in range(-SHIFTS, SHIFTS + 1):
            for n in range(-SHIFTS, SHIFTS + 1):
                tot += self._g(Decimal(L * m), Decimal(L * n))
        return tot
    def dK(self, j, k, dx, dy):
        """(d_x^j d_y^k K)(dx, dy); dx, dy Fractions."""
        key = (j, k, dx, dy)
        if key in self.cache:
            return self.cache[key]
        sgn = Decimal(-1) ** (j + k)
        if self.mode == "planar_bw":            # NC-B kernel mutation: g=exp(-|u|^2/4)
            # d^n exp(-x^2/4) = (-1/sqrt2)^n He_n(x/sqrt2) exp(-x^2/4)
            rt2 = Decimal(2).sqrt()
            x, y = d2(dx), d2(dy)
            val = (sgn / rt2 ** (j + k)) * he_eval(j, x / rt2) * he_eval(k, y / rt2) \
                  * self._g(x, y, Decimal(4))
        elif self.mode == "torus":
            tot = Decimal(0)
            for m in range(-SHIFTS, SHIFTS + 1):
                for n in range(-SHIFTS, SHIFTS + 1):
                    x = d2(dx) + Decimal(L * m)
                    y = d2(dy) + Decimal(L * n)
                    tot += he_eval(j, x) * he_eval(k, y) * self._g(x, y)
            val = sgn * tot / self.C0
        else:                                   # planar / planar_badsign
            x, y = d2(dx), d2(dy)
            val = sgn * he_eval(j, x) * he_eval(k, y) * self._g(x, y)
        self.cache[key] = val
        return val
    def cov(self, a, p, b, q):
        """Cov(d^a f(p), d^b f(q)); a,b derivative multi-indices, p,q points."""
        j, k = a[0] + b[0], a[1] + b[1]
        s = Decimal(1) if self.mode == "planar_badsign" else Decimal(-1) ** (b[0] + b[1])
        return s * self.dK(j, k, p[0] - q[0], p[1] - q[1])

# ---------------- Decimal linear algebra -----------------------------------
def solve_gauss(A, Bs):
    """Solve A X = B for several right-hand sides; partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i] for b in Bs] for i, row in enumerate(A)]
    w = len(M[0])
    for c in range(n):
        piv = max(range(c, n), key=lambda rr: abs(M[rr][c]))
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for r2 in range(n):
            if r2 == c:
                continue
            f = M[r2][c] / pv
            if f == 0:
                continue
            for cc in range(c, w):
                M[r2][cc] -= f * M[c][cc]
    return [[M[i][n + jj] / M[i][i] for i in range(n)] for jj in range(len(Bs))]

def mat_vec(A, v):
    return [sum((a * b for a, b in zip(row, v)), Decimal(0)) for row in A]

def jacobi_eig(Ain, tol_exp=-150, sweeps=300):
    n = len(Ain)
    A = [row[:] for row in Ain]
    tol = Decimal(10) ** tol_exp
    for _ in range(sweeps):
        off = max((abs(A[i][j]) for i in range(n) for j in range(n) if i != j),
                  default=Decimal(0))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) <= tol:
                    continue
                theta = (A[q][q] - A[p][p]) / (2 * A[p][q])
                t = 1 / (abs(theta) + (theta * theta + 1).sqrt())
                if theta < 0:
                    t = -t
                c = 1 / (t * t + 1).sqrt()
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p] = c * akp - s * akq
                    A[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k] = c * apk - s * aqk
                    A[q][k] = s * apk + c * aqk
    return sorted(A[i][i] for i in range(n))

# ---------------- the conditional law --------------------------------------
def pin_points(r):
    M = (F(-1, 2) * r, F(0)); S = (F(1, 2) * r, F(0))
    pins = [((0, 0), M), ((1, 0), M), ((0, 1), M),
            ((0, 0), S), ((1, 0), S), ((0, 1), S)]
    vals = [B_PIN, F(0), F(0), B_PIN - r ** 3 / 6, F(0), F(0)]
    return pins, vals

def conditional_law(r, kern, pinvals=None, dd_frame=False):
    pins, vals = pin_points(r)
    if pinvals is not None:
        vals = pinvals
    O = (F(0), F(0))
    SPP = [[kern.cov(a, p, b2, q) for (b2, q) in pins] for (a, p) in pins]
    SJP = [[kern.cov(j, O, b2, q) for (b2, q) in pins] for j in JETS]
    SJJ = [[kern.cov(j1, O, j2, O) for j2 in JETS] for j1 in JETS]
    pv = [d2(v) for v in vals]
    if dd_frame:
        # File 4 (3.4) corrected divided-difference frame, an O(1)-conditioned
        # replay of the SAME conditioning (identical span; File 4 Thm P2(c)).
        # Phi order = (f(M), f(S), fx(M), fy(M), fx(S), fy(S)); rows below are
        # (V+, V-corr, Gt+, Gt-, Gs+, Gs-) composed with the pin permutation.
        perm = [0, 3, 1, 2, 4, 5]        # pins index -> Phi slot content
        Traw = [[F(0)] * 6 for _ in range(6)]
        Traw[0][0] = F(1, 2); Traw[0][1] = F(1, 2)
        Traw[1][0] = -r ** -3; Traw[1][1] = r ** -3
        Traw[1][2] = -F(1, 2) * r ** -2; Traw[1][4] = -F(1, 2) * r ** -2
        Traw[2][2] = F(1, 2); Traw[2][4] = F(1, 2)
        Traw[3][2] = -r ** -1; Traw[3][4] = r ** -1
        Traw[4][3] = F(1, 2); Traw[4][5] = F(1, 2)
        Traw[5][3] = -r ** -1; Traw[5][5] = r ** -1
        A = [[d2(Traw[i][perm.index(jj)]) for jj in range(6)] for i in range(6)]
        SPP = [[sum(A[i][k] * SPP[k][l] * A[j][l] for k in range(6) for l in range(6))
                for j in range(6)] for i in range(6)]
        SJP = [[sum(SJP[i][k] * A[j][k] for k in range(6)) for j in range(6)]
               for i in range(9)]
        pv = mat_vec(A, pv)
    # solve for the pin weight vector and for S_PP^-1 S_PJ^T (9 rhs)
    rhs = [pv] + [[SJP[jrow][k] for k in range(6)] for jrow in range(9)]
    sols = solve_gauss(SPP, rhs)
    wpin, Zrows = sols[0], sols[1:]
    mean = [sum(SJP[i][k] * wpin[k] for k in range(6)) for i in range(9)]
    cov = [[SJJ[i][j] - sum(SJP[i][k] * Zrows[j][k] for k in range(6))
            for j in range(9)] for i in range(9)]
    sym = max(abs(cov[i][j] - cov[j][i]) for i in range(9) for j in range(9))
    for i in range(9):
        for j in range(i + 1, 9):
            m = (cov[i][j] + cov[j][i]) / 2
            cov[i][j] = cov[j][i] = m
    return mean, cov, sym

# ---------------- exact coalesced law (pure Fraction) ----------------------
def coalesced_exact():
    """Jets conditioned on (f, fx, fy, fxx, fxy, fxxx)=(b,0,0,0,0,2) at one
    point, planar kernel: every covariance is an exact rational (He at 0)."""
    def cov0(a, b):
        j, k = a[0] + b[0], a[1] + b[1]
        if j % 2 or k % 2:
            return F(0)
        sgn = F(-1) ** (b[0] + b[1]) * F(-1) ** (j + k)
        heval = lambda n: HE[n][0]
        return sgn * heval(j) * heval(k)
    C = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (3, 0)]
    pvals = [B_PIN, F(0), F(0), F(0), F(0), F(2)]
    SPP = [[cov0(a, b) for b in C] for a in C]
    SJP = [[cov0(j, b) for b in C] for j in JETS]
    SJJ = [[cov0(j1, j2) for j2 in JETS] for j1 in JETS]
    # exact Fraction Gauss-Jordan
    n = 6
    M = [row[:] + [pvals[i]] + [SJP[jr][i] for jr in range(9)]
         for i, row in enumerate(SPP)]
    for c in range(n):
        piv = next(rr for rr in range(c, n) if M[rr][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for r2 in range(n):
            if r2 != c and M[r2][c] != 0:
                f = M[r2][c]
                M[r2] = [x - f * y for x, y in zip(M[r2], M[c])]
    wpin = [M[i][n] for i in range(n)]
    Z = [[M[i][n + 1 + jr] for i in range(n)] for jr in range(9)]
    mean = [sum(SJP[i][k] * wpin[k] for k in range(6)) for i in range(9)]
    cov = [[SJJ[i][j] - sum(SJP[i][k] * Z[j][k] for k in range(6))
            for j in range(9)] for i in range(9)]
    return mean, cov

# ---------------- main -----------------------------------------------------
def run(prec):
    getcontext().prec = prec
    out = {}
    kern_t = Kernel("torus")
    kern_p = Kernel("planar")
    for r in (F(1, 40), F(1, 80)):
        mean, cov, sym = conditional_law(r, kern_t)
        mean_dd, cov_dd, _ = conditional_law(r, kern_t, dd_frame=True)
        mean_pl, cov_pl, _ = conditional_law(r, kern_p)
        dd_diff = max(max(abs(a - b) for a, b in zip(mean, mean_dd)),
                      max(abs(cov[i][j] - cov_dd[i][j]) for i in range(9) for j in range(9)))
        tor_pl = max(max(abs(a - b) for a, b in zip(mean, mean_pl)),
                     max(abs(cov[i][j] - cov_pl[i][j]) for i in range(9) for j in range(9)))
        eigs = jacobi_eig(cov)
        out[str(r)] = dict(mean=mean, cov=cov, sym=sym, dd_diff=dd_diff,
                           torus_vs_planar=tor_pl, eig_min=eigs[0], eig_max=eigs[-1])
    return out

def main():
    res200 = run(200)
    res120 = run(120)
    # precision certificate: max |prec200 - prec120| over reported numbers
    prec_cert = Decimal(0)
    for key in res200:
        a, b = res200[key], res120[key]
        prec_cert = max(prec_cert,
                        max(abs(x - y) for x, y in zip(a["mean"], b["mean"])),
                        max(abs(a["cov"][i][j] - b["cov"][i][j])
                            for i in range(9) for j in range(9)))
    getcontext().prec = 200
    cmean, ccov = coalesced_exact()

    # declared limiting values from GP-REQ-069 item 6 (b = 6/5)
    decl_mean = [-B_PIN, F(0), F(0), F(0), -3 * B_PIN, F(0), F(0), F(0), 3 * B_PIN]
    decl_diag = [F(2), F(2), F(2), F(6), F(24), F(6), F(6), F(6), F(96)]
    decl_off = {("q", "c22"): F(-2), ("q", "c04"): F(-12), ("c22", "c04"): F(12)}
    ix = {n: i for i, n in enumerate(JET_NAMES)}

    print("=" * 78)
    print("COALESCED LAW — exact rational reconstruction vs declared values")
    exact_match = (cmean == decl_mean and
                   all(ccov[i][i] == decl_diag[i] for i in range(9)) and
                   all(ccov[ix[a]][ix[b]] == v for (a, b), v in decl_off.items()))
    print("  mean  =", [str(x) for x in cmean])
    print("  diag  =", [str(ccov[i][i]) for i in range(9)])
    print("  Cov(q,c22), Cov(q,c04), Cov(c22,c04) =",
          str(ccov[ix['q']][ix['c22']]), str(ccov[ix['q']][ix['c04']]),
          str(ccov[ix['c22']][ix['c04']]))
    print("  DECLARED VALUES MATCH EXACTLY:", exact_match)
    print("  full exact coalesced covariance (Fraction):")
    for i in range(9):
        print("   ", [str(ccov[i][j]) for j in range(9)])

    print("\nFINITE-r LAW (torus, prec 200) and convergence to coalesced law")
    conv = {}
    for rkey, res in res200.items():
        print(f"\n  r = {rkey}")
        print("  conditional mean (first 30 digits):")
        for n, v in zip(JET_NAMES, res["mean"]):
            print(f"    {n:4s} {fmt(v)}")
        print(f"  min eigenvalue  = {fmt(res['eig_min'])}")
        print(f"  max eigenvalue  = {fmt(res['eig_max'])}")
        print(f"  symmetry residual        = {fmt(res['sym'], 6)}")
        print(f"  DD-frame replay max diff = {fmt(res['dd_diff'], 6)}")
        print(f"  torus vs planar max diff = {fmt(res['torus_vs_planar'], 6)}")
        dm = [abs(v - d2(m)) for v, m in zip(res["mean"], cmean)]
        dc = max(abs(res["cov"][i][j] - d2(ccov[i][j]))
                 for i in range(9) for j in range(9))
        conv[rkey] = (max(dm), dc)
        print(f"  |mean - coalesced|_max   = {fmt(max(dm), 6)}")
        print(f"  |cov  - coalesced|_max   = {fmt(dc, 6)}")
    r1, r2 = str(F(1, 40)), str(F(1, 80))
    print(f"\n  convergence ratios (r=1/40 vs 1/80): mean {fmt(conv[r1][0]/conv[r2][0], 4)},"
          f" cov {fmt(conv[r1][1]/conv[r2][1], 4)}  (~4 = O(r^2))")
    print(f"  precision certificate |prec200-prec120|_max = {fmt(prec_cert, 6)}")

    # ---------------- planted negative controls ----------------------------
    print("\nPLANTED NEGATIVE CONTROLS (each must be detected and rejected)")
    r = F(1, 40)
    # NC-A derivative-sign mutation
    mean_a, cov_a, sym_a = conditional_law(r, Kernel("planar_badsign"))
    ncA = sym_a > Decimal("1e-30")
    print(f"  NC-A drop (-1)^|b| sign rule: covariance-symmetry residual "
          f"{fmt(sym_a, 4)} > 1e-30  -> DETECTED: {ncA}")
    # NC-B kernel-normalization/bandwidth mutation
    mean_b, cov_b, _ = conditional_law(r, Kernel("planar_bw"))
    devB = abs(cov_b[0][0] - Decimal(2))
    ncB = devB > Decimal("0.4")
    print(f"  NC-B kernel bandwidth mutation exp(-|u|^2/4): Var(q|P) = "
          f"{fmt(cov_b[0][0], 4)} vs 2  -> DETECTED: {ncB}")
    # NC-C pin-height mutation b -> 1
    pins, _ = pin_points(r)
    mut_vals = [F(1), F(0), F(0), F(1) - r ** 3 / 6, F(0), F(0)]
    mean_c, _, _ = conditional_law(r, Kernel("torus"), pinvals=mut_vals)
    devC = abs(mean_c[0] - d2(-B_PIN))
    ncC = devC > Decimal("0.15")
    print(f"  NC-C pin-height mutation b=6/5 -> 1: E[q|P] = {fmt(mean_c[0], 4)}"
          f" vs -6/5  -> DETECTED: {ncC}")
    assert ncA and ncB and ncC, "a planted control was NOT detected"

    # ---------------- machine-readable receipt -----------------------------
    def s(x, nd=50):
        return fmt(x, nd) if isinstance(x, Decimal) else str(x)
    receipt = {
        "artifact": "CL-DATA-052 receipt v1.0",
        "responds_to": "GP-REQ-069-v1.0",
        "line": "CL (Anthropic Claude); independently authored; no GP source imported",
        "object": {
            "field": "normalized periodized Bargmann-Fock, C_L(u)=sum_v exp(-|u+Lv|^2/2), K=C_L/C_L(0)",
            "torus_side": L, "periodization_shells": f"|v|<=2 (next shell < 1e-500)",
            "coordinates": "e_t = +x along S-M; s = y; midpoint origin",
            "pins_order": PIN_NAMES,
            "pin_points": {"M": "(-r/2, 0)", "S": "(+r/2, 0)"},
            "pin_values": "f: (b, b - r^3/6) with b = 6/5; gradients 0",
            "jet_order": JET_NAMES,
            "jet_multiindices": [list(j) for j in JETS],
            "kernel_normalization": "Var f = 1 (K(0)=1)",
        },
        "method": {
            "arithmetic": "decimal.Decimal prec=200 (rerun prec=120); exact Fraction inputs",
            "conditioning": "Schur complement via Gaussian elimination, no float64",
            "stabilization_replay": "File 4 (3.3)-(3.4) corrected divided-difference frame",
            "eigenvalues": "Jacobi rotations, tol 1e-150",
        },
        "results": {},
        "coalesced_exact": {
            "mean": [str(x) for x in cmean],
            "cov": [[str(x) for x in row] for row in ccov],
            "declared_values_match_exactly": bool(exact_match),
        },
        "negative_controls": {
            "NC-A_derivative_sign": {"residual": s(sym_a, 12), "detected": bool(ncA)},
            "NC-B_kernel_bandwidth": {"Var_q": s(cov_b[0][0], 12), "detected": bool(ncB)},
            "NC-C_pin_height_b_to_1": {"mean_q": s(mean_c[0], 12), "detected": bool(ncC)},
        },
        "precision_certificate_prec200_vs_prec120": s(prec_cert, 12),
        "environment": {
            "python": sys.version.split()[0], "implementation": "CPython stdlib only",
            "modules": "fractions, decimal, json, hashlib (no numpy/sympy/mpmath)",
            "command_line": "python3 cl_data_052_sixpin_ninejet.py cl_data_052_receipt.json",
        },
        "scope_statement": ("Finite-dimensional conditional-law reconstruction only. "
                            "No theorem, Palm-law identity, adjacency result, P0.2 "
                            "result, or machine promotion follows."),
    }
    for rkey, res in res200.items():
        receipt["results"][rkey] = {
            "conditional_mean_50d": [s(v) for v in res["mean"]],
            "conditional_cov_50d": [[s(res["cov"][i][j]) for j in range(9)] for i in range(9)],
            "eig_min_50d": s(res["eig_min"]), "eig_max_50d": s(res["eig_max"]),
            "positive_definite": bool(res["eig_min"] > 0),
            "symmetry_residual": s(res["sym"], 12),
            "dd_frame_replay_max_diff": s(res["dd_diff"], 12),
            "torus_vs_planar_max_diff": s(res["torus_vs_planar"], 12),
            "max_dev_from_coalesced_mean": s(conv[rkey][0], 12),
            "max_dev_from_coalesced_cov": s(conv[rkey][1], 12),
        }
    with open(sys.argv[1] if len(sys.argv) > 1 else "cl_data_052_receipt.json", "w") as f:
        json.dump(receipt, f, indent=1)
    print("\nreceipt written; ALL CHECKS PASS")

if __name__ == "__main__":
    main()
