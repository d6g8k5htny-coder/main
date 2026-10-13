### HEADER
"""SIDE24's coefficient in d = 5: the birth-integrated cone moment D_4, certified, with c_{5,ref} and c_{5,24}.

Standard library only. Run from anywhere:  python -B -S cone_moment_d5.py [--mutant NAME]
Every printed interval [lo, hi] is a rigorous enclosure (interval arithmetic over `decimal` at 60 digits with directed
rounding; exact integer roots; series with explicit remainders; one-dimensional Taylor quadrature with the omitted tail
bounded by Cauchy's estimate at the cell centre and a geometric series).  Nothing is fitted, sampled or extrapolated.

Object.  For the reference law of coefficients/side24_v1 in dimension d = m + 1, A = Q + sqrt(2/3) Z I_m (Q the m x m GOE
of density proportional to exp(-tr Q^2/4)), D_m = E[det(A)^2 1{A < 0}], and (coefficients/side24_d4_20260930, NOTE
section 1)

    D_m = sqrt(3/(m+3)) / Z_m . int_{mu in (-inf,0)^m} prod mu_i^2 prod_{i<j} |mu_i - mu_j| exp(-|mu|^2/4 + S^2/(4(m+3))) dmu.

General decoupling (NOTE section 1).  In the ordered-sector coordinates mu_m = -a, mu_{m-j} = -(a + t_j),
t_j = p_1 + ... + p_j, with T = sum_j t_j and u = t - (T/(m-1)) 1, the a-integral has alpha = 3m/(4(m+3)),
beta = 3T/(2(m+3)), and after it the erfc branch carries exp(-T^2/(4m(m-1)) - |u|^2/4) with erfc(beta/(2 sqrt alpha)),
the polynomial branch exp(-T^2/((m-1)(m+3)) - |u|^2/4).  For m = 4 (this record) the u-integral runs over the triangle
p_2, p_3 > 0, 2 p_2 + p_3 < T with |u|^2/4 = (p_2^2 + p_2 p_3 + p_3^2)/6; in v = 2 p_2 + p_3 this is
(1/2) int_0^T dv e^{-v^2/24} int_0^v dp_3 (.) e^{-p_3^2/8}, the inner integral is exact through
J_k(v) = int_0^v q^k e^{-q^2/8} dq, and the v-integral is carried as the cumulative integrals
L_a(T) = int_0^T e^{-v^2/24} q_a(v) dv (section 2), certified cell by cell together with the outer T-integral (section 3).
Then D_4 = sqrt(3/7) . 4!/Z_4 . (1/6) I_4.  The same machinery evaluates Mehta's constant Z_4 = 1536 pi (a rule), and the
m = 2, 3 pipelines of the d = 4 record are re-run here (29/6 - sqrt6; the D_3 enclosure of Math-#199).

Coefficient.  [LP] (15.2): c_{d,ref} = Gamma(7/6) (3/2)^{1/3} |S^{d-1}| D_{d-1} / (sqrt3 sqrt pi (2 pi)^d); d = 5:
c_{5,ref} = Gamma(7/6) (3/2)^{1/3} D_4 / (12 sqrt3 pi^{7/2}) with |S^4| = 8 pi^2/3; c_{5,24} by the SIDE24 section 3-4
covariance sandwich with the d = 5 constants (NOTE section 4).  Scientific effect NONE.
"""
import argparse
import itertools
import json
import math
import sys
import time
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN, setcontext
from fractions import Fraction as Fr

MUTANTS = ("shift-variance", "vandermonde", "erfc-branch", "remainder-dropped", "mehta", "sphere-area", "image-shells",
           "sandwich-exponent", "recurrence", "jacobian", "cumulative-half")
MUT = None

### M4
def t_mul_poly(pc, tm):
    """Product of a short coefficient list pc (a polynomial's Taylor coefficients) with a full Taylor list tm."""
    K = len(tm) - 1
    out = []
    for n in range(K + 1):
        acc = Iv(ZERO)
        for i in range(min(n, len(pc) - 1) + 1):
            acc = acc + pc[i] * tm[n - i]
        out.append(acc)
    return out


def poly_taylor_short(coeffs, s0):
    """Taylor coefficients at s0 of sum_j coeffs[j] s^j, all degrees (exact rationals as intervals)."""
    deg = len(coeffs) - 1
    out = []
    for n in range(deg + 1):
        v = Fr(0)
        for j in range(n, deg + 1):
            v += coeffs[j] * math.comb(j, n) * Fr(s0) ** (j - n)
        out.append(Iv.frac(v))
    return out


def gauss_moment(j, gamma):
    """int_0^inf v^j e^{-gamma v^2} dv = (1/2) gamma^{-(j+1)/2} Gamma((j+1)/2), as an interval."""
    g = Iv.frac(gamma)
    a2 = j + 1
    gpow = ipow(g, a2 // 2) * (g.sqrt() if a2 % 2 else Iv(ONE))
    return gamma_half_integer(a2) / gpow / 2


def organize_tvp(P):
    """Polynomial dict in (T, v, p3) -> {a: {k: [coefficients in v by power]}} (a = T power, k = p3 power)."""
    out = {}
    for (i, j, k), c in P.items():
        out.setdefault(i, {}).setdefault(k, {})[j] = c
    return {a: {k: [d.get(j, Fr(0)) for j in range(max(d) + 1)] for k, d in ks.items()} for a, ks in out.items()}


def m4_setup():
    """The two branches of I_4 after the exact a- and p3-integrations, organised for the (T, L_a) quadrature."""
    m = 4
    alpha, beta, PE, PR, expE, rest = a_integrated_terms(m)
    # p1 = (T - 2 p2 - p3)/3 -> variables (T, p2, p3)
    T, p2, p3 = pvar(0, 3), pvar(1, 3), pvar(2, 3)
    p1 = pscale(padd(T, padd(pscale(p2, Fr(-2)), pscale(p3, Fr(-1)))), Fr(1, 3))
    img1 = [p1, p2, p3]
    PE1, PR1 = psubst(PE, img1, 3), psubst(PR, img1, 3)
    eE1, eR1, b1 = psubst(expE, img1, 3), psubst(rest, img1, 3), psubst(beta, img1, 3)
    # p2 = (v - p3)/2 -> variables (T, v, p3)
    T2, v, q = pvar(0, 3), pvar(1, 3), pvar(2, 3)
    img2 = [T2, pscale(padd(v, pscale(q, Fr(-1))), Fr(1, 2)), q]
    PE2, PR2 = psubst(PE1, img2, 3), psubst(PR1, img2, 3)
    eE2, eR2, b2 = psubst(eE1, img2, 3), psubst(eR1, img2, 3), psubst(b1, img2, 3)
    diag = {(2, 0, 0), (0, 2, 0), (0, 0, 2)}
    for e in (eE2, eR2):
        if set(e) != diag or e[(0, 2, 0)] != Fr(1, 24) or e[(0, 0, 2)] != Fr(1, 8):
            raise ValueError("m = 4 decoupling failed: " + repr(e))
    if set(b2) != {(1, 0, 0)}:
        raise ValueError("beta is not a multiple of T: " + repr(b2))
    gE, gR = eE2[(2, 0, 0)], eR2[(2, 0, 0)]
    bT = b2[(1, 0, 0)]
    kappa = Iv.frac(bT) / (2 * Iv.frac(alpha).sqrt())
    kappa2 = bT * bT / (4 * alpha)                                                 # rational: erfc(kappa T) <= e^{-kappa2 T^2}
    constE = Iv.frac(Fr(1, 2)) * (PI / Iv.frac(alpha)).sqrt()
    if MUT == "erfc-branch":
        constE = constE * Iv.frac(Fr(1, 2))
    jac = Fr(1, 6) if MUT != "jacobian" else Fr(1, 2)                              # dp1 dp2 dp3 = (1/3)(1/2) dT dv dp3
    pref = (Iv.frac(Fr(3, m + 3))).sqrt() / mehta_Z(m) * math.factorial(m) * Iv.frac(jac)
    branches = []
    for name, P, g, const, extra, gtail in (("E", PE2, gE, pref * constE, [ErfLin(kappa, -1)], gE + kappa2),
                                            ("R", PR2, gR, pref, [], gR)):
        inner = organize_tvp(P)
        # tail: |sum_a T^a L_a(T)| <= sum_a T^a Lbar_a, Lbar_a = sum_k sum_j |e_{a,k,j}| J_k(inf) int_0^inf v^j e^{-v^2/24} dv
        tail = {}
        for a, ks in inner.items():
            lb = Iv(ZERO)
            for k, ek in ks.items():
                jinf = Jk(k).infinity()
                for j, c in enumerate(ek):
                    if c:
                        lb = lb + Iv.frac(abs(c)) * jinf * gauss_moment(j, Fr(1, 24))
            tail[a] = Fr(lb.hi)
        tail_poly = [tail.get(a, Fr(0)) for a in range(max(tail) + 1)]
        branches.append({"name": name, "const": const, "prefactors": [Gauss(g)] + extra, "inner": inner,
                         "tail_poly": tail_poly, "gamma": gtail})
    return branches


def z4_setup():
    """Mehta's Z_4 through the same triangle machinery: Z_4 = 4! sqrt pi (1/6) int dT e^{-T^2/48} int_0^T dv e^{-v^2/24}
    int_0^v dp3 V e^{-p3^2/8}, V the Vandermonde p1 p2 p3 (p1+p2)(p2+p3)(p1+p2+p3) with p1 = (T - v)/3, p2 = (v - p3)/2."""
    T, v, q = pvar(0, 3), pvar(1, 3), pvar(2, 3)
    p1 = pscale(padd(T, pscale(v, Fr(-1))), Fr(1, 3))
    p2 = pscale(padd(v, pscale(q, Fr(-1))), Fr(1, 2))
    p3 = q
    V = pconst(1, 3)
    for f in (p1, p2, p3, padd(p1, p2), padd(p2, p3), padd(padd(p1, p2), p3)):
        V = pmul(V, f)
    jac = Fr(1, 6) if MUT != "jacobian" else Fr(1, 2)
    const = Iv.frac(24 * jac) * SQRTPI
    inner = organize_tvp(V)
    tail = {}
    for a, ks in inner.items():
        lb = Iv(ZERO)
        for k, ek in ks.items():
            jinf = Jk(k).infinity()
            for j, c in enumerate(ek):
                if c:
                    lb = lb + Iv.frac(abs(c)) * jinf * gauss_moment(j, Fr(1, 24))
        tail[a] = Fr(lb.hi)
    tail_poly = [tail.get(a, Fr(0)) for a in range(max(tail) + 1)]
    return [{"name": "Z4", "const": const, "prefactors": [Gauss(Fr(1, 48))], "inner": inner, "tail_poly": tail_poly,
             "gamma": Fr(1, 48)}]


def enclose_m4(branches, K=None, rho=None, cells=None, remainder=True):
    """int_0^{T_max} sum_branches const . prod(prefactors)(T) . sum_a T^a L_a(T) dT + tail, with
    L_a(T) = int_0^T e^{-v^2/24} q_a(v) dv, q_a = sum_k e_{a,k}(v) J_k(v), carried as Taylor factors: at each cell centre
    s_0 the value L_a(s_0) is the accumulated certified integral over [0, s_0] (full cells so far plus the left half of
    this cell) and the higher coefficients are those of the integrand divided by n.  Remainders as in enclose_integral
    (Cauchy at the centre, geometric factor); the disc bound of L_a is |L_a(s_0)| + rho . M_g."""
    K = K_ORDER if K is None else K
    rho = RHO if rho is None else rho
    cells = CELLS_M4 if cells is None else cells
    rho_pow = Iv.frac(Fr(rho) ** (K + 1))
    G24 = Gauss(Fr(1, 24))
    L = {(bi, a): Iv(ZERO) for bi, br in enumerate(branches) for a in br["inner"]}
    total = Iv(ZERO)
    for lo, hi in cells:
        h = Fr(hi) - Fr(lo)
        half = h / 2
        s0 = Fr(lo) + half
        if half >= Fr(rho):
            raise ValueError("cell half-width must be below the Cauchy radius")
        geom = 1 / (1 - half / Fr(rho))
        full_even = {n: Iv.frac(2 * half ** (n + 1) / (n + 1)) for n in range(0, K + 1, 2)}
        left = [Iv.frac((-1) ** n * half ** (n + 1) / (n + 1)) for n in range(K + 1)]      # int_{-h/2}^0 x^n dx
        rem_half = Iv.frac(half ** (K + 2) / (K + 2) * geom)
        rem_full = rem_half * 2
        g24 = G24.coefficients(s0, K)
        M24 = G24.disc_bound(s0, rho)
        outer_val, M_outer = Iv(ZERO), Iv(ZERO)
        for bi, br in enumerate(branches):
            acc, M_acc = None, Iv(ZERO)
            for a, ks in br["inner"].items():
                qsum, M_q = None, Iv(ZERO)
                for k, ek in ks.items():
                    jk = Jk(k)
                    prod = t_mul_poly(poly_taylor_short(ek, s0), jk.coefficients(s0, K))
                    qsum = prod if qsum is None else t_add(qsum, prod)
                    M_q = M_q + Poly(ek).disc_bound(s0, rho) * jk.disc_bound(s0, rho)
                g = t_mul(qsum, g24)
                M_g = M_q * M24
                Ihalf, Ifull = Iv(ZERO), Iv(ZERO)
                for n in range(K + 1):
                    Ihalf = Ihalf + g[n] * left[n]
                    if n % 2 == 0:
                        Ifull = Ifull + g[n] * full_even[n]
                if remainder:
                    Rh = (M_g / rho_pow * rem_half).hi
                    Ihalf = Ihalf + Iv(-Rh, Rh)
                    Ifull = Ifull + Iv(-2 * Rh, 2 * Rh)
                Lc = L[(bi, a)] + (Ihalf if MUT != "cumulative-half" else Iv(ZERO))
                L[(bi, a)] = L[(bi, a)] + Ifull
                cum = [Lc] + [g[n - 1] / n for n in range(1, K + 1)]
                M_cum = Iv(Lc.mag()) + Iv.frac(rho) * M_g
                ta = poly_taylor_short([Fr(0)] * a + [Fr(1)], s0)
                term = t_mul_poly(ta, cum)
                M_term = Poly([Fr(0)] * a + [Fr(1)]).disc_bound(s0, rho) * M_cum
                acc = term if acc is None else t_add(acc, term)
                M_acc = M_acc + M_term
            M = M_acc
            for f in br["prefactors"]:
                acc = t_mul(acc, f.coefficients(s0, K))
                M = M * f.disc_bound(s0, rho)
            for n in range(0, K + 1, 2):
                outer_val = outer_val + br["const"] * acc[n] * full_even[n]
            M_outer = M_outer + Iv(br["const"].mag()) * M
        if remainder:
            R = (M_outer / rho_pow * rem_full).hi
            outer_val = outer_val + Iv(-R, R)
        total = total + outer_val
    Tmax = Fr(cells[-1][1])
    tail = Iv(ZERO)
    for br in branches:
        g = Iv.of(br["gamma"])
        x = g * Iv.frac(Tmax * Tmax)
        for j, cj in enumerate(br["tail_poly"]):
            if cj == 0:
                continue
            a = Fr(j + 1, 2)
            if a.denominator == 2:
                xa1 = ipow(x, int(a - Fr(1, 2))) * x.sqrt() / x
                gpow = ipow(g, int(a)) * g.sqrt()
            else:
                xa1 = ipow(x, int(a) - 1)
                gpow = ipow(g, int(a))
            up = xa1 * (-x).exp()
            if a > 1:
                up = up / (1 - Iv.frac(a - 1) / x)
            tail = tail + Iv(br["const"].mag()) * Iv.frac(abs(cj)) * up / gpow / 2
    return total + Iv(-tail.hi, tail.hi)


### FLOATS
def mehta_Z_float(m):
    return 2 ** (m / 2 + m * (m - 1) / 4) * (2 * math.pi) ** (m / 2) * math.prod(math.gamma(1 + j / 2) for j in range(1, m + 1)) / math.gamma(1.5) ** m


def float_orthant(m, T, n):
    coef = math.sqrt(3 / (m + 3)) / mehta_Z_float(m)
    h = T / n
    grid = [-(i + 0.5) * h for i in range(n)]
    total = 0.0
    for mu in itertools.product(grid, repeat=m):
        S = sum(mu)
        w = 1.0
        for x in mu:
            w *= x * x
        for i in range(m):
            for j in range(i + 1, m):
                w *= abs(mu[i] - mu[j])
        total += w * math.exp(-sum(x * x for x in mu) / 4 + S * S / (4 * (m + 3)))
    return coef * total * h ** m


def float_D4_aint(T=26.0, n=56):
    """m = 4 after the exact a-integral only: 3-D midpoint quadrature over (p1, p2, p3); independent of the (T, v)
    substitutions, the J_k and the cumulative integrals."""
    m = 4
    alpha = 3 * m / (4 * (m + 3))
    def G_list(beta, nmax=8):
        G0 = 0.5 * math.sqrt(math.pi / alpha) * math.exp(beta * beta / (4 * alpha)) * math.erfc(beta / (2 * math.sqrt(alpha)))
        G = [G0, (1 - beta * G0) / (2 * alpha)]
        for k in range(1, nmax):
            G.append((k * G[k - 1] - beta * G[k]) / (2 * alpha))
        return G
    def pm(u, v):
        out = [0.0] * (len(u) + len(v) - 1)
        for i, x in enumerate(u):
            for j, y in enumerate(v):
                out[i + j] += x * y
        return out
    h = T / n
    tot = 0.0
    for i in range(n):
        p1 = (i + 0.5) * h
        for j in range(n):
            p2 = (j + 0.5) * h
            for k in range(n):
                p3 = (k + 0.5) * h
                t1, t2, t3 = p1, p1 + p2, p1 + p2 + p3
                Tsum = t1 + t2 + t3
                beta = 3 * Tsum / (2 * (m + 3))
                rest = (t1 * t1 + t2 * t2 + t3 * t3) / 4 - Tsum * Tsum / (4 * (m + 3))
                G = G_list(beta)
                poly = pm(pm(pm([0, 0, 1.0], [t1 * t1, 2 * t1, 1.0]), [t2 * t2, 2 * t2, 1.0]), [t3 * t3, 2 * t3, 1.0])
                val = sum(poly[q] * G[q] for q in range(9))
                vand = p1 * p2 * p3 * (p1 + p2) * (p2 + p3) * (p1 + p2 + p3)
                tot += val * vand * math.exp(-rest)
    return math.sqrt(3 / 7) * 24 / mehta_Z_float(4) * tot * h ** 3


### MAIN
def main():
    global MUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", choices=MUTANTS)
    args = ap.parse_args()
    MUT = args.mutant
    t0 = time.time()
    checks = {}

    D2 = enclose_integral(d2_terms(), T=Fr(40))
    D3 = enclose_integral(d3_terms())
    B4 = m4_setup()
    D4 = enclose_m4(B4)
    if MUT == "remainder-dropped":
        D4 = enclose_m4(B4, K=6, cells=[(Fr(i), Fr(i + 1)) for i in range(60)], remainder=False)
    D4_coarse = enclose_m4(B4, K=K_ORDER // 2, cells=[(Fr(i), Fr(i + 1)) for i in range(60)],
                           remainder=MUT != "remainder-dropped")
    Z4 = enclose_m4(z4_setup())
    D2_exact = Iv.frac(Fr(29, 6)) - SQRT6
    D3_pin = Iv(Decimal(D3_PIN[0]), Decimal(D3_PIN[1]))

    def c_ref(d, D):
        return GAMMA76 * CBRT32 * sphere_area(d) * D / (SQRT3 * SQRTPI * ipow(2 * PI, d))
    c2, c3, c4, c5 = c_ref(2, Iv.frac(Fr(4, 3))), c_ref(3, D2), c_ref(4, D3), c_ref(5, D4)
    c5_closed = GAMMA76 * CBRT32 * D4 / (12 * SQRT3 * PI * PI * PI * SQRTPI)

    # torus transfer d = 5, L = 24: shells (2j+1)^5 - (2j-1)^5 <= 242 j^4, |n|^6 <= 125 j^6, ratio < 1/2
    shells = 30250 if MUT != "image-shells" else 15125
    E5 = Iv.frac(2 * shells * (76 * 24 ** 6 + 15)) * (-Iv.frac(288)).exp()
    n_jet = 5 + 1 + 15
    eps = Iv.frac(6 * n_jet) * E5
    b_exp = 10 if MUT != "sandwich-exponent" else 8                            # d + n_A/2, n_A = 10; a = 29/3 <= 10
    num = Iv(ZERO)
    for k in range(1, 11):
        num = num + Iv.frac(math.comb(10, k) - (-1) ** k * math.comb(b_exp, k)) * ipow(eps, k)
    delta = num / ipow(1 - eps, b_exp)
    ratio = Iv((1 / (1 + delta)).lo, (1 + delta).hi)
    c5_torus = c5 * ratio

    # ---- rules
    f2, f3 = float_orthant(2, 16.0, 400), float_orthant(3, 14.0, 90)
    f4, f4a = float_orthant(4, 14.0, 28), float_D4_aint()
    def near(iv, fv, tol):
        f, t = Decimal(repr(fv)), Decimal(repr(tol))
        return CE.subtract(iv.lo, f) <= t and CE.subtract(f, iv.hi) <= t
    checks["FLOAT_INSIDE"] = bool(near(D2, f2, 2e-3) and near(D3, f3, 2e-2) and near(D4, f4, 1.5) and near(D4, f4a, 0.4))
    checks["D2_EXACT"] = bool(D2.intersects(D2_exact))
    checks["D3_CONSISTENT"] = bool(D3.intersects(D3_pin) and D3.width() < Decimal("1e-40"))
    checks["MEHTA_Z4"] = bool(Z4.intersects(mehta_Z(4)) and Z4.intersects(1536 * PI) and Z4.width() < Decimal("1e-30"))
    checks["WIDTHS"] = bool(D2.width() < Decimal("1e-40") and D4.width() < Decimal("1e-28") and c5.width() < Decimal("1e-28"))
    side = {d: Iv(Decimal(a), Decimal(b)) for d, (a, b) in SIDE24.items()}
    checks["SIDE24_CONSISTENT"] = bool(all(c.intersects(side[d] + Iv(-side[d].hi * Decimal("1e-106"), side[d].hi * Decimal("1e-106")))
                                           for d, c in ((2, c2), (3, c3))))
    checks["CLOSED_FORM_D5"] = bool(c5.intersects(c5_closed))
    checks["TRUNCATION_NESTING"] = bool(D4_coarse.intersects(D4) and D4_coarse.width() >= D4.width())
    checks["MONOTONE"] = bool(Iv.frac(Fr(4, 3)).hi < D2.lo < D2.hi < D3.lo < D3.hi < D4.lo
                              and c2.lo > c3.hi > c3.lo > c4.hi > c4.lo > c5.hi and c5.lo > 0)
    checks["IMAGE_BOUND"] = bool(Decimal("7e-111") < E5.lo and E5.hi < Decimal("8e-111"))
    checks["TRANSFER_BOUND"] = bool(delta.hi >= CF.multiply(Decimal(20), eps.lo) and delta.lo <= CC.multiply(Decimal(21), eps.hi)
                                    and delta.lo > CF.multiply(Decimal(19), eps.lo) and c5_torus.contains(c5))
    checks["LIBRARY_EXACT"] = bool(library_exact())
    pinned = {"D_4": D4, "c_5_ref": c5, "pi": PI}
    checks["PINNED"] = bool(all(str(pinned[k].lo).startswith(p) and str(pinned[k].hi).startswith(p) for k, p in PINNED.items()))
    if len(checks) != 13:
        raise ValueError("rule count changed")
    passed = all(checks.values())

    out = {
        "object": "CL-SIDE24-D5-COEFFICIENT-20260930-v1",
        "scientific_effect": "NONE",
        "certified": True,
        "mutant": MUT,
        "passed": passed,
        "checks": checks,
        "cone_moments": {"D_2": D2.pair(), "D_2_exact_29/6-sqrt6": D2_exact.pair(), "D_3": D3.pair(), "D_3_Math-199": list(D3_PIN),
                         "D_4": D4.pair(), "D_4_coarse": D4_coarse.pair()},
        "coefficients": {"c_2_ref": c2.pair(), "c_3_ref": c3.pair(), "c_4_ref": c4.pair(), "c_5_ref": c5.pair(),
                         "c_5_ref_closed": c5_closed.pair(), "c_5_24": c5_torus.pair()},
        "transfer": {"E_5": E5.pair(), "epsilon": eps.pair(), "delta": delta.pair(), "n_jet": n_jet,
                     "exponents": {"a": "29/3 (<= 10 used)", "b": 10}},
        "constants": {"pi": PI.pair(), "Gamma(7/6)": GAMMA76.pair(), "(3/2)^(1/3)": CBRT32.pair(),
                      "Z_4": mehta_Z(4).pair(), "Z_4_quadrature": Z4.pair(), "1536pi": (1536 * PI).pair(),
                      "|S^4|": sphere_area(5).pair()},
        "quadrature": {"K": K_ORDER, "cells": "width 1/2 on [0, 30], width 1 on [30, 60]", "rho": str(RHO)},
        "float_reference_10sig": {"D_2_orthant": "%.10g" % f2, "D_3_orthant": "%.10g" % f3, "D_4_orthant": "%.10g" % f4,
                                  "D_4_a_integrated_3D": "%.10g" % f4a},
        "exact_relations": {
            "D_m": "sqrt(3/(m+3)) Z_m^-1 int_{mu<0} prod mu_i^2 |Delta(mu)| exp(-|mu|^2/4 + (sum mu)^2/(4(m+3))) dmu",
            "decoupling": "alpha = 3m/(4(m+3)), beta = 3T/(2(m+3)); erfc branch exp(-T^2/(4m(m-1)) - |u|^2/4), polynomial branch exp(-T^2/((m-1)(m+3)) - |u|^2/4), u = t - (T/(m-1)) 1",
            "m=4": "p1 = (T - 2 p2 - p3)/3, v = 2 p2 + p3, p2 = (v - p3)/2; |u|^2/4 = v^2/24 + p3^2/8; dp1 dp2 dp3 = dT dv dp3/6; region 0 < p3 < v < T",
            "erfc argument": "beta/(2 sqrt alpha) = T sqrt21/28; erfc(x) <= e^{-x^2} in the tail",
            "L_a": "L_a(T) = int_0^T e^{-v^2/24} sum_k e_{a,k}(v) J_k(v) dv, accumulated cell by cell",
            "D_4": "sqrt(3/7) . 24/Z_4 . (1/6) I_4",
            "Z_4": "4! sqrt pi (1/6) int dT e^{-T^2/48} int_0^T dv e^{-v^2/24} int_0^v V e^{-p3^2/8} dp3 = 1536 pi",
            "c_d_ref": "Gamma(7/6) (3/2)^(1/3) |S^{d-1}| D_{d-1} / (sqrt3 sqrt pi (2 pi)^d); d = 5: Gamma(7/6) (3/2)^(1/3) D_4 / (12 sqrt3 pi^(7/2))",
            "transfer": "E_5 = 60500 (76 24^6 + 15) e^-288; eps = 126 E_5; c_{5,24}/c_{5,ref} in [1/(1+delta), 1+delta], delta = (1+eps)^10/(1-eps)^10 - 1",
        },
    }
    sys.stdout.write(json.dumps(out, indent=1) + "\n")
    sys.stderr.write("seconds %.1f\n" % (time.time() - t0))
    sys.exit(0 if passed else 1)


