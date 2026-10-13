"""Planar microscopic radius-tail constants for the parent kernel (d = 2): C_* of [R] (R13), the whole-cluster
split of [E] (E5-E6), the second-order coefficient C_2/C_0 and B_sign of [Q] (13), (19), with the near-mass
identity alpha_1 + 2 alpha_2 = K_0 int Q |Z|^-12 p_odd as a check against the merged Math-#168 values.
Standard library only; deterministic Gauss quadrature."""
import json, math, os, sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
SQ2PI = math.sqrt(2 * math.pi)
phi = lambda t: math.exp(-t * t / 2) / SQ2PI
Phi = lambda t: 0.5 * math.erfc(-t / math.sqrt(2))

# exact shape integrals (verified in rational arithmetic in the review scratch): I, U2, B, Uabs; [E]'s J and D
I_SHAPE = Fr(246528, 35); U2 = Fr(240192, 35); B_SH = Fr(-428544, 7); UABS = Fr(5898627, 880)
J_OUT = Fr(1083417, 280)
D_IN = lambda: 27066286003 / 223205220 - (79298560 / 4782969) * math.log(2)

def gauss_legendre(n):
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for m in range(2, n + 1):
                p0, p1 = p1, ((2 * m - 1) * x * p1 - (m - 1) * p0) / m
            dp = n * (x * p1 - p0) / (x * x - 1.0)
            dx = p1 / dp; x -= dx
            if abs(dx) < 1e-15: break
        p0, p1 = 1.0, x
        for m in range(2, n + 1):
            p0, p1 = p1, ((2 * m - 1) * x * p1 - (m - 1) * p0) / m
        dp = n * (x * p1 - p0) / (x * x - 1.0)
        xs.append(x); ws.append(2.0 / ((1 - x * x) * dp * dp))
    return xs, ws

def gl_integrate(f, lo, hi, rule):
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return half * sum(w * f(mid + half * x) for x, w in zip(*rule))

# contact structure in d = 2 (exact, Math-#168): A = f_zz | even pins ~ N(-b, 2); odd cubic jets ~ N(0, diag(2, 2, 6))
def m2b(b):
    m, s = -b, math.sqrt(2.0)
    t = m / s
    return (m * m + s * s) * Phi(-t) - m * s * phi(t)
def prefactor(k, b):
    """p_b(0)/z_0 = phi(b/sqrt2)/(sqrt2 * 36 k^2 m_(2,b)) = c_m h_0(A_0 = 0, .)/z_0 with the odd density factored out (c_1 = 1)."""
    return phi(b / math.sqrt(2)) / math.sqrt(2) / (36 * k * k * m2b(b))
def p_odd(a, beta, c):
    return math.exp(-a * a / 4 - beta * beta / 4 - c * c / 12) / ((2 * math.pi) ** 1.5 * math.sqrt(2 * 2 * 6))

def b0(u): return 3 - 12 * u * u
def Q(u, v): b = b0(u); return (4 * v * v - b * b) * (b - 4 * u * v)

def cusp_integrals(k, rule):
    """J_cusp = int gamma^11 G0 da;  C2-type integrals: int gamma^9 (1+12A^2) G0 da, int gamma^13 D_a G0 da;  int |A| gamma^10 G0 da."""
    def G0(a): return p_odd(a, a * a / (12 * k), a ** 3 / (144 * k * k))
    def gam(a): return math.sqrt(1 + (a / (12 * k)) ** 2)
    def DaG0(a):
        # D_a G0 = -(a^2/(12 vbeta) + a^4/(576 k^2 vc)) G0 with vbeta = 2, vc = 6
        return -(a * a / 24 + a ** 4 / (3456 * k * k)) * G0(a)
    L = 14.0
    Jc = gl_integrate(lambda a: gam(a) ** 11 * G0(a), -L, L, rule)
    T1 = gl_integrate(lambda a: gam(a) ** 9 * (1 + 12 * (a / (12 * k)) ** 2) * G0(a), -L, L, rule)
    T2 = gl_integrate(lambda a: gam(a) ** 13 * DaG0(a), -L, L, rule)
    Tb = gl_integrate(lambda a: abs(a / (12 * k)) * gam(a) ** 10 * G0(a), -L, L, rule)
    # control: int G0 da with the (beta, c) arguments at the cusp is NOT 1; instead check int p_odd over a with beta, c free = 1 is trivial; use gamma^0 sanity
    return Jc, T1, T2, Tb

def near_mass(k, rules):
    """K_0 int_S int_Z Q(u,v) |Z|^-12 p_odd(a, beta(u,v,Z,a), c(u,v,Z,a)) da dZ du dv with K_0 = 108 k^7 * prefactor.
    In y = 1/Z: |Z|^-12 dZ = y^10 dy. Returns the integral without the prefactor (multiply by prefactor(k,b) for alpha_1 + 2 alpha_2)."""
    ru, rv, ry, ra = rules
    tot = 0.0
    for (ulo, uhi, lower) in ((-1.5, -0.5, lambda u: (12 * u * u - 3) / 2), (-0.5, 0.5, lambda u: (3 - 12 * u * u) / 2)):
        def fu(u):
            b = b0(u)
            def fv(v):
                q = Q(u, v)
                c3 = 2 * k * (v - b * u)
                # y-range: Gaussian decay through beta (k b y^2, variance 4 in exp(-beta^2/4)) and c (c3 y^3, exp(-c^2/12))
                Y = 1.2 * max((40.0 / (k * abs(b) + 1e-300)) ** 0.5 if b != 0 else 0.0, (40.0 / (abs(c3) + 1e-300)) ** (1.0 / 3.0))
                Y = min(Y, 1e3)
                def fy(y):
                    beta_shift = k * b * y * y
                    c_shift = c3 * y ** 3
                    def fa(a):
                        return p_odd(a, a * a / (12 * k) + beta_shift, a ** 3 / (144 * k * k) + (a * b / 4) * y * y + c_shift)
                    return y ** 10 * gl_integrate(fa, -14.0, 14.0, ra)
                return q * gl_integrate(fy, -Y, Y, ry)
            return gl_integrate(fv, lower(u), 3 - 6 * u, rv)
        tot += gl_integrate(fu, ulo, uhi, ru)
    return 108 * k ** 7 * tot

if __name__ == '__main__':
    ra = gauss_legendre(48)
    out = {}
    for k in (0.5, 1.0, 2.0):
        Jc, T1, T2, Tb = cusp_integrals(k, ra)
        I = float(I_SHAPE); D = D_IN(); J = float(J_OUT)
        for b in (0.0, 1.0):
            pf = prefactor(k, b)
            K0 = 108 * k ** 7 * pf
            C0 = (216 / 11) * k ** 7 * pf * Jc            # per unit shape integral: C_* = C0 * I
            Cstar = C0 * I
            C2 = K0 * (float(U2) * T1 + (2 * float(B_SH) / 13) * T2)
            c = C2 / (C0 * I)                              # [Q]'s C_0 is the full coefficient C0*I
            Bsign = (K0 / (C0 * I)) * float(UABS) * Tb
            kappa = (2 / 13) * (11 / 13) ** 5.5
            key = 'k=%s,b=%s' % (k, b)
            out[key] = {'J_cusp': Jc, 'C_star': Cstar, 'C0': C0, 'G_max': C0 * (I - D), 'G2': C0 * J, 'B2': C0 * D, 'single': C0 * (I - D - J),
                        'C2': C2, 'C2_over_C0': c, 'TV_coeff_kappa_c': kappa * c, 'B_sign': Bsign, 'lower_bound_4587_856': 4587 / 856}
            print(key, 'C_* = %.6g' % Cstar, ' C2/C0 = %.6g (bound 5.3587)' % c, ' kappa c = %.5g' % (kappa * c), ' B_sign = %.5g' % Bsign, flush=True)
    import time
    t0 = time.time()
    for k in (1.0,):
        nm = near_mass(k, (gauss_legendre(24), gauss_legendre(24), gauss_legendre(64), gauss_legendre(32)))
        for b in (0.0, 1.0):
            print('near mass k=%s b=%s: K0 int = %.6f  vs #168 alpha1+2alpha2 = %s' % (k, b, prefactor(k, b) * nm, {0.0: 1.32906 + 2 * 0.0255091, 1.0: 0.380523 + 2 * 0.00730347}[b]), flush=True)
    print('near mass time', time.time() - t0)
