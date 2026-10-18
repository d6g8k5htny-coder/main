"""C_alg(2,L) from ORIGINAL (1)/(12) by angular quadrature at the ORIGINAL moments,
compared with a*c_{2,inf} (ORIGINAL (32) times the exact a-scaling at kappa=0),
plus first-order sensitivities to kappa4, kappa6 and a. Analysis only."""
import sys, importlib.util, os
import mpmath as mp

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l24m", os.path.join(here, "l24_moments.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def calg_from(a, k4, k6):
    def integrand(th):
        e1, e2 = mp.cos(th), mp.sin(th)
        v1, v2 = -e2, e1
        B11 = a*a*(1+2*e1*e1) + k4*e1*e1
        B22 = a*a*(1+2*e2*e2) + k4*e2*e2
        B12 = a*a*2*e1*e2
        detB = B11*B22 - B12*B12
        D1 = e1*(a*a + k4*v1*v1)
        D2 = e2*(a*a + k4*v2*v2)
        q = (B22*D1*D1 - 2*B12*D1*D2 + B11*D2*D2)/detB
        Gam = 3*a*a + k4*(v1**4 + v2**4) - q
        s4 = e1**4 + e2**4; s6 = e1**6 + e2**6
        V = 6*a**3 + 9*a*k4*s4 + (k6 - k4*k4/a)*s6
        return (Gam/2)*V**(mp.mpf(2)/3)/mp.sqrt(detB)
    # integrand has period pi/2 (cubic symmetry); integrate over [0,pi/2] and multiply by 4
    I = 4*mp.quad(integrand, [0, mp.pi/4, mp.pi/2])
    pref = mp.gamma(mp.mpf(7)/6)/(mp.cbrt(24)*mp.sqrt(mp.pi)*(2*mp.pi)**2*a)
    return pref*I

for dps, L in [(170, 24), (60, 12), (60, 6), (60, 3), (60, 2)]:
    mp.mp.dps = dps
    L = mp.mpf(L)
    z, a, b, c, _ = m.moments(L)
    k4 = b - 3*a*a; k6 = c - 15*a*b + 30*a**3
    C = calg_from(a, k4, k6)
    c2 = m.c2inf()
    print("L=%s dps=%d" % (mp.nstr(L, 4), dps))
    print("   C_alg(2,L)          =", mp.nstr(C, 40))
    print("   c_2inf              =", mp.nstr(c2, 40))
    print("   C/(a c_2inf) - 1    =", mp.nstr(C/(a*c2) - 1, 8))
    print("   C/c_2inf - 1        =", mp.nstr(C/c2 - 1, 8))
    # sensitivities at this point (finite differences, relative step tiny)
    hstep = mp.mpf(10)**(-dps//3)
    dCk4 = (calg_from(a, k4+hstep, k6) - calg_from(a, k4-hstep, k6))/(2*hstep)
    dCk6 = (calg_from(a, k4, k6+hstep) - calg_from(a, k4, k6-hstep))/(2*hstep)
    dCa = (calg_from(a+hstep, k4, k6) - calg_from(a-hstep, k4, k6))/(2*hstep)
    print("   dlnC/dk4, dlnC/dk6, dlnC/dlna =", mp.nstr(dCk4/C, 8), mp.nstr(dCk6/C, 8), mp.nstr(dCa*a/C, 8))
    print("   first-order predicted C/(a c2)-1 =", mp.nstr((dCk4*k4 + dCk6*k6)/C, 8))
    V = 576 if L == 24 else None
    if L == 24:
        print("   V*C (bars per field per unit w) =", mp.nstr(L**2*C, 12))
