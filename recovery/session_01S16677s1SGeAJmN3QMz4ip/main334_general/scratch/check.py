"""Numerical check of the general amplitude-mixture law with a toy nu_H.

Pure Python (no numpy available).  All integrals in log-space trapezoid.
"""
import math

C_H = 1.0

def nu_H(u):
    # near part ~ C_H u^(-1/3), Gaussian decay at infinity; far part bounded, -> 0 at 0
    return C_H * u ** (-1.0 / 3.0) * math.exp(-u * u) + 0.5 * u * math.exp(-u * u)

def log_grid(lo, hi, n):
    a, b = math.log(lo), math.log(hi)
    h = (b - a) / n
    return [math.exp(a + i * h) for i in range(n + 1)], h

def m_H(u, n=4000):
    # int_0^u nu_H(s) ds, using s = w^{3/2} for the singular piece
    # near piece: int_0^u C_H s^{-1/3} e^{-s^2} ds = (3/2) int_0^{u^{2/3}} C_H e^{-w^3} dw
    W = u ** (2.0 / 3.0)
    h = W / n
    s = 0.0
    for i in range(n + 1):
        w = i * h
        wt = 0.5 if i in (0, n) else 1.0
        s += wt * math.exp(-w ** 3)
    near = 1.5 * C_H * h * s
    # far piece: int_0^u 0.5 s e^{-s^2} ds = 0.25 (1 - e^{-u^2})
    far = 0.25 * (1.0 - math.exp(-u * u))
    return near + far

# ---- amplitude laws: density p_A(a) on (0, inf), and m = E A^{-2/3} ----
def amp_power(alpha):
    a_max = math.sqrt((alpha + 2.0) / alpha)
    def p(a):
        return alpha * a_max ** (-alpha) * a ** (alpha - 1.0) if 0 < a < a_max else 0.0
    m = a_max ** (-2.0 / 3.0) * alpha / (alpha - 2.0 / 3.0) if alpha > 2.0 / 3.0 else float('inf')
    return p, m, "AMP power law alpha=%g (bounded by a=%.4f)" % (alpha, a_max)

def amp_lognormal(sigma):
    def p(a):
        return math.exp(-(math.log(a)) ** 2 / (2 * sigma * sigma)) / (a * sigma * math.sqrt(2 * math.pi))
    m = math.exp((2.0 / 3.0) ** 2 * sigma * sigma / 2.0)
    return p, m, "lognormal sigma=%g (unbounded, finite m)" % sigma

def amp_unbounded_infinite_m():
    # density (1/2) a^{-1/3} on (0,1) ... gives E A^{-2/3} = (1/2) int_0^1 a^{-1} da = inf ; add a Pareto tail
    # p(a) = (1/3) a^{-1/3} on (0,1)  [mass 1/2]  +  (1/2) a^{-2} on (1,inf) [mass 1/2]
    def p(a):
        if a <= 0: return 0.0
        if a < 1: return (1.0 / 3.0) * a ** (-1.0 / 3.0)
        return 0.5 * a ** (-2.0)
    return p, float('inf'), "unbounded law, density a^(-1/3)/3 on (0,1) + a^(-2)/2 on (1,inf): m = inf"

def nu_F(ell, p, n=6000):
    # nu_F(ell) = E[A^{-1} nu_H(ell/A)] = int_0^inf nu_H(u) p_A(ell/u) du/u   (u = ell/A)
    us, h = log_grid(1e-14, 60.0, n)
    s = 0.0
    for i, u in enumerate(us):
        wt = 0.5 if i in (0, n) else 1.0
        s += wt * nu_H(u) * p(ell / u)
    return h * s

def m_F(t, p, n=3000):
    # m_F(t) = E[m_H(t/A)] = int m_H(t/a) p_A(a) da ; log grid in a
    As, h = log_grid(1e-12, 1e6, n)
    s = 0.0
    for i, a in enumerate(As):
        wt = 0.5 if i in (0, n) else 1.0
        pa = p(a)
        if pa > 0:
            s += wt * m_H(t / a, n=600) * pa * a
    return h * s

def sup_g():
    us, _ = log_grid(1e-12, 60.0, 4000)
    return max(u ** (1.0 / 3.0) * nu_H(u) for u in us)

# ---- envelope integral check, d=2:  H_SG(b,k) = (1+|b|+k)^4 exp(-(b^2+k^2)) ----
D = 2
def H_SG(b, k):
    return (1.0 + abs(b) + k) ** (2 * D) * math.exp(-(b * b + k * k))

def I_SG(nb=400, nk=400):
    bs = [-8.0 + 16.0 * i / nb for i in range(nb + 1)]
    hb = 16.0 / nb
    ks, hk = log_grid(1e-8, 8.0, nk)
    s = 0.0
    for i, b in enumerate(bs):
        wb = 0.5 if i in (0, nb) else 1.0
        for j, k in enumerate(ks):
            wk = 0.5 if j in (0, nk) else 1.0
            s += wb * wk * k ** (-2.0 / 3.0) * H_SG(b, k) * k
    return s * hb * hk

def I_F(p, nb=120, nk=120, na=160):
    # int k^{-2/3} E[A^{-2} H_SG(b/A, k/A)] db dk, computed WITHOUT the substitution (direct 3d quadrature)
    bs = [-40.0 + 80.0 * i / nb for i in range(nb + 1)]
    hb = 80.0 / nb
    ks, hk = log_grid(1e-8, 40.0, nk)
    As, ha = log_grid(1e-6, 1e3, na)
    s = 0.0
    for i, b in enumerate(bs):
        wb = 0.5 if i in (0, nb) else 1.0
        for j, k in enumerate(ks):
            wk = 0.5 if j in (0, nk) else 1.0
            inner = 0.0
            for l, a in enumerate(As):
                wa = 0.5 if l in (0, na) else 1.0
                pa = p(a)
                if pa > 0:
                    inner += wa * a ** (-2.0) * H_SG(b / a, k / a) * pa * a
            s += wb * wk * k ** (-2.0 / 3.0) * inner * ha * k
    return s * hb * hk

if __name__ == "__main__":
    print("toy nu_H: sup_u u^(1/3) nu_H(u) = %.4f ; C_H = %.4f ; total mass int nu_H = %.4f" % (sup_g(), C_H, m_H(50.0)))
    laws = [amp_power(1.0), amp_lognormal(1.5), amp_power(0.5), amp_unbounded_infinite_m(), amp_power(2.0/3.0)]
    ells = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    for p, m, name in laws:
        print("\n== %s ==   m = E A^(-2/3) = %s ; predicted limit m*C_H = %s" % (name, m, m * C_H))
        print("   ell        ell^(1/3) nu_F(ell)      m_F(ell)/ell^(2/3)   [target (3/2) m C_H = %s]" % (1.5 * m * C_H))
        for ell in ells:
            v = ell ** (1.0 / 3.0) * nu_F(ell, p)
            mf = m_F(ell, p) / ell ** (2.0 / 3.0)
            print("   %.0e    %14.6f    %14.6f" % (ell, v, mf))
    print("\n== envelope integral check (d=2): int k^(-2/3) H^F  vs  m * int k^(-2/3) H_SG ==")
    isg = I_SG()
    print("   I_SG = %.6f" % isg)
    for p, m, name in laws[:2]:
        iF = I_F(p)
        print("   %s: I_F (direct 3d quadrature) = %.6f ; m*I_SG = %.6f ; ratio = %.5f" % (name, iF, m * isg, iF / (m * isg)))
