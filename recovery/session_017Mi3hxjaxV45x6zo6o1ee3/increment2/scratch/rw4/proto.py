import sys, importlib.util, time, math
from fractions import Fraction as Fr
from decimal import Decimal
sys.argv = ["x"]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
SP = "<SCRATCH>/"
W = load("w3", SP + "wtw3/frontiers/c8_window_coefficient_d3_torus_20260930/window_coefficient_d3.py")
Iv, PI, SQRT2, SQRTPI, SQRT2PI, ZERO = W.Iv, W.PI, W.SQRT2, W.SQRTPI, W.SQRT2PI, W.ZERO
CC, CF = W.CC, W.CF

# ---- exact polynomial algebra (dict exponent tuple -> Fraction)
def padd(u, v):
    w = dict(u)
    for k, c in v.items():
        w[k] = w.get(k, Fr(0)) + c
    return {k: c for k, c in w.items() if c != 0}
def pmul(u, v):
    w = {}
    for k1, c1 in u.items():
        for k2, c2 in v.items():
            k = tuple(a + b for a, b in zip(k1, k2)); w[k] = w.get(k, Fr(0)) + c1 * c2
    return {k: c for k, c in w.items() if c != 0}
def pscale(u, c): return {k: v * c for k, v in u.items() if v * c != 0}
def pconst(c, n): return {(0,) * n: Fr(c)} if c != 0 else {}
def pvar(i, n):
    e = [0] * n; e[i] = 1; return {tuple(e): Fr(1)}
def plin(coeffs):   # sum coeffs[i] * var_i
    n = len(coeffs); out = {}
    for i, c in enumerate(coeffs):
        if c: out = padd(out, pscale(pvar(i, n), Fr(c)))
    return out

def Jk_closed(k):
    """J_k(s) = int_0^s q^k e^{-q^2/8} dq for odd k: (B_k, C_k) with J_k = B_k + C_k(s) e^{-s^2/8}, C_k a list by power."""
    assert k % 2 == 1
    B, C = Fr(4), [Fr(-4)]
    j = 1
    while j < k:
        # J_{j+2} = -4 s^{j+1} e^{-s^2/8} + 4 (j+1) J_j
        C2 = [Fr(0)] * (j + 2)
        for i, c in enumerate(C): C2[i] += 4 * (j + 1) * c
        C2[j + 1] += -4
        B, C, j = 4 * (j + 1) * B, C2, j + 2
    return B, C

def moments(gamma, nmax):
    """M_j(T) = int_0^T s^j e^{-gamma s^2} ds = alpha_j + rho_j(T) e^{-gamma T^2} + beta_j E(T), E(T) = (1/2) sqrt(pi/gamma) erf(sqrt(gamma) T)."""
    g = Fr(gamma)
    M = [(Fr(0), [], Fr(1)), (1 / (2 * g), [-1 / (2 * g)], Fr(0))]
    for j in range(nmax - 1):
        a, r, b = M[j]
        r2 = [x * (j + 1) / (2 * g) for x in r] + [Fr(0)] * (j + 2 - len(r))
        r2[j + 1] += -1 / (2 * g)
        M.append(((j + 1) * a / (2 * g), r2, (j + 1) * b / (2 * g)))
    return M

def lpadd(u, v):
    n = max(len(u), len(v)); return [(u[i] if i < len(u) else 0) + (v[i] if i < len(v) else 0) for i in range(n)]
def lpmul(u, v):
    if not u or not v: return []
    w = [Fr(0)] * (len(u) + len(v) - 1)
    for i, a in enumerate(u):
        for j, b in enumerate(v): w[i + j] += a * b
    return w

def build_terms(m):
    """g_m(T) = exp(-3T^2/(4m(m+3))) F_m(T) as a list of (kind, poly in T, gamma) with kind in {'1', 'E24', 'E6'} etc."""
    if m == 3:
        T, s, q = pvar(0, 3), pvar(1, 3), pvar(2, 3)
        a = plin([Fr(1, 3), Fr(-1, 3), 0]); p = plin([0, Fr(1, 2), Fr(-1, 2)])
        ap, apq = padd(a, p), padd(padd(a, p), q)
        P = pmul(pmul(pmul(a, a), pmul(ap, ap)), pmul(apq, apq))
        P = pmul(P, pmul(pmul(p, q), padd(p, q)))
        # group by q power
        byk = {}
        for (i, j, k), c in P.items():
            byk.setdefault(k, {})[(i, j)] = byk.setdefault(k, {}).get((i, j), Fr(0)) + c
        assert all(k % 2 == 1 for k in byk), "even q power"
        U, V = {}, {}       # polys in (T, s)
        for k, ck in byk.items():
            Bk, Ck = Jk_closed(k)
            U = padd(U, pscale(ck, Bk))
            Ckp = {(0, n): c for n, c in enumerate(Ck) if c}
            V = padd(V, pmul(ck, Ckp))
        # s-integral 0..T: U with e^{-s^2/24}, V with e^{-s^2/24 - s^2/8} = e^{-s^2/6}
        out = {}
        def acc(key, poly):
            out[key] = lpadd(out.get(key, []), poly)
        for W_, gam in ((U, Fr(1, 24)), (V, Fr(1, 6))):
            jmax = max(j for (_, j) in W_)
            Ms = moments(gam, jmax)
            for (i, j), c in W_.items():
                a_, r_, b_ = Ms[j]
                Ti = [Fr(0)] * i + [c]
                acc(("1",), lpmul(Ti, [a_]))
                acc(("e", gam), lpmul(Ti, r_))
                acc(("E", gam), lpmul(Ti, [b_]))
        return out, Fr(1, 24)
    if m == 2:
        T, p = pvar(0, 2), pvar(1, 2)
        P = pscale(pmul(pmul(padd(pmul(T, T), pscale(pmul(p, p), -1)), padd(pmul(T, T), pscale(pmul(p, p), -1))), p), Fr(1, 16))
        out = {}
        for (i, k), c in P.items():
            Bk, Ck = Jk_closed(k)
            out[("1",)] = lpadd(out.get(("1",), []), lpmul([Fr(0)] * i + [c], [Bk]))
            out[("e", Fr(1, 8))] = lpadd(out.get(("e", Fr(1, 8)), []), lpmul([Fr(0)] * i + [c], Ck))
        return out, Fr(3, 40)

for m in (2, 3):
    t0 = time.time()
    out, gout = build_terms(m)
    for k, v in out.items():
        while v and v[-1] == 0: v.pop()
        print(m, k, len(v), [str(x) for x in v][:4], "...")
    # float check of g at some T vs direct 2D/1D integration
    def F_closed(Tf):
        tot = 0.0
        for key, poly in out.items():
            pv = sum(float(c) * Tf ** i for i, c in enumerate(poly))
            if key[0] == "1": tot += pv
            elif key[0] == "e": tot += pv * math.exp(-float(key[1]) * Tf * Tf)
            else:
                g = float(key[1]); tot += pv * 0.5 * math.sqrt(math.pi / g) * math.erf(math.sqrt(g) * Tf)
        return tot
    def F_direct(Tf, n=400):
        if m == 2:
            h = Tf / n; s = 0.0
            for i in range(n + 1):
                p = i * h; w = 1 if i in (0, n) else (4 if i % 2 else 2)
                s += w * (Tf * Tf - p * p) ** 2 * p / 16 * math.exp(-p * p / 8)
            return s * h / 3
        hs = Tf / 200; tot = 0.0
        for i in range(201):
            sv = i * hs; ws = 1 if i in (0, 200) else (4 if i % 2 else 2)
            inner = 0.0; hq = sv / 200 if sv > 0 else 0
            for j in range(201):
                qv = j * hq; wq = 1 if j in (0, 200) else (4 if j % 2 else 2)
                a = (Tf - sv) / 3; p = (sv - qv) / 2
                inner += wq * a*a*(a+p)**2*(a+p+qv)**2*p*qv*(p+qv)*math.exp(-sv*sv/24 - qv*qv/8)
            tot += ws * inner * hq / 3
        return tot * hs / 3
    for Tf in (0.7, 2.0, 5.0):
        print(m, Tf, F_closed(Tf), F_direct(Tf))
    print("sec", time.time() - t0)

# ================= quadrature engine
K_ORDER, RHO, CELL, T_CUT = 40, Fr(6), Fr(1, 2), Fr(60)
def ipow(x, n):
    r = Iv(W.ONE)
    for _ in range(n): r = r * x
    return r

def gauss_coeffs(gamma, s0, K):
    g = Iv.of(gamma); S0 = Iv.frac(s0)
    y = [(-(g * S0.sq())).exp()]; prev = Iv(ZERO)
    for n in range(K):
        nxt = -(2 * g) * (S0 * y[n] + prev) / (n + 1); prev = y[n]; y.append(nxt)
    return y
def gauss_bound(gamma, s0, rho):
    g = Iv.of(gamma); inner = max(Fr(0), abs(Fr(s0)) - Fr(rho))
    return (g * Iv.frac(rho * rho) - g * Iv.frac(inner * inner)).exp()
def erf_coeffs(kappa, s0, K):
    k = kappa; S0 = Iv.frac(s0)
    c0 = W.erf_iv(k * S0)
    y = gauss_coeffs(k.sq(), s0, K)
    pref = 2 * k / SQRTPI
    return [c0] + [pref * y[n] / (n + 1) for n in range(K)]
def erf_bound(kappa, s0, rho):
    return 2 * kappa / SQRTPI * Iv.frac(abs(Fr(s0)) + Fr(rho)) * (kappa.sq() * Iv.frac(Fr(rho) ** 2)).exp()
def poly_coeffs(c, s0, K):
    out = []
    for n in range(K + 1):
        v = Fr(0)
        for j, cj in enumerate(c):
            if j >= n: v += cj * math.comb(j, n) * Fr(s0) ** (j - n)
        out.append(Iv.frac(v))
    return out
def poly_bound(c, s0, rho):
    r = abs(Fr(s0)) + Fr(rho); return Iv.frac(sum(abs(cj) * r ** j for j, cj in enumerate(c)))
def t_mul(u, v):
    K = len(u) - 1; out = []
    for n in range(K + 1):
        acc = Iv(ZERO)
        for i in range(n + 1): acc = acc + u[i] * v[n - i]
        out.append(acc)
    return out

def phi_aff_coeffs(alpha, beta, s0, K):
    """Phi(alpha - beta T) at T = s0 + x: c_0 = Phi(w0), c_{n+1} = -beta/sqrt(2 pi) y_n/(n+1), y the coefficients of
    exp(-(w0 - beta x)^2/2): (n+1) y_{n+1} = beta w0 y_n - beta^2 y_{n-1}."""
    w0 = alpha - beta * Iv.frac(s0)
    c0 = W.Phi(w0)
    y = [(-(w0.sq()) / 2).exp()]; prev = Iv(ZERO); b2 = beta.sq()
    for n in range(K):
        nxt = (beta * w0 * y[n] - b2 * prev) / (n + 1); prev = y[n]; y.append(nxt)
    pref = beta / SQRT2PI
    return [c0] + [-(pref * y[n]) / (n + 1) for n in range(K)]
def phi_aff_bound(beta, rho):
    br = beta * Iv.frac(rho)
    return 1 + br * (br.sq() / 2).exp() / SQRT2PI

class Model:
    """g(T) = sum over terms of const * poly(T) * exp(-gam T^2) * [erf(kap T)], cached Taylor data per cell."""
    def __init__(self, terms, K=K_ORDER, cell=CELL, T=T_CUT, rho=RHO):
        self.terms, self.K, self.cell, self.T, self.rho = terms, K, cell, T, rho
        half = cell / 2; self.half = half
        ncell = int(T / cell); assert ncell * cell == T
        mom = [Iv.frac(2 * half ** (n + 1) / (n + 1)) if n % 2 == 0 else Iv(ZERO) for n in range(K + 1)]
        rem_moment = Iv.frac(2 * half ** (K + 2) / (K + 2) / (1 - Fr(half) / Fr(rho)))
        rho_pow = Iv.frac(Fr(rho) ** (K + 1))
        self.cells = []
        Rsum = Iv(ZERO)
        for c in range(ncell):
            s0 = (2 * c + 1) * half
            G = [Iv(ZERO)] * (K + 1); M = Iv(ZERO)
            for const, poly, gam, kap in terms:
                pc = poly_coeffs(poly, s0, K); Mb = poly_bound(poly, s0, rho)
                pc = t_mul(pc, gauss_coeffs(gam, s0, K)); Mb = Mb * gauss_bound(gam, s0, rho)
                if kap is not None:
                    pc = t_mul(pc, erf_coeffs(kap, s0, K)); Mb = Mb * erf_bound(kap, s0, rho)
                G = [x + const * y for x, y in zip(G, pc)]
                M = M + Iv(const.mag()) * Mb
            Gm = []
            for j in range(K + 1):
                acc = Iv(ZERO)
                for i in range(K + 1 - j):
                    if (i + j) % 2 == 0: acc = acc + G[i] * mom[i + j]
                Gm.append(acc)
            self.cells.append((s0, Gm))
            Rsum = Rsum + M / rho_pow * rem_moment
        self.Rsum = Rsum.hi
        # tail beyond T: int_T^inf T^j e^{-gam T^2} <= (1/2) gam^{-(j+1)/2} Gamma((j+1)/2, gam T^2)
        tail = Iv(ZERO)
        for const, poly, gam, kap in terms:
            g = Iv.of(gam); x = g * Iv.frac(T * T)
            cmag = Iv(const.mag()) * (1 if kap is None else 1)    # |erf| <= 1
            for j, cj in enumerate(poly):
                if cj == 0: continue
                a = Fr(j + 1, 2)
                if a.denominator == 2:
                    xa1 = ipow(x, int(a - Fr(1, 2))) * x.sqrt() / x; gpow = ipow(g, int(a)) * g.sqrt()
                else:
                    xa1 = ipow(x, int(a) - 1); gpow = ipow(g, int(a))
                up = xa1 * (-x).exp()
                if a > 1: up = up / (1 - Iv.frac(a - 1) / x)
                tail = tail + cmag * Iv.frac(abs(cj)) * up / gpow / 2
        self.tail = tail.hi

    def integral(self, alpha=None, beta=None):
        """int_0^inf g(T) Phi(alpha - beta T) dT (alpha, beta intervals; alpha None means the factor 1)."""
        K = self.K; tot = Iv(ZERO)
        if alpha is None:
            for s0, Gm in self.cells: tot = tot + Gm[0]
            R = self.Rsum
        else:
            MW = phi_aff_bound(beta, self.rho)
            for s0, Gm in self.cells:
                Wc = phi_aff_coeffs(alpha, beta, s0, K)
                for j in range(K + 1): tot = tot + Wc[j] * Gm[j]
            R = (Iv(self.Rsum) * MW).hi
        return tot + Iv(-R - self.tail, R + self.tail)

def model_for(m, gamma_out=None):
    out, gref = build_terms(m)
    gout = gref if gamma_out is None else gamma_out
    terms = []
    for key, poly in out.items():
        poly = list(poly)
        while poly and poly[-1] == 0: poly.pop()
        if not poly: continue
        if key[0] == "1":
            terms.append((Iv(W.ONE), poly, Iv.frac(gout), None))
        elif key[0] == "e":
            terms.append((Iv(W.ONE), poly, Iv.frac(gout + key[1]), None))
        else:
            g = key[1]
            terms.append(((PI / Iv.frac(g)).sqrt() / 2, poly, Iv.frac(gout), Iv.frac(g).sqrt()))
    return terms

if __name__ == "__main__":
    t0 = time.time()
    M3 = Model(model_for(3)); print("model3", time.time() - t0)
    lam_inf = M3.integral()
    D3 = lam_inf / (96 * PI)
    atan2 = PI / 2 - W.atan_iv(Iv.frac(Fr(1, 2)))
    D3x = (50 * PI + 200 * atan2 - 228) / (9 * PI)
    print("D3 quad", D3.pair(), "\nD3 exact", D3x.pair(), D3.intersects(D3x), D3.width())
    t1 = time.time()
    s3 = W.SQRT3
    for b in (Fr(0), Fr(1), Fr(-1), Fr(2)):
        v = M3.integral(s3 * Iv.frac(b), 1 / (2 * s3))
        print(b, (v / (96 * PI)).pair())
    print("per-b", (time.time() - t1) / 4)
    t0 = time.time()
    M2 = Model(model_for(2)); print("model2", time.time() - t0)
    pref2 = Iv.frac(Fr(3, 5)).sqrt() / (8 * SQRT2PI)
    D2 = pref2 * M2.integral()
    print("D2", D2.pair(), D2.intersects(W.D2_EXACT))
    sw = Iv.frac(Fr(5, 2)).sqrt()
    def I2(lo, hi):
        def lam(b):
            if b == math.inf: return M2.integral()
            if b == -math.inf: return Iv(ZERO)
            return M2.integral(sw * Iv.frac(b), sw / 5)
        return pref2 * (lam(hi) - lam(lo))
    for B in ((Fr(0), Fr(1)), (Fr(-1), Fr(1)), (Fr(3), Fr(7, 2)), (Fr(0), math.inf), (-math.inf, Fr(0))):
        a, b_ = I2(*B), W.I3_ref(*B)
        print(B, a.pair(), b_.pair(), a.intersects(b_))
