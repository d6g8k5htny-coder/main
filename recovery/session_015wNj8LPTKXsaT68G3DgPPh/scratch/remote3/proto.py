"""Prototype: d = 3 remote contact kernel Lambda_j(x) of [RM] (13) for K = exp(-|x|^2/2) by Monte Carlo, with the
far-field control from the shifted-GOE eigenvalue quadrature."""
import math, random, time, itertools
from math import exp, sqrt, pi, erf, gamma, factorial

def hermite(n, t):
    h0, h1 = 1.0, t
    if n == 0: return 1.0
    for m in range(1, n): h0, h1 = h1, t*h1 - m*h0
    return h1
def kder(gam, y):   # d^gam K(y), K = exp(-|y|^2/2)
    v = 1.0
    for g, yi in zip(gam, y): v *= (-1)**g * hermite(g, yi) * exp(-yi*yi/2)
    return v
def cov(al, s, be, t):
    return (-1)**sum(al) * kder(tuple(a+b for a, b in zip(al, be)), [ti-si for si, ti in zip(s, t)])
D = 3
def U(i, n=1):
    v = [0]*D; v[i] = n; return tuple(v)
def add(*vs): return tuple(sum(v[i] for v in vs) for i in range(D))
Z = (0,0,0)
# site-0 jets: observations U_0 = (f, f_x, f_xx, f_xxx, f_y, f_z, f_xy, f_xz); targets A0 = (f_yy, f_yz, f_zz)
OBS0 = [Z, U(0), U(0,2), U(0,3), U(1), U(2), add(U(0),U(1)), add(U(0),U(2))]
V0 = lambda b, k: [b, 0.0, 0.0, 12*k, 0.0, 0.0, 0.0, 0.0]
A0 = [add(U(1),U(1)), add(U(1),U(2)), add(U(2),U(2))]
OBSX = [U(0), U(1), U(2), Z]                       # Y_x = (grad f(x), f(x))
HX = [add(U(i),U(j)) for i in range(3) for j in range(i,3)]   # f_xx f_xy f_xz f_yy f_yz f_zz

def chol(S):
    n = len(S); L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = S[i][j] - sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                if s <= 0: raise ValueError('not SPD %g' % s)
                L[i][i] = sqrt(s)
            else: L[i][j] = s/L[j][j]
    return L
def solve(L, b):
    n = len(L); y = [0.0]*n
    for i in range(n): y[i] = (b[i] - sum(L[i][k]*y[k] for k in range(i)))/L[i][i]
    x = [0.0]*n
    for i in reversed(range(n)): x[i] = (y[i] - sum(L[k][i]*x[k] for k in range(i+1, n)))/L[i][i]
    return x
def det3(m):
    return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
def nneg3(m):
    d1 = m[0][0]; d2 = m[0][0]*m[1][1]-m[0][1]*m[1][0]; d3 = det3(m)
    seq = [1.0, d1, d2, d3]
    return sum(1 for i in range(3) if seq[i]*seq[i+1] < 0)

def conditioning(x, b, k):
    """mean/cov of T = (A0, H_x) given U_0 = v_0, Y_x = (0, b); and p_(Y_x|U_0)(0, b)."""
    sites_obs = [(a, Z) for a in OBS0] + [(a, x) for a in OBSX]
    vals = V0(b, k) + [0.0, 0.0, 0.0, b]
    tg = [(a, Z) for a in A0] + [(a, x) for a in HX]
    Soo = [[cov(a, s, c, t) for (c, t) in sites_obs] for (a, s) in sites_obs]
    Sto = [[cov(a, s, c, t) for (c, t) in sites_obs] for (a, s) in tg]
    Stt = [[cov(a, s, c, t) for (c, t) in tg] for (a, s) in tg]
    L = chol(Soo)
    W = [solve(L, row) for row in Sto]
    mu = [sum(W[i][j]*vals[j] for j in range(len(vals))) for i in range(len(tg))]
    cc = [[Stt[i][j] - sum(W[i][l]*Sto[j][l] for l in range(len(vals))) for j in range(len(tg))] for i in range(len(tg))]
    # density of Y_x given U_0 at (0,0,0,b): condition the 4 Y_x obs on the 8 U_0 obs
    n0 = len(OBS0)
    S00 = [r[:n0] for r in Soo[:n0]]; Sy0 = [r[:n0] for r in Soo[n0:]]; Syy = [r[n0:] for r in Soo[n0:]]
    L0 = chol(S00); Wy = [solve(L0, row) for row in Sy0]
    muy = [sum(Wy[i][j]*vals[j] for j in range(n0)) for i in range(4)]
    cy = [[Syy[i][j] - sum(Wy[i][l]*Sy0[j][l] for l in range(n0)) for j in range(4)] for i in range(4)]
    Ly = chol(cy); dy = [vals[n0+i]-muy[i] for i in range(4)]
    q = solve(Ly, dy); q2 = sum(v*v for v in q)     # (dy)^T cy^-1 dy via L L^T: solve gives cy^-1 dy; need dy.cy^-1 dy
    q2 = sum(dy[i]*q[i] for i in range(4))
    logdet = 2*sum(math.log(Ly[i][i]) for i in range(4))
    p = exp(-0.5*q2 - 0.5*logdet - 2*math.log(2*pi))
    return mu, cc, p

def lam_mc(x, b, k, zs):
    mu, cc, p = conditioning(x, b, k)
    L = chol(cc)
    acc = [0.0]*4; acc2 = [0.0]*4
    for z in zs:
        t = [mu[i] + sum(L[i][j]*z[j] for j in range(i+1)) for i in range(9)]
        a0 = [[t[0], t[1]], [t[1], t[2]]]
        dA = t[0]*t[2]-t[1]*t[1]
        w0 = 36*k*k*dA*dA if (t[0] < 0 and dA > 0) else 0.0
        if w0 == 0.0: continue
        H = [[t[3], t[4], t[5]], [t[4], t[6], t[7]], [t[5], t[7], t[8]]]
        j = nneg3(H); v = w0*abs(det3(H))
        acc[j] += v
    n = len(zs)
    return [p*a/n for a in acc], p, mu

# far field: Lambda_inf_j = phi(b) (2pi)^(-3/2) E[|det(-bI + G3)| 1{index j}] ; z0 cancels
def phi(t): return exp(-t*t/2)/sqrt(2*pi)
def phi2(t): return phi(t/sqrt(2))/sqrt(2)
def gauss_legendre(n):
    xs, ws = [], []
    for i in range(1, n+1):
        x = math.cos(math.pi*(i-0.25)/(n+0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for m in range(2, n+1): p0, p1 = p1, ((2*m-1)*x*p1-(m-1)*p0)/m
            dp = n*(x*p1-p0)/(x*x-1.0); dx = p1/dp; x -= dx
            if abs(dx) < 1e-15: break
        xs.append(x); ws.append(2/((1-x*x)*dp*dp))
    return xs, ws
def gl(f, lo, hi, rule):
    mid, half = 0.5*(lo+hi), 0.5*(hi-lo)
    return half*sum(w*f(mid+half*x) for x, w in zip(*rule))
def c_m(m): return pi**(m*(m-1)/4)*factorial(m)*gamma(1.5)**m/math.prod(gamma(1+j/2) for j in range(1, m+1))
def goe_index_absdet(m, b, n=40, L=14.0):
    """E[|det(-bI+G_m)| 1{index j}] for j = 0..m, by ordered eigenvalue density of A = -bI + G (eigenvalues l, density
    c_m (2pi)^{-m(m-1)/4} prod phi2(l_i + b) Vandermonde)."""
    rule = gauss_legendre(n); pref = c_m(m)*(2*pi)**(-m*(m-1)/4)
    out = [0.0]*(m+1)
    def rec(prev, depth, acc_det, nneg):
        if depth == m:
            out[nneg] += acc_det
            return
        # integrate next ordered eigenvalue over (-L, 0) and (0, L) separately to count negatives
        for (lo, hi, neg) in (((prev[-1] if prev else -L-abs(b)), 0.0, 1), (max(0.0, prev[-1] if prev else 0.0), L+abs(b), 0)):
            if hi <= lo: continue
            def f(l):
                v = phi2(l+b)*abs(l)
                for pp in prev: v *= (l-pp)
                return v
            # nested: need integral over l of f(l) * (sub-integral); implement via recursion with weights
            xs, ws = rule; mid, half = 0.5*(lo+hi), 0.5*(hi-lo)
            for xg, wg in zip(xs, ws):
                l = mid+half*xg
                # careful: ordering constraint lo = prev[-1]; for the positive branch lo = max(0, prev)
                rec(prev+[l], depth+1, acc_det*f(l)*half*wg, nneg+neg)
    rec([], 0, pref, 0)
    return out

if __name__ == '__main__':
    random.seed(3)
    for b in (0.0, 1.0):
        t0 = time.time()
        E = goe_index_absdet(3, b, n=30)
        lam_inf = [phi(b)*(2*pi)**(-1.5)*e for e in E]
        print(f'b={b}: E|det|1{{j}} = {E}, sum {sum(E):.8f}; Lambda_inf_j = {lam_inf}, sum {sum(lam_inf):.8f}  [{time.time()-t0:.1f}s]')
        E2 = goe_index_absdet(3, b, n=40); print('   order 40 sum', sum(E2), [e for e in E2])
    # unconditional 3x3 GOE(2,1): E|det| check by MC
    rng = random.Random(1); acc = 0.0; N = 200000
    for _ in range(N):
        a = [sqrt(2)*rng.gauss(0,1) for _ in range(3)]; o = [rng.gauss(0,1) for _ in range(3)]
        H = [[a[0], o[0], o[1]], [o[0], a[1], o[2]], [o[1], o[2], a[2]]]
        acc += abs(det3(H))
    print('MC E|det G3| =', acc/N, ' quadrature (b=0) =', sum(goe_index_absdet(3, 0.0, 30)))
    # MC Lambda at a far point vs Lambda_inf
    k = 1.0
    zs = [[random.gauss(0,1) for _ in range(9)] for _ in range(20000)]
    t0 = time.time()
    for b in (0.0, 1.0):
        lam, p, mu = lam_mc((8.0, 0.0, 0.0), b, k, zs)
        z0 = 36*k*k  # placeholder, need m_3,b
        print(f'far point b={b}: p={p:.6e} (phi(b)(2pi)^-1.5 = {phi(b)*(2*pi)**-1.5:.6e}); raw E[w0 F_j] = {lam}')
    print('time', time.time()-t0)
