import math, random, sys, time
from math import erf, exp, sqrt, pi, gamma, factorial
sys.path.insert(0, '.')
from core import cov_point, c_even
def phi(x): return exp(-x*x/2)/sqrt(2*pi)
def Phi(x): return 0.5*(1+erf(x/sqrt(2)))
def phi2(x): return phi(x/sqrt(2))/sqrt(2)      # N(0,2) density
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
def c_m(m):   # [SC] (8) via Mehta's integral
    return pi**(m*(m-1)/4)*factorial(m)*gamma(1.5)**m/math.prod(gamma(1+j/2) for j in range(1, m+1))
def m2(b): mu, s = -b, sqrt(2); t = mu/s; return (mu*mu+s*s)*Phi(-t) - mu*s*phi(t)
# numerator N_d(b) = c_m (2pi)^{-m(m-1)/4} ∫_{0<h_2<..<h_m} prod h^3 phi2(h-b) Vandermonde dh   (m = d-1 hard eigenvalues: m-1 of them)
def N_d(d, b, n=60, L=14.0):
    m = d-1; k = m-1               # number of hard eigenvalues
    rule = gauss_legendre(n)
    pref = c_m(m)*(2*pi)**(-m*(m-1)/4)
    if k == 0: return pref
    def rec(prev, depth):
        # integrate h_{depth} from prev to L
        if depth == k: return 1.0
        def f(h):
            v = h**3*phi2(h-b)
            for p in prev: v *= (h-p)
            return v*rec(prev+[h], depth+1)
        return gl(f, prev[-1] if prev else 0.0, L, rule)
    return pref*rec([], 0)
# m_{d,b} = E[(det A)^2 1{A neg def}] for A = -b I + GOE_m (diag var 2, off var 1): ordered-eigenvalue quadrature and matrix MC
def m_d_quad(d, b, n=60, L=14.0):
    m = d-1; rule = gauss_legendre(n); pref = c_m(m)*(2*pi)**(-m*(m-1)/4)
    def rec(prev, depth):
        if depth == m: return 1.0
        def f(l):
            v = l*l*phi2(l+b)
            for p in prev: v *= (l-p)
            return v*rec(prev+[l], depth+1)
        return gl(f, prev[-1] if prev else -L, 0.0, rule)
    return pref*rec([], 0)
def det(M):
    n = len(M); A = [r[:] for r in M]; d = 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if A[p][c] == 0: return 0.0
        if p != c: A[c], A[p] = A[p], A[c]; d = -d
        d *= A[c][c]
        for r in range(c+1, n):
            f = A[r][c]/A[c][c]
            for j in range(c, n): A[r][j] -= f*A[c][j]
    return d
def n_negative(M):
    """number of negative eigenvalues of a symmetric matrix via signs of leading principal minors (generic case)."""
    n = len(M); signs = [1.0]
    for k in range(1, n+1): signs.append(det([row[:k] for row in M[:k]]))
    return sum(1 for i in range(n) if signs[i]*signs[i+1] < 0)
def m_d_mc(d, b, n=200000, seed=7):
    random.seed(seed); m = d-1; acc = acc2 = 0.0
    for _ in range(n):
        A = [[0.0]*m for _ in range(m)]
        for i in range(m):
            A[i][i] = -b+sqrt(2)*random.gauss(0, 1)
            for j in range(i+1, m): A[i][j] = A[j][i] = random.gauss(0, 1)
        v = det(A)**2 if n_negative(A) == m else 0.0
        acc += v; acc2 += v*v
    mu = acc/n; return mu, sqrt(max(acc2/n-mu*mu, 0)/n)
def R_d(d, b):
    return N_d(d, b)*m2(b)/(phi2(b)*m_d_quad(d, b))
if __name__ == '__main__':
    print('c_m:', [c_m(m) for m in (1, 2, 3, 4)])
    for d in (2, 3, 4, 5):
        for b in (0.0, 1.0):
            t = time.time(); N = N_d(d, b); M = m_d_quad(d, b); R = N*m2(b)/(phi2(b)*M)
            mc = m_d_mc(d, b, 100000 if d < 5 else 60000)
            print(f'd={d} b={b}: N_d={N:.10f} m_d(quad)={M:.10f} m_d(MC)={mc[0]:.5f}±{mc[1]:.5f} R_d={R:.10f}  [{time.time()-t:.0f}s]', flush=True)
    print('R_3(0) closed form (32+28 sqrt2)/17 =', (32+28*sqrt(2))/17)
    # convergence of the quadrature order for d=4,5
    for d in (4, 5):
        print(d, [N_d(d, 0.0, n) for n in (40, 60, 80)], [m_d_quad(d, 0.0, n) for n in (40, 60, 80)])
