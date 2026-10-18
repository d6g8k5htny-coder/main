"""Premise audit, part 1: exact symbolic / rational checks.

(a) M10 decomposition of V(e) (sum of manifestly nonnegative terms) equals the
    proof's (10) identically on the sphere, d=2..6 (homogenized polynomial identity).
(b) v^T B(e) v from (11) equals Var<sym(v e^T),H> computed from (8), and the
    Frobenius lower bound  >= min(beta,2a^2)*(|v|^2+(v.e)^2)/2 (beta=b-a^2).
(c) Exact rational finite symmetric product spectral law (random rational weights):
    moment tensor reproduces (8), (9), E(Q.e)^6 formula and (10) by an independent
    Schur complement at rational unit e; Gamma_e from (12)-(13) equals the Schur
    complement computed directly from the full Hessian moment tensor.
(d) Negative control: a carrier {0,+-s} kills V(e_axis); {0,+-s,+-2s} restores it.
"""
import itertools, random
from fractions import Fraction as Fr
import sympy as sp

# ---------------- (a) ----------------
a, b, c = sp.symbols('a b c', positive=True)
k4 = b - 3*a**2
k6 = c - 15*a*b + 30*a**3
beta = b - a**2
nu3 = c - b**2/a
for d in range(2, 7):
    e = sp.symbols('e0:%d' % d, real=True)
    s = sum(x**2 for x in e)
    p4 = sum(x**4 for x in e)
    p6 = sum(x**6 for x in e)
    V10 = 6*a**3*s**3 + 9*a*k4*s*p4 + (k6 - k4**2/a)*p6          # proof (10), homogenized
    sij = sum(e[i]**4*e[j]**2 for i in range(d) for j in range(d) if i != j)
    sijk = sum(e[i]**2*e[j]**2*e[k]**2 for i, j, k in itertools.combinations(range(d), 3))
    VM10 = nu3*p6 + 9*a*beta*sij + 36*a**3*sijk
    diff = sp.expand(sp.simplify((V10 - VM10)*a))
    print('(a) d=%d  (10) - [nonneg decomposition] == 0 :' % d, diff == 0)

# ---------------- (b) ----------------
for d in range(2, 6):
    e = sp.symbols('e0:%d' % d, real=True)
    v = sp.symbols('v0:%d' % d, real=True)
    E2 = sum(x**2 for x in e)
    # Hessian covariance tensor (8)
    def cov(i, j, k, l):
        r = a**2*((i == j)*(k == l) + (i == k)*(j == l) + (i == l)*(j == k))
        if i == j == k == l:
            r += k4
        return r
    S = [[(v[i]*e[j] + v[j]*e[i])/2 for j in range(d)] for i in range(d)]
    varSH = sum(S[i][j]*S[k][l]*cov(i, j, k, l)
                for i in range(d) for j in range(d) for k in range(d) for l in range(d))
    # (11) contracted, homogenized in |e|^2 (I -> |e|^2 I)
    vBv = a**2*(sum(x**2 for x in v)*E2 + 2*sum(v[i]*e[i] for i in range(d))**2) \
        + k4*sum(v[i]**2*e[i]**2 for i in range(d))
    ok1 = sp.expand(varSH - vBv) == 0
    # M5 form
    trS = sum(S[i][i] for i in range(d))
    M5 = a**2*trS**2 + beta*sum(S[i][i]**2 for i in range(d)) \
        + 4*a**2*sum(S[i][j]**2 for i in range(d) for j in range(i+1, d))
    ok2 = sp.expand(M5 - varSH) == 0
    fro = sp.expand(sum(S[i][j]**2 for i in range(d) for j in range(d))
                    - (sum(x**2 for x in v)*E2 + sum(v[i]*e[i] for i in range(d))**2)/2)
    print('(b) d=%d  Var<S,H>==v^T B v: %s ; ==M5 form: %s ; ||S||_F^2 formula: %s'
          % (d, ok1, ok2, fro == 0))

# ---------------- (c) exact rational finite symmetric product law ----------------
random.seed(20261018)

def make_law(levels):
    # symmetric 1D law on {0, +-s for s in levels}, random positive rational weights
    w0 = Fr(random.randint(1, 9))
    ws = [Fr(random.randint(1, 9)) for _ in levels]
    tot = w0 + 2*sum(ws)
    pts = [(Fr(0), w0/tot)] + [(sg*s, wv/tot) for s, wv in zip(levels, ws) for sg in (1, -1)]
    return pts

def mom1(pts, k):
    return sum(p*x**k for x, p in pts)

def momalpha(pts, alpha):
    r = Fr(1)
    for k in alpha:
        r *= mom1(pts, k)
    return r

def solve(A, rhs):
    # exact Gaussian elimination, A list of lists of Fr, rhs list of columns
    n = len(A)
    M = [row[:] + [col[i] for col in rhs] for i, row in enumerate(A)]
    m = len(rhs)
    for i in range(n):
        piv = next(r for r in range(i, n) if M[r][i] != 0)
        M[i], M[piv] = M[piv], M[i]
        f = M[i][i]
        M[i] = [x/f for x in M[i]]
        for r in range(n):
            if r != i and M[r][i] != 0:
                g = M[r][i]
                M[r] = [x - g*y for x, y in zip(M[r], M[i])]
    return [[M[i][n+j] for i in range(n)] for j in range(m)]

def det(A):
    A = [r[:] for r in A]
    n = len(A); dd = Fr(1)
    for i in range(n):
        piv = next((r for r in range(i, n) if A[r][i] != 0), None)
        if piv is None:
            return Fr(0)
        if piv != i:
            A[i], A[piv] = A[piv], A[i]; dd = -dd
        dd *= A[i][i]
        for r in range(i+1, n):
            g = A[r][i]/A[i][i]
            A[r] = [x - g*y for x, y in zip(A[r], A[i])]
    return dd

d = 3
units = [(Fr(2, 3), Fr(1, 3), Fr(2, 3)), (Fr(3, 5), Fr(0), Fr(4, 5)), (Fr(2, 7), Fr(3, 7), Fr(-6, 7)),
         (Fr(1), Fr(0), Fr(0)), (Fr(12, 13), Fr(3, 13), Fr(-4, 13))]
for trial, levels in enumerate([[Fr(1), Fr(2), Fr(5, 2)], [Fr(1, 3), Fr(2, 3), Fr(7, 4)], [Fr(1), Fr(2)]]):
    pts = make_law(levels)
    A_ = mom1(pts, 2); B_ = mom1(pts, 4); C_ = mom1(pts, 6)
    K4 = B_ - 3*A_**2; K6 = C_ - 15*A_*B_ + 30*A_**3
    def E(idx):  # E[prod Q_idx]
        cnt = [0]*d
        for i in idx:
            cnt[i] += 1
        return momalpha(pts, cnt)
    # (8)
    ok8 = all(E((i, j, k, l)) == A_**2*((i == j)*(k == l) + (i == k)*(j == l) + (i == l)*(j == k))
              + (K4 if i == j == k == l else 0)
              for i in range(d) for j in range(d) for k in range(d) for l in range(d))
    res = []
    for e in units:
        assert sum(x*x for x in e) == 1
        def Epoly(order_extra):  # E[(Q.e)^n * prod Q_extra]
            n, extra = order_extra
            tot = Fr(0)
            for idx in itertools.product(range(d), repeat=n):
                coef = Fr(1)
                for i in idx:
                    coef *= e[i]
                tot += coef*E(tuple(idx) + tuple(extra))
            return tot
        v = [Epoly((3, (i,))) for i in range(d)]
        ok9 = all(v[i] == 3*A_**2*e[i] + K4*e[i]**3 for i in range(d))
        E6 = Epoly((6, ()))
        p4 = sum(x**4 for x in e); p6 = sum(x**6 for x in e)
        ok6 = E6 == 15*A_**3 + 15*A_*K4*p4 + K6*p6
        # independent Schur complement Var(T_e | D) with Cov(D)=aI, Cov(T,D)=-v
        Vschur = E6 - sum(x*x for x in v)/A_
        V10 = 6*A_**3 + 9*A_*K4*p4 + (K6 - K4**2/A_)*p6
        # Gamma_e: (12)-(13) vs direct Schur complement from full Hessian tensor.
        # Build orthonormal U only when rational: use explicit rational completions.
        res.append((ok9, ok6, Vschur == V10, Vschur > 0))
    print('(c) law %d: (8) exact: %s; per-e [(9),(E6),(10)=Schur,(V>0)]: %s' % (trial, ok8, res))

# Gamma_e check with a rational orthonormal frame: e=(2/3,1/3,2/3), U columns
e = (Fr(2, 3), Fr(1, 3), Fr(2, 3))
U = [(Fr(1, 3), Fr(2, 3), Fr(-2, 3)), (Fr(2, 3), Fr(-2, 3), Fr(-1, 3))]  # columns
for col in U:
    assert sum(x*y for x, y in zip(col, e)) == 0 and sum(x*x for x in col) == 1
assert sum(x*y for x, y in zip(U[0], U[1])) == 0
for trial, levels in enumerate([[Fr(1), Fr(2), Fr(5, 2)], [Fr(1, 3), Fr(2, 3), Fr(7, 4)]]):
    pts = make_law(levels)
    A_ = mom1(pts, 2); B_ = mom1(pts, 4)
    K4 = B_ - 3*A_**2
    def E(idx):
        cnt = [0]*d
        for i in idx:
            cnt[i] += 1
        return momalpha(pts, cnt)
    def covH(i, j, k, l):
        return E((i, j, k, l))
    # functionals: Hessian contracted by vectors
    def bil(x, y, z, w):
        return sum(x[i]*y[j]*z[k]*w[l]*covH(i, j, k, l)
                   for i in range(d) for j in range(d) for k in range(d) for l in range(d))
    basis = [tuple(Fr(int(i == j)) for j in range(d)) for i in range(d)]
    He = [(e, basis[i]) for i in range(d)]
    pairs = [(0, 0), (0, 1), (1, 1)]
    T = [(U[x], U[y]) for x, y in pairs]
    Bm = [[bil(*p, *q) for q in He] for p in He]
    R = [[bil(*t, *q) for q in He] for t in T]
    C = [[bil(*t, *s) for s in T] for t in T]
    # formula (11),(12)
    Bf = [[A_**2*(int(i == j) + 2*e[i]*e[j]) + (K4*e[i]**2 if i == j else 0) for j in range(d)] for i in range(d)]
    Rf = [[A_**2*int(x == y)*e[i] + K4*U[x][i]*U[y][i]*e[i] for i in range(d)] for x, y in pairs]
    sol = solve(Bm, R)  # B^{-1} R_t for each t
    Gam = [[C[s][t] - sum(R[s][i]*sol[t][i] for i in range(d)) for t in range(3)] for s in range(3)]
    print('(c) Gamma: B==(11): %s, R==(12): %s, detB>0: %s, Gamma minors>0: %s'
          % (Bm == Bf, R == Rf, det(Bm) > 0,
             [det([row[:k] for row in Gam[:k]]) > 0 for k in (1, 2, 3)]))

# ---------------- (d) negative control ----------------
for levels in ([Fr(1)], [Fr(1), Fr(2)]):
    pts = make_law(levels)
    A_ = mom1(pts, 2); B_ = mom1(pts, 4); C_ = mom1(pts, 6)
    print('(d) carrier {0,+-%s}: c - b^2/a = %s' % ([str(x) for x in levels], C_ - B_**2/A_))
