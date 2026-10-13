"""Independent numerical cross-check of PROOF.md Section 3 ranks (9) and the
Section 4 Jacobian identity (10). d=2 and d=3, L=1, K5 box, random sites/frames.

Rows checked (Section 3, (9)):
  O_(p,q)             : F,gradF at p and at q               -> 2d+2 rows
  U_0                 : F,F_u,F_uu,F_uuu,(F_vi,F_uvi)_i     -> 2d+2 rows
  U_0 + sym A entries : + F_{vi vj}, i<=j                    -> + d(d-1)/2
  U_0 + (F,grad)(p)   : + d+1
  U_0 + first jets p,q: + 2(d+1)
For each, the rank of the row-functional matrix on span{cos,sin of K5 modes}
must equal the number of rows; the same with the constant mode removed (P5^0).
Also K3 is checked for the single-site/two-site rows as the text states.
"""
import itertools, numpy as np
rng = np.random.default_rng(12345)

def modes(d, K):
    """one representative per nonzero pair {n,-n} plus zero"""
    reps = []
    for n in itertools.product(range(-K, K+1), repeat=d):
        n = np.array(n)
        if np.all(n == 0):
            continue
        # representative: first nonzero coordinate positive
        first = n[np.nonzero(n)[0][0]]
        if first > 0:
            reps.append(n)
    return reps

def basis_funcs(d, K, omega, zero_mean=False):
    """returns list of (kind, n): kind in {'c','s','1'}"""
    B = [] if zero_mean else [('1', np.zeros(d, int))]
    for n in modes(d, K):
        B.append(('c', n)); B.append(('s', n))
    return B

def deriv(kind, n, z, omega, alpha):
    """derivative D^alpha (alpha a tuple of coordinate indices) of basis func at z"""
    xi = omega * n
    k = len(alpha)
    coef = np.prod([xi[a] for a in alpha]) if k else 1.0
    ph = xi @ z
    if kind == '1':
        return 1.0 if k == 0 else 0.0
    # d^k/dt^k cos = cos(ph + k pi/2) etc.
    if kind == 'c':
        return coef * np.cos(ph + k*np.pi/2)
    else:
        return coef * np.sin(ph + k*np.pi/2)

def dir_deriv(kind, n, z, omega, dirs):
    """directional derivative along unit vectors dirs (list), via multilinearity"""
    d = len(z)
    tot = 0.0
    for alpha in itertools.product(range(d), repeat=len(dirs)):
        w = np.prod([dirs[i][alpha[i]] for i in range(len(dirs))])
        if w != 0:
            tot += w * deriv(kind, n, z, omega, alpha)
    return tot

def rows_for(d, omega, x, frame, p, q, which):
    """return list of row functionals as callables on basis (kind,n)"""
    u = frame[0]; vs = frame[1:]
    e = np.eye(d)
    R = []
    if 'O' in which:
        for site in (p, q):
            R.append(lambda kn, s=site: dir_deriv(kn[0], kn[1], s, omega, []))
            for j in range(d):
                R.append(lambda kn, s=site, j=j: dir_deriv(kn[0], kn[1], s, omega, [e[j]]))
    if 'U0' in which:
        R.append(lambda kn: dir_deriv(kn[0], kn[1], x, omega, []))
        R.append(lambda kn: dir_deriv(kn[0], kn[1], x, omega, [u]))
        R.append(lambda kn: dir_deriv(kn[0], kn[1], x, omega, [u, u]))
        R.append(lambda kn: dir_deriv(kn[0], kn[1], x, omega, [u, u, u]))
        for v in vs:
            R.append(lambda kn, v=v: dir_deriv(kn[0], kn[1], x, omega, [v]))
            R.append(lambda kn, v=v: dir_deriv(kn[0], kn[1], x, omega, [u, v]))
    if 'A' in which:
        for i in range(len(vs)):
            for j in range(i, len(vs)):
                R.append(lambda kn, a=vs[i], b=vs[j]: dir_deriv(kn[0], kn[1], x, omega, [a, b]))
    if 'p' in which:
        R.append(lambda kn: dir_deriv(kn[0], kn[1], p, omega, []))
        for j in range(d):
            R.append(lambda kn, j=j: dir_deriv(kn[0], kn[1], p, omega, [e[j]]))
    if 'q' in which:
        R.append(lambda kn: dir_deriv(kn[0], kn[1], q, omega, []))
        for j in range(d):
            R.append(lambda kn, j=j: dir_deriv(kn[0], kn[1], q, omega, [e[j]]))
    return R

def rank_of(d, K, zero_mean, which, trial):
    L = 1.0; omega = 2*np.pi/L
    x, p, q = rng.uniform(0, L, d), rng.uniform(0, L, d), rng.uniform(0, L, d)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    frame = [Q[:, i] for i in range(d)]
    B = basis_funcs(d, K, omega, zero_mean)
    R = rows_for(d, omega, x, frame, p, q, which)
    M = np.array([[r(kn) for kn in B] for r in R])
    s = np.linalg.svd(M, compute_uv=False)
    return len(R), np.linalg.matrix_rank(M, tol=1e-9 * s[0]), s[-1] / s[0]

configs = [
    ('O',), ('U0',), ('U0', 'A'), ('U0', 'p'), ('U0', 'p', 'q'), ('U0', 'A', 'p', 'q'),
]
for d in (2, 3):
    for K in (3, 5):
        for zm in (False, True):
            for which in configs:
                worst = None
                for t in range(3):
                    nrows, rk, cond = rank_of(d, K, zm, which, t)
                    worst = (nrows, rk, cond) if worst is None else (nrows, min(worst[1], rk), min(worst[2], cond))
                print(f"d={d} K={K} zero_mean={zm} rows={'+'.join(which):10s} nrows={worst[0]:2d} rank={worst[1]:2d} min_sv_ratio={worst[2]:.2e}")

# ---- Section 4 Jacobian identity (10): det D Phi = det(Hess F)/det B(p) up to sign
print("\nJacobian identity check (d=2,K=2, random tail = fixed smooth K4 poly):")
d, K, L = 2, 2, 1.0; omega = 2*np.pi/L
B = basis_funcs(d, K, omega)
N = len(B)
# tail: a fixed trig polynomial with K=4 modes outside block (treat as deterministic)
Bt = [kn for kn in basis_funcs(d, 4, omega) if np.max(np.abs(kn[1])) > K]
tail_c = rng.normal(size=len(Bt))
def gradT(z):
    return np.array([sum(c*deriv(kn[0], kn[1], z, omega, (j,)) for c, kn in zip(tail_c, Bt)) for j in range(d)])
def hessT(z):
    return np.array([[sum(c*deriv(kn[0], kn[1], z, omega, (i, j)) for c, kn in zip(tail_c, Bt)) for j in range(d)] for i in range(d)])
def Gmat(z):  # coefficient -> gradient matrix, d x N
    return np.array([[deriv(kn[0], kn[1], z, omega, (j,)) for kn in B] for j in range(d)])
def Hten(z):  # coefficient -> hessian, d x d x N
    return np.array([[[deriv(kn[0], kn[1], z, omega, (i, j)) for kn in B] for j in range(d)] for i in range(d)])
p0 = rng.uniform(0, L, d)
G = Gmat(p0)
# choose d columns with invertible minor
cols = None
for c in itertools.combinations(range(N), d):
    if abs(np.linalg.det(G[:, c])) > 0.3:
        cols = list(c); break
rest = [i for i in range(N) if i not in cols]
eta = rng.normal(size=len(rest))
def a_of(p, eta):
    G = Gmat(p)
    return -np.linalg.solve(G[:, cols], gradT(p) + G[:, rest] @ eta)
def Phi(p, eta):
    return np.concatenate([a_of(p, eta), eta])
# numeric Jacobian of Phi wrt (p, eta)
h = 1e-6
z0 = np.concatenate([p0, eta])
J = np.zeros((N, N))
for i in range(N):
    dz = np.zeros(N); dz[i] = h
    J[:, i] = (Phi(z0[:d]+dz[:d], z0[d:]+dz[d:]) - Phi(z0[:d]-dz[:d], z0[d:]-dz[d:])) / (2*h)
coef = np.zeros(N); coef[cols] = a_of(p0, eta); coef[rest] = eta
HessF = hessT(p0) + Hten(p0) @ coef
detB = np.linalg.det(G[:, cols])
print("  |det DPhi|            =", abs(np.linalg.det(J)))
print("  |det HessF|/|det B(p)| =", abs(np.linalg.det(HessF)) / abs(detB))
print("  gradient at (p0,coef) =", gradT(p0) + G @ coef, "(should be ~0)")
