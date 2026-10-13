"""Propagation of the jet-covariance deviation delta through the contact regressions (prototype)."""
import math
from ia import *
from torus_proto import delta_bound

def dfact(n):  # (n-1)!! for even n handled by caller: double factorial of odd numbers
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r

def D(g):
    """d^g G(0) for G = exp(-|z|^2/2): (-1)^((g1+g2)/2) (g1-1)!! (g2-1)!! if both even, else 0."""
    g1, g2 = g
    if g1 % 2 or g2 % 2:
        return 0
    return (-1) ** ((g1 + g2) // 2) * dfact(g1 - 1) * dfact(g2 - 1)

def cov_ref(a, b):
    return (-1) ** (b[0] + b[1]) * D((a[0] + b[0], a[1] + b[1]))

def icov(jets, delta):
    return [[(dn(cov_ref(a, b) - delta), up(cov_ref(a, b) + delta)) for b in jets] for a in jets]

# ---- interval 3x3 / 2x2 linear algebra
def det3(M):
    return sub(add(mul(M[0][0], sub(mul(M[1][1], M[2][2]), mul(M[1][2], M[2][1]))),
                   mul(M[0][2], sub(mul(M[1][0], M[2][1]), mul(M[1][1], M[2][0])))),
               mul(M[0][1], sub(mul(M[1][0], M[2][2]), mul(M[1][2], M[2][0]))))

def inv3(M):
    d = det3(M)
    C = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            r = [x for x in range(3) if x != i]
            c = [x for x in range(3) if x != j]
            m = sub(mul(M[r[0]][c[0]], M[r[1]][c[1]]), mul(M[r[0]][c[1]], M[r[1]][c[0]]))
            C[i][j] = m if (i + j) % 2 == 0 else neg(m)
    return [[div(C[j][i], d) for j in range(3)] for i in range(3)]   # adjugate / det

def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[ (lambda i, j: __import__('functools').reduce(add, [mul(A[i][t], B[t][j]) for t in range(m)]))(i, j) for j in range(p)] for i in range(n)]

def transpose(A):
    return [list(r) for r in zip(*A)]

delta = delta_bound(10.0)
# odd block
Y = [(1, 0), (0, 1), (3, 0)]; X = [(2, 1), (1, 2), (0, 3)]
SYY = icov(Y, delta); SXY = [[(dn(cov_ref(a, b) - delta), up(cov_ref(a, b) + delta)) for b in Y] for a in X]
SXX = icov(X, delta)
Ri = inv3(SYY)
Bcoef = matmul(SXY, Ri)                  # regression coefficients
Sc = [[sub(SXX[i][j], matmul(Bcoef, transpose(SXY))[i][j]) for j in range(3)] for i in range(3)]
v = [Bcoef[i][2] for i in range(3)]      # mean = 12 k v
print('delta', delta)
print('mean coefficient v (mean = 12k v):', v)
print('Sc:', Sc)
# epsilon: Frobenius norm of Sigma0^-1/2 (Sc - Sigma0) Sigma0^-1/2
S0 = [2.0, 2.0, 6.0]
eps2 = 0.0
for i in range(3):
    for j in range(3):
        dij = sub(Sc[i][j], ((S0[i] if i == j else 0.0),) * 2)
        e = absmax(dij) / math.sqrt(S0[i] * S0[j]) * (1 + 1e-15)
        eps2 = up(eps2 + up(e * e))
eps = up(math.sqrt(eps2) * (1 + 1e-15))
mu2 = 0.0
for i in range(3):
    m = up(24.0 * absmax(v[i]))          # k <= 2
    mu2 = up(mu2 + up(m * m / S0[i]))
print('eps', eps, 'mu~^2', mu2)
# chi^2 <= (1 - eps^2)^(-3/2) (exp(m) - 1) + ((1 - eps^2)^(-3/2) - 1), m = mu2/(1 - eps)
m = up(mu2 / (1 - eps) * (1 + 1e-15))
chi2 = up(1.01 * (up(m * math.exp(m) * 1.0000001)) + up(2.0 * eps * eps))
print('chi^2 <=', chi2)
