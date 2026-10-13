import math, random
from math import sqrt, erf
from proto import conditional, chol
from proto2 import inner_prob, det_bounds, ORDERS
r, b, k = 0.3, 0.0, 1.0
mu, C = conditional(r, b, k)
print('corr matrix')
for i in range(6): print(['%7.4f' % (C[i][j] / sqrt(C[i][i] * C[j][j])) for j in range(6)])
# eigenvalues via numpy-free power? just print pivots of Cholesky in each order
for o in ORDERS:
    P = [[C[i][j] for j in o] for i in o]
    L = chol([[P[i][j] + (1e-18 if i == j else 0) for j in range(6)] for i in range(6)])
    print(o, 'pivots', ['%.2e' % L[i][i] for i in range(6)])
# MC of one slab box
y0, y1 = -2.0, -1.5; c1, c2, dl = 2.5, 2.0, 0.4
sd = [sqrt(C[i][i]) for i in range(6)]
lo = [mu[0] - c1 * sd[0], -c2 * sd[1], y0, mu[3] - c1 * sd[3], -c2 * sd[4], y0 - dl]
hi = [mu[0] + c1 * sd[0], c2 * sd[1], y1, mu[3] + c1 * sd[3], c2 * sd[4], y1 + dl]
L = chol([[C[i][j] + (1e-18 if i == j else 0) for j in range(6)] for i in range(6)])
rng = random.Random(3); n = 200000; cnt = 0; acc = 0.0
for _ in range(n):
    z = [rng.gauss(0, 1) for _ in range(6)]
    h = [mu[i] + sum(L[i][j] * z[j] for j in range(6)) for i in range(6)]
    if all(lo[i] <= h[i] <= hi[i] for i in range(6)):
        cnt += 1
        acc += abs(h[0] * h[2] - h[1] ** 2) * abs(h[3] * h[5] - h[4] ** 2)
print('MC P(box) %.5f  E[W 1box] %.5f' % (cnt / n, acc / n))
for o in ORDERS:
    print(o, 'inner_prob %.5f' % inner_prob(mu, C, lo, hi, o), 'detmin %.5f' % det_bounds(lo, hi))
