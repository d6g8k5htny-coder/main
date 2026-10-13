# exact check of the d = 3 contact laws (reference kernel exp(-|z|^2/2)), rational arithmetic
from fractions import Fraction as Fr
import itertools, math
def h(n):
    return 0 if n % 2 else (-1) ** (n // 2) * math.prod(range(n - 1, 0, -2))
def cov(a, b):   # E[d^a f(0) d^b f(0)] = (-1)^|b| d^{a+b} G(0)
    return Fr((-1) ** sum(b) * math.prod(h(x + y) for x, y in zip(a, b)))
pins = [(0,0,0), (1,0,0), (2,0,0), (3,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1)]   # f, f_x, f_xx, f_xxx, f_y1, f_xy1, f_y2, f_xy2
vals = lambda b, k: [b, 0, 0, 12 * k, 0, 0, 0, 0]
targets = {'v1': (2,1,0), 'v2': (2,0,1), 'O11': (1,2,0), 'O12': (1,1,1), 'O22': (1,0,2),
           'T111': (0,3,0), 'T112': (0,2,1), 'T122': (0,1,2), 'T222': (0,0,3),
           'A11': (0,2,0), 'A12': (0,1,1), 'A22': (0,0,2)}
def solve(M, rhs):
    n = len(M); A = [row[:] + [r] for row, r in zip(M, rhs)]
    for c in range(n):
        p = next(i for i in range(c, n) if A[i][c] != 0); A[c], A[p] = A[p], A[c]
        for i in range(n):
            if i != c and A[i][c] != 0:
                f = A[i][c] / A[c][c]; A[i] = [x - f * y for x, y in zip(A[i], A[c])]
    return [A[i][n] / A[i][i] for i in range(n)]
S = [[cov(p, q) for q in pins] for p in pins]
names = list(targets)
W = {n: solve(S, [cov(targets[n], q) for q in pins]) for n in names}
b, k = Fr(7, 5), Fr(3, 4)
mean = {n: sum(w * x for w, x in zip(W[n], vals(b, k))) for n in names}
C = {(m, n): cov(targets[m], targets[n]) - sum(W[m][i] * cov(targets[n], pins[i]) for i in range(8)) for m in names for n in names}
for n in names: print(n, 'mean', mean[n], 'var', C[(n, n)])
off = [(m, n, C[(m, n)]) for m in names for n in names if m < n and C[(m, n)] != 0]
print('nonzero conditional covariances:', off)
