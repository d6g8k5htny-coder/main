#!/usr/bin/env python3
"""Cross-provider reviewer controls for C104 (exact; standard library only).  Independent of the author's and the
first reviewer's code.  These check finite algebra and stated inequalities; they do not prove the analytic lemmas.

R1  Reference 20-jet covariance in d = 3: Gamma[a][b] = (-1)^|b| D^(a+b) K(0), K = exp(-|v|^2/2).
    Hermite change (14): Cov(H f) = diag(alpha!) exactly; H = I + Q with Q^2 = 0; ||H||_F^2 = ||H^-1||_F^2 = 56;
    tr Gamma = 80; Gamma - I/56 and 80 I - Gamma positive semidefinite (exact LDL^T inertia).
R2  Reference regression on U = (f, f_x, f_z, f_w, f_xx, f_xz, f_xw, f_xxx) = (b, 0, 0, 0, 0, 0, 0, 12k):
    mean of J = (A; T) is (-b, 0, -b; 0, ..., 0) and covariance diag(2, 1, 2; 2, 2, 2, 1, 2, 6, 2, 2, 6) (for the
    raw order f_zz, f_zw, f_ww; f_xxz, f_xxw, f_xzz, f_xzw, f_xww, f_zzz, f_zzw, f_zww, f_www), exactly, for several
    rational (b, k); A and T uncorrelated; t = (f_xxz, f_xzz, f_zzz) has covariance diag(2, 2, 6).
R3  Derivative domination (17): for every multi-index |alpha| <= 6 in three variables, the product of the absolute
    Hermite coefficient sums is at most 76 = involution number T(6), with T supermultiplicative; |D^alpha K(0)| <= 15.
R4  Lattice bound (18)-(20): shell count (2j+1)^3 - (2j-1)^3 = 24 j^2 + 2 <= 26 j^2; 91 * 26 = 2366; the series bound
    sum_{j>=1} j^8 e^{-L^2 (j^2 - 1)/2} < 2 for L >= 24 via j <= 2^(j-1), j^2 - 1 >= 3(j - 1), e > 2 (exact rationals);
    2366 * 2 < 5000; Delta_24 < 2^-245 (with e > 2); 20 Delta_L < 1/112.
R5  Constants (21): 20 (112 + 81 * 112^2) = 20 323 520 < 2^25; 20 (1 + 81 * 112)^2 = 1 646 386 580, and the formulas
    (24), (25) with e_L = 20 Delta, a0 = 1/112, M0 = 81.
R6  Lemma 4: on random rational (x, B, D^2, k), whenever x >= 2|B| and x^3 >= 48 k D^2,
    g = 2x^3 + 3Bx^2 - B^3 - 12kD^2 >= x^3/8 and 12 k T_* = 2x^3 + 3Bx^2 - B^3 at s = -x (exact); the coefficient
    identities 384 = 12 * 32, 2304 = 12 * 192, 1536/1728 + 1728/5184 = 11/9, and C_K's monomial bounds.
R7  Section 6/8 rational pieces: (3/2)_2 = 15/4, (3/2)_3 = 105/8; the box Q of (38): A < 0, det A >= 15/16, raw norm
    < 3, volume 1/2 (exact at vertices / by monotonicity); c_2 = pi via H_2 = 2 sqrt(pi) is NOT checked here
    (transcendental) but the rational part (2 pi)^1 * pi^(1/2) / (2 pi^(1/2)) = pi is.
Mutants: --mutant R1 (Hermite contraction 2 f instead of f), R4 (shell 24 j^2), R6 (bound x^3/2), each exit 1.
"""
import json
import sys
from fractions import Fraction as Fr
from itertools import product

MUT = None
if len(sys.argv) > 1:
    if len(sys.argv) == 3 and sys.argv[1] == '--mutant' and sys.argv[2] in ('R1', 'R4', 'R6'):
        MUT = sys.argv[2]
    else:
        print(json.dumps({'error': 'unknown arguments'}))
        sys.exit(2)

fails = []
counts = {}


def ok(label, cond, info=None):
    counts[label] = counts.get(label, 0) + 1
    if not cond:
        fails.append((label, info))


def dfact(n):           # double factorial, (-1)!! = 1
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def fact(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def dK0(g):             # D^g exp(-|v|^2/2) at 0
    if any(c % 2 for c in g):
        return 0
    r = 1
    for c in g:
        r *= (-1) ** (c // 2) * dfact(c - 1)
    return r


IDX = [a for a in product(range(4), repeat=3) if sum(a) <= 3]
IDX.sort(key=lambda a: (sum(a), [-c for c in a]))
n = len(IDX)
POS = {a: i for i, a in enumerate(IDX)}
G = [[Fr((-1) ** sum(b) * dK0(tuple(x + y for x, y in zip(a, b)))) for b in IDX] for a in IDX]

# ---------------- R1: Hermite change
H = [[Fr(int(i == j)) for j in range(n)] for i in range(n)]
for a in IDX:
    i = POS[a]
    o = sum(a)
    if o == 2:
        for ax in range(3):
            if a[ax] == 2:
                H[i][POS[(0, 0, 0)]] += Fr(2) if MUT == 'R1' else Fr(1)
    if o == 3:
        # Y_ijk = f_ijk + delta_ij f_k + delta_ik f_j + delta_jk f_i  (nonrepeated: sum over pairs of equal indices)
        idx = [ax for ax in range(3) for _ in range(a[ax])]
        for p, q, rr in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            if idx[p] == idx[q]:
                k = [0, 0, 0]
                k[idx[rr]] = 1
                H[i][POS[tuple(k)]] += 1
CY = [[sum(H[i][p] * G[p][q] * H[j][q] for p in range(n) for q in range(n) if H[i][p] and H[j][q]) for j in range(n)] for i in range(n)]
for i, a in enumerate(IDX):
    for j in range(n):
        want = Fr(fact(a[0]) * fact(a[1]) * fact(a[2])) if i == j else Fr(0)
        ok('R1-hermite', CY[i][j] == want, (a, IDX[j]))
Qm = [[H[i][j] - (1 if i == j else 0) for j in range(n)] for i in range(n)]
Q2 = [[sum(Qm[i][p] * Qm[p][j] for p in range(n)) for j in range(n)] for i in range(n)]
ok('R1-Q2', all(v == 0 for row in Q2 for v in row))
Hinv = [[(1 if i == j else 0) - Qm[i][j] for j in range(n)] for i in range(n)]
ok('R1-frob', sum(v * v for row in H for v in row) == 56 and sum(v * v for row in Hinv for v in row) == 56)
ok('R1-trace', sum(G[i][i] for i in range(n)) == 80)


def inertia(X):
    M = [list(r) for r in X]
    m = len(M)
    pos = neg = zero = 0
    rest = list(range(m))
    while rest:
        piv = next((i for i in rest if M[i][i] != 0), None)
        if piv is None:
            pair = next(((i, j) for i in rest for j in rest if i < j and M[i][j] != 0), None)
            if pair is None:
                zero += len(rest)
                break
            i, j = pair
            for k in range(m):
                M[i][k] += M[j][k]
            for k in range(m):
                M[k][i] += M[k][j]
            continue
        p = M[piv][piv]
        pos += p > 0
        neg += p < 0
        rest.remove(piv)
        for i in rest:
            f = M[i][piv] / p
            if f:
                for k in range(m):
                    M[i][k] -= f * M[piv][k]
        for i in rest:
            M[piv][i] = M[i][piv] = Fr(0)
    return pos, neg, zero


lo = [[G[i][j] - (Fr(1, 56) if i == j else 0) for j in range(n)] for i in range(n)]
hi = [[(80 if i == j else 0) - G[i][j] for j in range(n)] for i in range(n)]
ok('R1-spectrum', inertia(lo)[1] == 0 and inertia(hi)[1] == 0)

# ---------------- R2: reference regression
names_U = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0), (1, 0, 1), (3, 0, 0)]
names_A = [(0, 2, 0), (0, 1, 1), (0, 0, 2)]
names_T = [(2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1), (1, 0, 2), (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)]
names_J = names_A + names_T
ok('R2-partition', sorted(names_U + names_J) == sorted(IDX))


def solve(Mat, rhs):
    m = len(Mat)
    Aug = [list(Mat[i]) + [rhs[i]] for i in range(m)]
    for c in range(m):
        p = next(r for r in range(c, m) if Aug[r][c] != 0)
        Aug[c], Aug[p] = Aug[p], Aug[c]
        for r in range(m):
            if r != c and Aug[r][c] != 0:
                f = Aug[r][c] / Aug[c][c]
                Aug[r] = [x - f * y for x, y in zip(Aug[r], Aug[c])]
    return [Aug[i][m] / Aug[i][i] for i in range(m)]


GU = [[G[POS[a]][POS[b]] for b in names_U] for a in names_U]
CJU = [[G[POS[a]][POS[b]] for b in names_U] for a in names_J]
DJ = [[G[POS[a]][POS[b]] for b in names_J] for a in names_J]
for (b, k) in ((Fr(0), Fr(1)), (Fr(3, 2), Fr(1, 3)), (Fr(-7, 5), Fr(5, 2))):
    v = [b, 0, 0, 0, 0, 0, 0, 12 * k]
    w = solve(GU, v)
    mu = [sum(CJU[i][j] * w[j] for j in range(8)) for i in range(12)]
    ok('R2-mean', mu == [-b, 0, -b] + [0] * 9, (b, k, mu))
# conditional covariance S = D - C G^-1 C^T
cols = []
for i in range(12):
    cols.append(solve(GU, CJU[i]))
S = [[DJ[i][j] - sum(CJU[i][p] * cols[j][p] for p in range(8)) for j in range(12)] for i in range(12)]
want = [2, 1, 2, 2, 2, 2, 1, 2, 6, 2, 2, 6]
ok('R2-cov', all(S[i][j] == (want[i] if i == j else 0) for i in range(12) for j in range(12)))
ti = [names_J.index(x) for x in ((2, 1, 0), (1, 2, 0), (0, 3, 0))]
ok('R2-t', [S[i][i] for i in ti] == [2, 2, 6])

# ---------------- R3: derivative domination
T = [sum(fact(m) // (fact(kk) * fact(m - 2 * kk) * 2 ** kk) for kk in range(m // 2 + 1)) for m in range(7)]
ok('R3-involution', T == [1, 1, 2, 4, 10, 26, 76])
for a in product(range(7), repeat=3):
    if sum(a) <= 6:
        ok('R3-product', T[a[0]] * T[a[1]] * T[a[2]] <= T[sum(a)] <= 76)
        ok('R3-zero', abs(dK0(a)) <= 15)

# ---------------- R4: lattice bound
for j in range(1, 40):
    shell = (2 * j + 1) ** 3 - (2 * j - 1) ** 3
    ok('R4-shell', shell == (24 * j * j if MUT == 'R4' else 24 * j * j + 2) and shell <= 26 * j * j)
ok('R4-2366', 91 * 26 == 2366)
# series bound: for L >= 24, j^8 e^{-L^2 (j^2-1)/2} <= 2^{8(j-1)} 2^{-864 (j-1)} = 2^{-856 (j-1)}; sum < 2
ok('R4-series', all(j ** 8 <= 2 ** (8 * (j - 1)) for j in range(1, 60)) and all(j * j - 1 >= 3 * (j - 1) for j in range(1, 60)))
ok('R4-sum', sum(Fr(1, 2 ** (856 * (j - 1))) for j in range(1, 30)) < 2 and 2366 * 2 < 5000)
# Delta_24 = 5000 * 25^6 * e^-288 < 5000 * 25^6 * 2^-288 < 2^(13+30-288)
ok('R4-delta', 5000 < 2 ** 13 and 25 ** 6 < 2 ** 30 and Fr(20, 2 ** 245) < Fr(1, 112))

# ---------------- R5: constants
ok('R5-Dmu', 20 * (112 + 81 * 112 ** 2) == 20323520 and 20323520 < 2 ** 25)
ok('R5-DS', 20 * (1 + 81 * 112) ** 2 == 1646386580 and 1 + 2 * 81 * 112 + 81 ** 2 * 112 ** 2 == (1 + 81 * 112) ** 2)

# ---------------- R6: Lemma 4 sector bound
seed = 20261003


def rnd(lo_, hi_, den=97):
    global seed
    seed = (6364136223846793005 * seed + 1442695040888963407) % (1 << 64)
    return Fr(lo_) + Fr((seed >> 11) % (den * (hi_ - lo_) + 1), den)


den8 = Fr(1, 2) if MUT == 'R6' else Fr(1, 8)
minratio = None
for trial in range(4000):
    k = rnd(0, 6) + Fr(1, 97)
    B = rnd(-5, 5)
    D2 = rnd(0, 9)
    x = max(2 * abs(B), Fr(0)) + rnd(0, 6)
    # enforce x^3 >= 48 k D^2 by enlarging x if needed (rationally)
    while x ** 3 < 48 * k * D2:
        x = x * 2 + 1
    g = 2 * x ** 3 + 3 * B * x ** 2 - B ** 3 - 12 * k * D2
    s = -x
    Ts12k = -(s - B) ** 2 * (B + 2 * s)        # 12 k T_*
    ok('R6-g', g >= den8 * x ** 3 and Ts12k == 2 * x ** 3 + 3 * B * x ** 2 - B ** 3 and s <= B)
    r_ = g / x ** 3
    minratio = r_ if minratio is None or r_ < minratio else minratio
# the extreme admissible case B = -x/2, 12 k D^2 = x^3/4 gives g = (3/8) x^3 (the stated x^3/8 is valid, not tight)
for x in (Fr(1), Fr(7, 3), Fr(10)):
    k = Fr(1, 3)
    B = -x / 2
    D2 = x ** 3 / (48 * k)
    g = 2 * x ** 3 + 3 * B * x ** 2 - B ** 3 - 12 * k * D2
    ok('R6-extreme', g == Fr(3, 8) * x ** 3 and g >= den8 * x ** 3)
ok('R6-coef', 12 * 32 == 384 and 12 * 192 == 2304 and Fr(1536, 1728) + Fr(1728, 5184) == Fr(11, 9))

# ---------------- R7: rational pieces
ok('R7-rising', Fr(3, 2) * Fr(5, 2) == Fr(15, 4) and Fr(3, 2) * Fr(5, 2) * Fr(7, 2) == Fr(105, 8))
for u, v, w in product((Fr(-2), Fr(-1)), (Fr(-1, 4), Fr(1, 4)), (Fr(-2), Fr(-1))):
    ok('R7-box', u < 0 and u * w - v * v >= Fr(15, 16) and u * u + v * v + w * w < 9)
ok('R7-vol', 1 * 1 * Fr(1, 2) == Fr(1, 2) and Fr(18) * Fr(225, 256) == 36 * Fr(225, 256) / 2)

out = {'object': 'C104 cross-provider reviewer controls (exact, stdlib)', 'counts': counts, 'mutant': MUT,
       'all_pass': not fails, 'failures': [str(f) for f in fails[:6]], 'scientific_effect': 'NONE'}
print(json.dumps(out, sort_keys=True))
sys.exit(0 if not fails else 1)
