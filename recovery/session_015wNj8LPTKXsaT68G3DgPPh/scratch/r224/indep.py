"""Independent recheck of Math-#224 (Claude nonauthor read). Own code; does not import the packet."""
from fractions import Fraction as F
from itertools import product
import mpmath as mp
mp.mp.dps = 60

def dfact(k):  # (k-1)!! for even k, product of odd numbers below k
    r = 1
    for j in range(k - 1, 0, -2): r *= j
    return r

def cov0(a, b):
    """Cov(d^a f, d^b f) = (-1)^{|b|} d^{a+b} phi(0), phi = exp(-|x|^2/2)."""
    s = [x + y for x, y in zip(a, b)]
    if any(x % 2 for x in s): return F(0)
    v = (-1) ** (sum(s) // 2)
    for x in s: v *= dfact(x)
    return F((-1) ** sum(b) * v)

def mi(d, *ax):
    m = [0] * d
    for a in ax: m[a] += 1
    return tuple(m)

def labels(d, birth=False):
    u = 0; th = list(range(1, d))
    L = []
    if birth: L.append(mi(d))
    L += [mi(d, i) for i in range(d)]                 # grad
    L += [mi(d, 0, i) for i in range(d)]              # H u
    L.append(mi(d, 0, 0, 0))                          # f_uuu
    L += [mi(d, i, j) for k, i in enumerate(th) for j in th[k:]]  # A upper
    L += [mi(d, 0, 0, i) for i in th]                 # gamma
    L.append(mi(d, 0, 0, 0, 0))                       # f4
    return L

def ldl_pivots(M):
    n = len(M); A = [row[:] for row in M]; piv = []
    for k in range(n):
        p = A[k][k]; piv.append(p)
        if p <= 0: return piv, False
        for i in range(k + 1, n):
            f = A[i][k] / p
            for j in range(k + 1, n): A[i][j] -= f * A[k][j]
    return piv, True

print("== Slice A ==")
for d in (1, 2, 3):
    L = labels(d); N = len(L)
    assert N == 3 * d + 1 + d * (d - 1) // 2
    C = [[cov0(a, b) for b in L] for a in L]
    for fl in (F(1, 3), F(1, 2)):
        piv, ok = ldl_pivots([[C[i][j] - (fl if i == j else 0) for j in range(N)] for i in range(N)])
        print(f"d={d} N={N} C0-{fl}I PD: {ok}", "pivots", [str(p) for p in piv] if fl == F(1, 3) else "")
    Lb = labels(d, True)
    Cb = [[cov0(a, b) for b in Lb] for a in Lb]
    for fl in (F(1, 3), F(1, 4)):
        piv, ok = ldl_pivots([[Cb[i][j] - (fl if i == j else 0) for j in range(len(Lb))] for i in range(len(Lb))])
        print(f"   birth-included N={len(Lb)} minus {fl}I PD: {ok}")
    print("   Schur (417d+2018)/(15(3d+5)) =", F(417 * d + 2018, 15 * (3 * d + 5)))

# (T7): conditional on Q=0 (birth integrated) vs conditional on (f=b, Q=0)
def condcov(d, keep, given):
    import itertools
    G = [[cov0(a, b) for b in given] for a in given]
    n = len(G)
    # exact inverse via Gauss-Jordan
    A = [row[:] + [F(int(i == j)) for j in range(n)] for i, row in enumerate(G)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0); A[c], A[p] = A[p], A[c]
        pv = A[c][c]; A[c] = [x / pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]; A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    Gi = [row[n:] for row in A]
    K = [[cov0(a, b) for b in given] for a in keep]
    out = [[cov0(a, b) - sum(K[i][k] * Gi[k][l] * K[j][l] for k in range(n) for l in range(n))
            for j, b in enumerate(keep)] for i, a in enumerate(keep)]
    return out

d = 3
Q = [mi(d, i) for i in range(d)] + [mi(d, 0, i) for i in range(d)] + [mi(d, 0, 0, 0)]
keep = [mi(d, 0, 0, 0, 0), mi(d, 1, 1), mi(d, 2, 2)]
cm = condcov(d, keep, Q)
print("(T7) Cov(f4,A11,A22 | Q=0):", [[str(x) for x in r] for r in cm])
cb = condcov(d, keep + [mi(d)], Q)
print("     Var(f | Q=0) =", cb[3][3], " Cov(f4,f|Q)=", cb[0][3], " Cov(A11,f|Q)=", cb[1][3])
cf = condcov(d, keep, Q + [mi(d)])
print("     Cov(f4,A11,A22 | f,Q):", [[str(x) for x in r] for r in cf])

print("== Slice B ==")
T = [1, 1]
for j in range(2, 9): T.append(T[-1] + (j - 1) * T[-2])
print("involution numbers T_0..T_8:", T, " 7!! =", dfact(8))

# exact E_d(L), delta, eta with mpmath; upward-rounded table check
table = {1: ("6.89628366587e-9", "3.29775477012e-109"), 2: ("1.32408646215e-6", "6.33168915861e-107"),
         3: ("4.90170601596e-5", "2.34396164673e-105")}
for d in (1, 2, 3):
    N = 3 * d + 1 + d * (d - 1) // 2
    for k, Lside in enumerate((10, 24)):
        E = 3 * d * 3 ** (d - 1) * (764 * d ** 4 * mp.mpf(Lside) ** 8 + 105) * mp.e ** (-mp.mpf(Lside) ** 2 / 2)
        eta = 39 * N * E
        disp = mp.mpf(table[d][k])
        ok = disp >= eta and (disp - eta) / eta < mp.mpf("1e-11")
        print(f"d={d} L={Lside} delta={mp.nstr(3*N*E,8)} eta={mp.nstr(eta,15)} displayed-upward-ok={ok}")

# direct lattice check of the entrywise bound in a random frame, d=3, L=10
def Dphi_contract(x, vs):
    """D^j phi(x)[v1..vj] by partial-matching formula (independent of T12's count)"""
    j = len(vs)
    def rec(idx):
        if not idx: return mp.mpf(1)
        i0, rest = idx[0], idx[1:]
        tot = -mp.fsum(xi * vi for xi, vi in zip(x, vs[i0])) * rec(rest)
        for t in range(len(rest)):
            pair = -mp.fsum(a * b for a, b in zip(vs[i0], vs[rest[t]]))
            tot += pair * rec(rest[:t] + rest[t + 1:])
        return tot
    return rec(tuple(range(j))) * mp.e ** (-mp.fsum(xi * xi for xi in x) / 2)

import random
random.seed(224)
def rand_frame(d):
    M = mp.matrix([[random.gauss(0, 1) for _ in range(d)] for _ in range(d)])
    Qm, _ = mp.qr(M)
    return [[Qm[i, j] for i in range(d)] for j in range(d)]  # columns
for d in (2, 3):
    Lside = 10
    fr = rand_frame(d)
    Lb = labels(d)
    def KL_deriv(s):  # D^{s} K_L(0) in frame directions, s multi-index
        vs = [fr[i] for i, c in enumerate(s) for _ in range(c)]
        num = mp.mpf(0); SL = mp.mpf(0)
        for n in product(range(-2, 3), repeat=d):
            x = [Lside * ni for ni in n]
            num += Dphi_contract(x, vs)
            if any(n): SL += mp.e ** (-mp.fsum(xi * xi for xi in x) / 2)
        return num / (1 + SL)
    maxdev = mp.mpf(0)
    seen = {}
    for a in Lb:
        for b in Lb:
            s = tuple(x + y for x, y in zip(a, b))
            if s not in seen: seen[s] = KL_deriv(s)
            dev = abs((-1) ** sum(b) * seen[s] - cov0(a, b))
            maxdev = max(maxdev, dev)
    N = len(Lb)
    E = 3 * d * 3 ** (d - 1) * (764 * d ** 4 * mp.mpf(Lside) ** 8 + 105) * mp.e ** (-mp.mpf(Lside) ** 2 / 2)
    print(f"d={d} random frame, L=10: max|C_L-C_0| = {mp.nstr(maxdev,6)}  <= E_d = {mp.nstr(E,6)}: {maxdev <= E}")
