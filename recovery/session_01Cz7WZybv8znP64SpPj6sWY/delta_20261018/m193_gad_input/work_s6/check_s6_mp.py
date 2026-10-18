"""High-precision (mpmath, dps=50) check of proof Sec.6 eqs (10)-(14) for
irrational unit directions e and the ACTUAL lattice Gaussian spectral law
w_k ∝ exp(-(2 pi k/L)^2/2) (truncated at |k|<=K, tail < 1e-45) and for a
heavy-zero finite law with kappa_4>0.

Covariances are built directly from the random Fourier construction:
at spectral point Q with mass w, the X-(cos) and Y-(sin) coefficients of
  F(0): (1,0); D_i: (0,Q_i); (He)_i: (-Q_i (Q.e),0);
  (U^T H U)_pq: (-(Q.u_p)(Q.u_q),0); T_e=d_e^3F: (0,-(Q.e)^3);
Cov = sum_Q w (cos cos' + sin sin').
Then Y=(T_e, s(UHU)) is conditioned on Z=(D,He)=0 by Schur complement and
compared with the proof's closed forms V(e) (10) and Gamma_e (13) built from
B (11), R (12), C. Also compared with V2's (M10) positive decomposition.

Run: <venv>/bin/python -I check_s6_mp.py
"""
import mpmath as mp
from itertools import product
mp.mp.dps = 50

def lattice_law(L, tol=mp.mpf('1e-45')):
    h = 2 * mp.pi / L
    K = 0
    while mp.e ** (-(h * K) ** 2 / 2) > tol: K += 1
    ws = {k: mp.e ** (-(h * k) ** 2 / 2) for k in range(-K, K + 1)}
    z = mp.fsum(ws.values())
    return [(h * k, w / z) for k, w in ws.items()]

def heavy_zero_law():
    return [(mp.mpf(0), mp.mpf(9) / 10), (mp.mpf(1), mp.mpf(1) / 30), (mp.mpf(-1), mp.mpf(1) / 30),
            (mp.mpf(5) / 2, mp.mpf(1) / 60), (mp.mpf(-5) / 2, mp.mpf(1) / 60)]

def gram_schmidt_perp(e):
    d = len(e)
    basis = []
    for k in range(d):
        v = [mp.mpf(int(i == k)) for i in range(d)]
        for b in [e] + basis:
            pr = mp.fsum(x * y for x, y in zip(v, b))
            v = [x - pr * y for x, y in zip(v, b)]
        nv = mp.sqrt(mp.fsum(x * x for x in v))
        if nv > mp.mpf('1e-10'):
            basis.append([x / nv for x in v])
        if len(basis) == d - 1: break
    return basis

def dot(u, v): return mp.fsum(x * y for x, y in zip(u, v))

def check(law1d, d, e, label):
    e = [mp.mpf(x) for x in e]; ne = mp.sqrt(dot(e, e)); e = [x / ne for x in e]
    U = gram_schmidt_perp(e); n = d - 1
    a = mp.fsum(w * q ** 2 for q, w in law1d)
    b = mp.fsum(w * q ** 4 for q, w in law1d)
    c = mp.fsum(w * q ** 6 for q, w in law1d)
    k4 = b - 3 * a ** 2; k6 = c - 15 * a * b + 30 * a ** 3
    free = [(p, q) for p in range(n) for q in range(p, n)]
    names = ['F'] + [f'D{i}' for i in range(d)] + [f'He{i}' for i in range(d)] + ['Te'] + [f'A{p}{q}' for p, q in free]
    N = len(names)
    S = mp.zeros(N, N)
    for pts in product(law1d, repeat=d):
        Q = [p[0] for p in pts]; w = mp.fprod(p[1] for p in pts)
        Qe = dot(Q, e); QU = [dot(Q, u) for u in U]
        cosc = [1] + [0] * d + [-Q[i] * Qe for i in range(d)] + [0] + [-QU[p] * QU[q] for p, q in free]
        sinc = [0] + [Q[i] for i in range(d)] + [0] * d + [-Qe ** 3] + [0] * len(free)
        for r in range(N):
            if cosc[r] == 0 and sinc[r] == 0: continue
            for s in range(r, N):
                S[r, s] += w * (cosc[r] * cosc[s] + sinc[r] * sinc[s])
    for r in range(N):
        for s in range(r): S[r, s] = S[s, r]
    iD = list(range(1, 1 + d)); iH = list(range(1 + d, 1 + 2 * d)); iT = 1 + 2 * d
    iA = list(range(2 + 2 * d, N))
    sub = lambda rows, cols: mp.matrix([[S[r, s] for s in cols] for r in rows])
    Z = iD + iH; Y = [iT] + iA
    cond = sub(Y, Y) - sub(Y, Z) * mp.inverse(sub(Z, Z)) * sub(Z, Y)
    # proof closed forms
    s4 = mp.fsum(x ** 4 for x in e); s6 = mp.fsum(x ** 6 for x in e)
    V = 6 * a ** 3 + 9 * a * k4 * s4 + (k6 - k4 ** 2 / a) * s6
    B = mp.matrix(d, d)
    for i in range(d):
        for k in range(d):
            B[i, k] = a ** 2 * ((i == k) + 2 * e[i] * e[k]) + k4 * (i == k) * e[i] ** 2
    R = lambda p, q, i: a ** 2 * (p == q) * e[i] + k4 * U[p][i] * U[q][i] * e[i]
    C = lambda p, q, r, s: a ** 2 * ((p == q) * (r == s) + (p == r) * (q == s) + (p == s) * (q == r)) \
        + k4 * mp.fsum(U[p][i] * U[q][i] * U[r][i] * U[s][i] for i in range(d))
    Bi = mp.inverse(B)
    Gam = lambda p, q, r, s: C(p, q, r, s) - mp.fsum(R(p, q, i) * Bi[i, j] * R(r, s, j) for i in range(d) for j in range(d))
    # V2 M10 decomposition
    beta = b - a ** 2; nu3 = c - b ** 2 / a
    M10 = nu3 * s6 + 9 * a * beta * mp.fsum(e[i] ** 4 * e[j] ** 2 for i in range(d) for j in range(d) if i != j) \
        + 36 * a ** 3 * mp.fsum(e[i] ** 2 * e[j] ** 2 * e[k] ** 2 for i in range(d) for j in range(i + 1, d) for k in range(j + 1, d))
    errs = {}
    scale = max(abs(S[r, r]) for r in range(N))
    errs['B(11)'] = max(abs(S[iH[i], iH[k]] - B[i, k]) for i in range(d) for k in range(d))
    errs['R(12)'] = max(abs(S[iA[t], iH[i]] - R(*free[t], i)) for t in range(len(free)) for i in range(d))
    errs['C'] = max(abs(S[iA[t], iA[u]] - C(*free[t], *free[u])) for t in range(len(free)) for u in range(len(free)))
    errs['V(10)'] = abs(cond[0, 0] - V)
    errs['V=M10'] = abs(V - M10)
    errs['Gamma(13)'] = max(abs(cond[1 + t, 1 + u] - Gam(*free[t], *free[u])) for t in range(len(free)) for u in range(len(free)))
    errs['Te_A_condcov'] = max([abs(cond[0, 1 + t]) for t in range(len(free))] + [0])
    errs['D_H_cov'] = max(abs(S[i, j]) for i in iD for j in iH + iA + [0])
    errs['Te_H_cov'] = max(abs(S[iT, j]) for j in iH + iA + [0])
    errs['Te_D_cov(9)'] = max(abs(S[iT, iD[i]] + 3 * a ** 2 * e[i] + k4 * e[i] ** 3) for i in range(d))
    worst = max(errs.values()) / scale
    G = mp.matrix([[Gam(*free[t], *free[u]) for u in range(len(free))] for t in range(len(free))])
    print(f"{label:34s} d={d} a={mp.nstr(a,6):>10s} k4={mp.nstr(k4,6):>11s} V={mp.nstr(V,6):>10s} "
          f"detB={mp.nstr(mp.det(B),5):>10s} detG={mp.nstr(mp.det(G),5):>10s} max rel err={mp.nstr(worst,3)}")
    return worst

worst = mp.mpf(0)
th = mp.mpf('0.7')
dirs2 = [[mp.cos(th), mp.sin(th)], [1, mp.sqrt(2)], [1, 0]]
dirs3 = [[1, 1, 1], [mp.sin(1) * mp.cos(2), mp.sin(1) * mp.sin(2), mp.cos(1)], [1, mp.sqrt(3), 0]]
for L in [mp.mpf(3), mp.mpf(6), mp.mpf(10)]:
    law = lattice_law(L)
    for e in dirs2:
        worst = max(worst, check(law, 2, e, f"lattice L={mp.nstr(L,3)} K={len(law)//2}"))
law3 = lattice_law(mp.mpf(3))
for e in dirs3:
    worst = max(worst, check(law3, 3, e, f"lattice L=3 K={len(law3)//2}"))
hz = heavy_zero_law()
for e in dirs2: worst = max(worst, check(hz, 2, e, "heavy-zero (k4>0)"))
for e in dirs3: worst = max(worst, check(hz, 3, e, "heavy-zero (k4>0)"))
print("WORST relative error:", mp.nstr(worst, 5), "PASS" if worst < mp.mpf('1e-35') else "FAIL")
