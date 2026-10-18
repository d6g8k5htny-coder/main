"""Independent exact-rational check of proof Sec.5-6, eqs (8)-(14).

Construction (independent of the proof's closed forms):
  finite symmetric 1D law mu on {0, +-q_1, ..., +-q_K} with random rational
  masses; Q=(Q_1..Q_d) iid mu (product law, hyperoctahedral invariant).
  Field  F(x) = sum_{Q in supp} sqrt(w_Q) [X_Q cos(Q.x) + Y_Q sin(Q.x)],
  X_Q,Y_Q iid N(0,1)  =>  Cov(F(x),F(y)) = E cos(Q.(x-y)).
  A derivative d^alpha F(0) is the linear functional with coefficients
  Re(i^|alpha| Q^alpha) on X_Q and Im(i^|alpha| Q^alpha) on Y_Q.
  Covariances are weighted inner products of coefficient vectors (exact).
Directional variables (He, U^T H U, T_e) are formed as linear combinations
of the index-level coefficient vectors D_i, H_ij, T_ijk, so the contraction
formulas (11),(12) and the Schur complement (13) are tested, not assumed.

Run: python3 -I check_s6_exact.py
"""
from fractions import Fraction as Fr
from itertools import product, combinations_with_replacement
import random, sys

random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 20261017)

# ---------- exact linear algebra ----------
def mat_inv(M):
    n = len(M)
    A = [list(map(Fr, row)) + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [row[n:] for row in A]

def det(M):
    n = len(M); A = [list(map(Fr, r)) for r in M]; s = Fr(1)
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None: return Fr(0)
        if p != c: A[c], A[p] = A[p], A[c]; s = -s
        s *= A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return s

# ---------- random finite symmetric 1D law ----------
def random_law(K=3):
    qs = sorted(set(Fr(random.randint(1, 9), random.randint(1, 4)) for _ in range(K)))
    raw = [random.randint(1, 9) for _ in range(len(qs) + 1)]
    tot = sum(raw)
    law = {Fr(0): Fr(raw[0], tot)}
    for q, r in zip(qs, raw[1:]):
        law[q] = Fr(r, 2 * tot); law[-q] = Fr(r, 2 * tot)
    return law

def moments(law):
    a = sum(w * q**2 for q, w in law.items())
    b = sum(w * q**4 for q, w in law.items())
    c = sum(w * q**6 for q, w in law.items())
    return a, b, c

# ---------- coefficient vectors ----------
I_POW = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # i^k = (re, im)

def coeff(Q, alpha):
    """(cos-coef, sin-coef) of d^alpha F(0) at spectral point Q."""
    mono = Fr(1)
    for idx in alpha: mono *= Q[idx]
    re, im = I_POW[len(alpha) % 4]
    return (re * mono, im * mono)

class Field:
    def __init__(self, law, d):
        self.d = d
        self.pts = [(tuple(qs), _prod(law[q] for q in qs)) for qs in product(law.keys(), repeat=d)]
        assert sum(w for _, w in self.pts) == 1
    def var(self, alpha):
        return [coeff(Q, alpha) for Q, _ in self.pts]
    def cov(self, u, v):
        return sum(w * (cu[0] * cv[0] + cu[1] * cv[1]) for (Q, w), cu, cv in zip(self.pts, u, v))

def _prod(it):
    p = Fr(1)
    for x in it: p *= x
    return p

def lin(coefs_vecs):
    """linear combination sum c * vec of coefficient vectors"""
    out = None
    for c, v in coefs_vecs:
        if c == 0: continue
        if out is None:
            out = [(c * x, c * y) for x, y in v]
        else:
            out = [(o[0] + c * x, o[1] + c * y) for o, (x, y) in zip(out, v)]
    if out is None:
        out = [(Fr(0), Fr(0))] * len(coefs_vecs[0][1])
    return out

def delta(i, j): return 1 if i == j else 0

# ---------- the check ----------
def check(law, d, e, Ucols, label, report):
    a, b, c = moments(law)
    k4 = b - 3 * a**2
    k6 = c - 15 * a * b + 30 * a**3
    fld = Field(law, d)
    e = [Fr(x) for x in e]; U = [[Fr(x) for x in col] for col in Ucols]  # U[a] = column a
    n = d - 1
    assert sum(x * x for x in e) == 1
    for p in range(n):
        assert sum(U[p][i] * e[i] for i in range(d)) == 0
        for q in range(n):
            assert sum(U[p][i] * U[q][i] for i in range(d)) == delta(p, q)
    Fv = fld.var(())
    Dv = [fld.var((i,)) for i in range(d)]
    Hv = {(i, j): fld.var((i, j)) for i in range(d) for j in range(d)}
    T3 = {t: fld.var(t) for t in product(range(d), repeat=3)}
    Te = lin([(e[i] * e[j] * e[k], T3[(i, j, k)]) for (i, j, k) in T3])
    He = [lin([(e[j], Hv[(i, j)]) for j in range(d)]) for i in range(d)]
    UHU = {(p, q): lin([(U[p][i] * U[q][j], Hv[(i, j)]) for i in range(d) for j in range(d)])
           for p in range(n) for q in range(n)}
    free = [(p, q) for p in range(n) for q in range(p, n)]
    cov = fld.cov
    fails = []
    def chk(name, got, want):
        if got != want: fails.append((name, got, want))
    # (8) gradient and Hessian covariance
    for i in range(d):
        for j in range(d):
            chk(f"CovD[{i}{j}]", cov(Dv[i], Dv[j]), a * delta(i, j))
    for (i, j) in Hv:
        for (k, l) in Hv:
            want = a**2 * (delta(i, j) * delta(k, l) + delta(i, k) * delta(j, l) + delta(i, l) * delta(j, k)) \
                + k4 * sum(delta(i, s) * delta(j, s) * delta(k, s) * delta(l, s) for s in range(d))
            chk(f"CovH[{i}{j},{k}{l}]", cov(Hv[(i, j)], Hv[(k, l)]), want)
    # odd/even block independence (D, T_e, all T_ijk) vs (F, H)
    for (i, j) in Hv:
        for k in range(d): chk(f"Cov(D{k},H{i}{j})", cov(Dv[k], Hv[(i, j)]), 0)
        chk(f"Cov(Te,H{i}{j})", cov(Te, Hv[(i, j)]), 0)
    chk("Cov(Te,F)", cov(Te, Fv), 0)
    for k in range(d): chk(f"Cov(D{k},F)", cov(Dv[k], Fv), 0)
    # height IS correlated with Hessian: Cov(F,H_ij) = -a delta_ij
    for (i, j) in Hv: chk(f"Cov(F,H{i}{j})", cov(Fv, Hv[(i, j)]), -a * delta(i, j))
    # (9)
    v = [3 * a**2 * e[i] + k4 * e[i]**3 for i in range(d)]
    for i in range(d): chk(f"Cov(Te,D{i})", cov(Te, Dv[i]), -v[i])
    s4 = sum(x**4 for x in e); s6 = sum(x**6 for x in e)
    chk("Var(Te)", cov(Te, Te), 15 * a**3 + 15 * a * k4 * s4 + k6 * s6)
    chk("|v|^2", sum(x * x for x in v), 9 * a**4 + 6 * a**2 * k4 * s4 + k4**2 * s6)
    # (11) B(e)
    Bf = [[a**2 * (delta(i, k) + 2 * e[i] * e[k]) + k4 * delta(i, k) * e[i]**2 for k in range(d)] for i in range(d)]
    for i in range(d):
        for k in range(d): chk(f"B[{i}{k}]", cov(He[i], He[k]), Bf[i][k])
    # (12) R
    Rf = {(p, q, i): a**2 * delta(p, q) * e[i] + k4 * U[p][i] * U[q][i] * e[i]
          for p in range(n) for q in range(n) for i in range(d)}
    for (p, q, i), want in Rf.items(): chk(f"R[{p}{q},{i}]", cov(UHU[(p, q)], He[i]), want)
    # C
    Cf = {}
    for (p, q) in UHU:
        for (r, s) in UHU:
            Cf[(p, q, r, s)] = a**2 * (delta(p, q) * delta(r, s) + delta(p, r) * delta(q, s) + delta(p, s) * delta(q, r)) \
                + k4 * sum(U[p][i] * U[q][i] * U[r][i] * U[s][i] for i in range(d))
            chk(f"C[{p}{q},{r}{s}]", cov(UHU[(p, q)], UHU[(r, s)]), Cf[(p, q, r, s)])
    # (13) proof's Gamma_e from its closed forms
    Binv = mat_inv(Bf)
    Gam = {(p, q, r, s): Cf[(p, q, r, s)] - sum(Rf[(p, q, i)] * Binv[i][j] * Rf[(r, s, j)]
                                              for i in range(d) for j in range(d))
           for (p, q) in UHU for (r, s) in UHU}
    # (10) proof's V(e)
    Vf = 6 * a**3 + 9 * a * k4 * s4 + (k6 - k4**2 / a) * s6
    # Independent: condition explicit Gaussian vector Y=(T_e, s(UHU)) on Z=(D, He)=0
    Y = [Te] + [UHU[pq] for pq in free]
    Z = Dv + He
    SZZ = [[cov(z1, z2) for z2 in Z] for z1 in Z]
    dZ = det(SZZ)
    SZZi = mat_inv(SZZ)
    SYZ = [[cov(y, z) for z in Z] for y in Y]
    SYY = [[cov(y1, y2) for y2 in Y] for y1 in Y]
    m = len(Y); nz = len(Z)
    Scond = [[SYY[r][s] - sum(SYZ[r][i] * SZZi[i][j] * SYZ[s][j] for i in range(nz) for j in range(nz))
              for s in range(m)] for r in range(m)]
    chk("Var(Te|D=0,He=0) vs V(e) (10)", Scond[0][0], Vf)
    # T_e | D=0 alone
    SDD = [[cov(x, y) for y in Dv] for x in Dv]; SDDi = mat_inv(SDD)
    vTD = [cov(Te, x) for x in Dv]
    chk("Var(Te|D=0) vs V(e)", cov(Te, Te) - sum(vTD[i] * SDDi[i][j] * vTD[j] for i in range(d) for j in range(d)), Vf)
    for r in range(1, m): chk(f"condCov(Te,A{free[r-1]})", Scond[0][r], 0)
    for r in range(1, m):
        for s in range(1, m):
            p, q = free[r - 1]; rr, ss = free[s - 1]
            chk(f"Gamma[{p}{q},{rr}{ss}]", Scond[r][s], Gam[(p, q, rr, ss)])
    # A_e | He=0 alone (without D) equals Gamma too
    SHH = [[cov(x, y) for y in He] for x in He]; SHHi = mat_inv(SHH)
    for r in range(1, m):
        for s in range(1, m):
            p, q = free[r - 1]; rr, ss = free[s - 1]
            g = SYY[r][s] - sum(SYZ[r][d + i] * SHHi[i][j] * SYZ[s][d + j] for i in range(d) for j in range(d))
            chk(f"Gamma|He-only[{p}{q},{rr}{ss}]", g, Gam[(p, q, rr, ss)])
    # positivity / nondegeneracy
    GamFree = [[Gam[(*free[r], *free[s])] for s in range(len(free))] for r in range(len(free))]
    dG = det(GamFree); dB = det(Bf)
    pos = (Vf > 0) and (dB > 0) and (dG > 0) and (dZ > 0)
    # extra: conditioning also on F=0 changes Gamma (supports 'would generally alter')
    Z2 = Z + [Fv]
    S2 = [[cov(z1, z2) for z2 in Z2] for z1 in Z2]; S2i = mat_inv(S2)
    SYZ2 = [[cov(y, z) for z in Z2] for y in Y]
    Gh = [[SYY[r][s] - sum(SYZ2[r][i] * S2i[i][j] * SYZ2[s][j] for i in range(len(Z2)) for j in range(len(Z2)))
           for s in range(1, m)] for r in range(1, m)]
    height_changes = any(Gh[r][s] != GamFree[r][s] for r in range(len(free)) for s in range(len(free)))
    ok = not fails and pos
    report.append((label, d, [str(x) for x in e], ok, len(fails), pos, height_changes,
                   float(a), float(k4), float(k6), float(Vf), float(dB), float(dG)))
    if fails:
        for f in fails[:10]: print("   FAIL", label, f)
    return ok

def main():
    report = []
    cases = []
    r3 = [[1, 2, 2], [2, 1, -2], [2, -2, 1]]
    r7 = [[2, 3, 6], [3, -6, 2], [6, 2, -3]]
    h4 = [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]
    for trial in range(3):
        law = random_law(3)
        # d=2
        cases.append((law, 2, [1, 0], [[0, 1]], f"t{trial} d2 axis"))
        cases.append((law, 2, [Fr(3, 5), Fr(4, 5)], [[Fr(-4, 5), Fr(3, 5)]], f"t{trial} d2 (3,4)/5"))
        cases.append((law, 2, [Fr(5, 13), Fr(-12, 13)], [[Fr(12, 13), Fr(5, 13)]], f"t{trial} d2 (5,-12)/13"))
        # d=3
        cases.append((law, 3, [1, 0, 0], [[0, 1, 0], [0, 0, 1]], f"t{trial} d3 axis"))
        cases.append((law, 3, [Fr(x, 3) for x in r3[0]], [[Fr(x, 3) for x in r3[1]], [Fr(x, 3) for x in r3[2]]], f"t{trial} d3 (1,2,2)/3"))
        cases.append((law, 3, [Fr(x, 7) for x in r7[0]], [[Fr(x, 7) for x in r7[1]], [Fr(x, 7) for x in r7[2]]], f"t{trial} d3 (2,3,6)/7"))
        # d=3, rotated transverse frame (frame dependence check of Gamma tensor done separately)
        u1 = [Fr(x, 7) for x in r7[1]]; u2 = [Fr(x, 7) for x in r7[2]]
        c_, s_ = Fr(3, 5), Fr(4, 5)
        w1 = [c_ * x + s_ * y for x, y in zip(u1, u2)]; w2 = [-s_ * x + c_ * y for x, y in zip(u1, u2)]
        cases.append((law, 3, [Fr(x, 7) for x in r7[0]], [w1, w2], f"t{trial} d3 (2,3,6)/7 rotU"))
    law4 = random_law(2)
    cases.append((law4, 4, [Fr(1, 2)] * 4, [[Fr(x, 2) for x in h4[k]] for k in (1, 2, 3)], "d4 (1,1,1,1)/2"))
    cases.append((law4, 4, [Fr(x, 7) for x in r7[0]] + [0], [[Fr(x, 7) for x in r7[1]] + [0], [Fr(x, 7) for x in r7[2]] + [0], [0, 0, 0, 1]], "d4 (2,3,6,0)/7"))
    allok = True
    for law, d, e, U, lab in cases:
        ok = check(law, d, e, U, lab, report)
        allok &= ok
    print(f"{'case':28s} d  ok   nfail pos  F=0-cond-changes-Gamma  a       k4       k6       V       detB      detGamma")
    for (lab, d, e, ok, nf, pos, hc, a, k4, k6, V, dB, dG) in report:
        print(f"{lab:28s} {d}  {ok!s:5s} {nf:3d}  {pos!s:5s} {hc!s:5s}  {a:8.4g} {k4:8.4g} {k6:8.4g} {V:9.4g} {dB:9.4g} {dG:9.4g}")
    print("ALL EXACT CHECKS PASS" if allok else "SOME CHECK FAILED")

if __name__ == "__main__":
    main()
