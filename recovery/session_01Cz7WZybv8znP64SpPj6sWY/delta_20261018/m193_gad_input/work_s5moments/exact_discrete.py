# Exact-rational cross-check on explicit finite symmetric PRODUCT spectral measures (random rational
# atoms/weights, including an atom at 0), d=2,3,4. Covariances of derivative jets are computed from the
# EXPLICIT random trigonometric series (5) (cos/sin basis with independent N(0,1) coefficients), i.e.
# Cov(d^al F(0), d^be F(0)) = sum over modes of products of the deterministic derivative coefficients,
# not from the moment formulas being tested. Then the Schur complement Var(T_e|D) is formed exactly.
from fractions import Fraction as Fr
import itertools, random
random.seed(20261017)

def rand_law():
    # symmetric 1D law on {0, +-s1, +-s2, +-s3} with random rational s and positive rational weights
    s = sorted(random.sample(range(1, 40), 3)); s = [Fr(x, random.randint(1, 9)) for x in s]
    w0 = Fr(random.randint(1, 20)); ws = [Fr(random.randint(1, 20)) for _ in s]
    Z = w0 + 2*sum(ws)
    law = {Fr(0): w0/Z}
    for si, wi in zip(s, ws):
        law[si] = law.get(si, 0) + wi/Z; law[-si] = law.get(-si, 0) + wi/Z
    return law

def mom1(law, n): return sum(p*q**n for q, p in law.items())

# derivative of cos(q.x)/sin(q.x) at x=0 for multi-index al with |al|=n: Re/Im of i^n * q^al
def ipow(n): return [(1,0),(0,1),(-1,0),(0,-1)][n % 4]

def jet_cov(atoms, ops):
    """atoms: list of (q-vector, mass) of the full symmetric product measure.
    ops: list of functions q -> (order n, polynomial value P(q)) for linear derivative functionals,
    each functional = sum_al coef * d^al, homogeneous of order n, P(q)=sum coef q^al.
    Field: sqrt(w0) Z0 + sum_{m in H} sqrt(2w_m)[X cos + Y sin]. Pair {q,-q} collapsed into H."""
    n_ops = len(ops)
    C = [[Fr(0)]*n_ops for _ in range(n_ops)]
    seen = set()
    for q, w in atoms:
        if all(x == 0 for x in q):
            # zero mode: only order-0 functionals see it
            for A in range(n_ops):
                for B in range(n_ops):
                    nA, PA = ops[A](q); nB, PB = ops[B](q)
                    if nA == 0 and nB == 0: C[A][B] += w*PA*PB
            continue
        key = tuple(q); neg = tuple(-x for x in q)
        if neg in seen: continue
        seen.add(key)
        for A in range(n_ops):
            nA, PA = ops[A](q); rA, iA = ipow(nA)
            for B in range(n_ops):
                nB, PB = ops[B](q); rB, iB = ipow(nB)
                # cos coefficient: Re(i^n) P(q), sin coefficient: Im(i^n) P(q); variance weight 2w
                C[A][B] += 2*w*(rA*rB + iA*iB)*PA*PB
    return C

def solve(M, rhs):
    n = len(M); A = [row[:] + [rhs[i]] for i, row in enumerate(M)]
    for col in range(n):
        piv = next(r for r in range(col, n) if A[r][col] != 0); A[col], A[piv] = A[piv], A[col]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]/A[col][col]; A[r] = [x - f*y for x, y in zip(A[r], A[col])]
    return [A[i][n]/A[i][i] for i in range(n)]

unit_vecs = {
    2: [(Fr(3,5), Fr(4,5)), (Fr(-5,13), Fr(12,13)), (Fr(1), Fr(0)), (Fr(8,17), Fr(-15,17))],
    3: [(Fr(2,3), Fr(2,3), Fr(1,3)), (Fr(2,7), Fr(3,7), Fr(-6,7)), (Fr(0), Fr(1), Fr(0)), (Fr(1,9), Fr(4,9), Fr(8,9))],
    4: [(Fr(1,2),)*4, (Fr(1,2), Fr(-1,2), Fr(1,2), Fr(1,2)), (Fr(2,5), Fr(2,5), Fr(4,5), Fr(1,5)), (Fr(0), Fr(0), Fr(0), Fr(1))],
}
allok = True
for d in (2, 3, 4):
    for trial in range(2):
        law = rand_law()
        a, b, c = mom1(law, 2), mom1(law, 4), mom1(law, 6)
        k4 = b - 3*a*a; k6 = c - 15*a*b + 30*a**3
        atoms = []
        for combo in itertools.product(law.items(), repeat=d):
            q = tuple(x[0] for x in combo); w = Fr(1)
            for x in combo: w *= x[1]
            atoms.append((q, w))
        assert sum(w for _, w in atoms) == 1
        # (8): Hessian covariance from explicit series vs formula
        hess_idx = [(i, j) for i in range(d) for j in range(d)]
        ops = [ (lambda i,j: (lambda q: (2, q[i]*q[j])))(i, j) for (i, j) in hess_idx ]
        CH = jet_cov(atoms, ops)
        dl = lambda p, r: 1 if p == r else 0
        ok8 = True
        for A, (i, j) in enumerate(hess_idx):
            for B, (k, l) in enumerate(hess_idx):
                rhs = a*a*(dl(i,j)*dl(k,l) + dl(i,k)*dl(j,l) + dl(i,l)*dl(j,k)) + k4*sum(dl(i,s)*dl(j,s)*dl(k,s)*dl(l,s) for s in range(d))
                if CH[A][B] != rhs: ok8 = False
        for e in unit_vecs[d]:
            assert sum(x*x for x in e) == 1
            # functionals: D_1..D_d (order 1), T_e (order 3), Hessian entries (order 2), and F itself (order 0)
            ops = [ (lambda i: (lambda q: (1, q[i])))(i) for i in range(d) ]
            ops.append(lambda q, e=e: (3, sum(x*y for x, y in zip(e, q))**3))
            ops += [ (lambda i,j: (lambda q: (2, q[i]*q[j])))(i, j) for (i, j) in hess_idx ]
            ops.append(lambda q: (0, Fr(1)))
            C = jet_cov(atoms, ops)
            nD = d; iT = d; iH0 = d+1; iF = len(ops)-1
            okD = all(C[i][j] == (a if i == j else 0) for i in range(d) for j in range(d))
            v = [a*0 + 3*a*a*e[i] + k4*e[i]**3 for i in range(d)]
            ok9 = all(C[iT][i] == -v[i] for i in range(d))
            okodd = all(C[x][y] == 0 for x in list(range(d)) + [iT] for y in range(iH0, iH0 + d*d))
            p4 = sum(x**4 for x in e); p6 = sum(x**6 for x in e)
            ok6 = C[iT][iT] == 15*a**3 + 15*a*k4*p4 + k6*p6
            okv2 = sum(x*x for x in v) == 9*a**4 + 6*a*a*k4*p4 + k4*k4*p6
            # exact Schur complement Var(T_e | D=0)
            covD = [[C[i][j] for j in range(d)] for i in range(d)]
            cTD = [C[iT][i] for i in range(d)]
            Vreg = C[iT][iT] - sum(x*y for x, y in zip(cTD, solve(covD, cTD)))
            V10 = 6*a**3 + 9*a*k4*p4 + (k6 - k4*k4/a)*p6
            ok10 = (Vreg == V10) and Vreg > 0
            # zero mode: Var F = 1, Cov(F, D)=0, Cov(F,H_ii) = -a (height correlated with Hessian), Cov(F,T)=0
            okF = C[iF][iF] == 1 and all(C[iF][i] == 0 for i in range(d)) and C[iF][iT] == 0 \
                  and all(C[iF][iH0 + i*d + i] == -a for i in range(d))
            row = all([ok8, okD, ok9, okodd, ok6, okv2, ok10, okF])
            allok &= row
            print(f'd={d} trial={trial} e={tuple(str(x) for x in e)}: (8)={ok8} CovD={okD} (9)={ok9} odd-indep={okodd} '
                  f'E6={ok6} |v|^2={okv2} (10)={ok10} [V={float(Vreg):.6g}] F-row={okF}')
print('ALL EXACT DISCRETE CHECKS PASS' if allok else 'FAILURE')
