# Hp bookkeeping checks for the NG proof's p >= 1 steps (sympy):
# the axial profile between exact pins with F_sss = 12k, the endpoint axial Schur limits, and the index/degree table
# j = d-p-1, index_F(M) = j+1, index_F(S) = j, g-degrees p and p+1, positives(A_0) = d-1-j = p.
import sympy as sp
s, r, k = sp.symbols('s r k', positive=True)
fp = 6*k*(s**2 - r**2/4)           # F' along the axis with pins at s = -r/2, r/2 and F''' = 12k
print('F_sss =', sp.diff(fp, s, 2))
print('F(M)-F(S) =', sp.simplify(-sp.integrate(fp, (s, -r/2, r/2))))
print('alpha_M =', sp.simplify(sp.diff(fp, s).subs(s, -r/2)/r), ' alpha_S =', sp.simplify(sp.diff(fp, s).subs(s, r/2)/r))
# block identity (17), symbolic, general (d-1)x(d-1) A for d-1 = 2 and 3
for m in (2, 3):
    A = sp.Matrix(m, m, lambda i, j: sp.Symbol('a%d%d' % (min(i, j), max(i, j))))
    beta = sp.Matrix(m, 1, lambda i, j: sp.Symbol('b%d' % i)); al = sp.Symbol('alpha')
    H = sp.Matrix(sp.BlockMatrix([[sp.Matrix([[r*al]]), r*beta.T], [r*beta, A]]))
    print('(17) d=%d:' % (m+1), sp.simplify(H.det() - r*(al*A.det() - r*(beta.T*A.adjugate()*beta)[0])) == 0)
ok = True
for d in range(2, 7):
    for p in range(d):
        j = d-p-1; indM, indS = j+1, j; degM, degS = d-indM, d-indS
        ok &= (degM == p and degS == p+1 and d-1-j == p)
print('index/degree table consistent for d = 2..6:', ok)
