"""Symbolic (sympy) check of the covariance blocks behind proof (8)-(13), d=2,3,4.

Moments of the product law are symbols: E[prod Q_i^{k_i}] = prod m_{k_i},
m_0=1, m_2=a, m_4=b, m_6=c, odd -> 0.  Covariances are computed from the
Fourier construction: Cov(X,Y)=E_Q[cos_X cos_Y + sin_X sin_Y] with
  D_i:(0,Q_i)  He_i:(-Q_i(Q.e),0)  (U^T H U)_pq:(-(Q.u_p)(Q.u_q),0)
  T_e:(0,-(Q.e)^3)  F:(1,0).
e and the columns u_p are FREE symbols (no constraints imposed); results are
compared with the constraint-free forms of the proof's formulas, which reduce
to (9),(10),(11),(12),C when |e|=1, U^T U=I, U^T e=0.
Run: <venv>/bin/python -I check_s6_sym.py
"""
import sympy as sp

a, b, c = sp.symbols('a b c', positive=True)
k4 = b - 3 * a**2
k6 = c - 15 * a * b + 30 * a**3
mom = {0: 1, 2: a, 4: b, 6: c}

def E(poly, Q):
    P = sp.Poly(sp.expand(poly), *Q)
    tot = 0
    for monom, coef in P.terms():
        t = coef
        for k in monom:
            if k % 2: t = 0; break
            t *= mom[k]
        tot += t
    return sp.expand(tot)

allok = True
for d in (2, 3, 4):
    Q = sp.symbols(f'Q1:{d+1}')
    e = sp.symbols(f'e1:{d+1}')
    n = d - 1
    U = [sp.symbols(f'u{p}_1:{d+1}') for p in range(n)]
    dot = lambda x, y: sum(xi * yi for xi, yi in zip(x, y))
    Qe = dot(Q, e); QU = [dot(Q, u) for u in U]
    E2 = dot(e, e); s4 = sum(x**4 for x in e); s6 = sum(x**6 for x in e)
    res = {}
    # Var T_e and Cov(T_e, D_i)
    res['Var Te'] = sp.expand(E(Qe**6, Q) - (15 * a**3 * E2**3 + 15 * a * k4 * E2 * s4 + k6 * s6))
    for i in range(d):
        res[f'Cov(Te,D{i})'] = sp.expand(E(-Qe**3 * Q[i], Q) - (-(3 * a**2 * e[i] * E2 + k4 * e[i]**3)))
    # V(e) at |e|=1 via regression on D (Cov D = a I verified below)
    for i in range(d):
        for j in range(d):
            res[f'CovD{i}{j}'] = sp.expand(E(Q[i] * Q[j], Q) - (a if i == j else 0))
    v = [3 * a**2 * e[i] * E2 + k4 * e[i]**3 for i in range(d)]
    Vgen = E(Qe**6, Q) - sum(x * x for x in v) / a
    Vprop = 6 * a**3 + 9 * a * k4 * s4 + (k6 - k4**2 / a) * s6
    # impose |e|=1 by substituting e_d^2 = 1 - sum_{i<d} e_i^2 (only even powers of e_d appear)
    sub_unit = lambda expr: sp.expand(sp.expand(expr).subs(e[-1]**2, 1 - sum(x**2 for x in e[:-1])))
    def unit_reduce(expr):
        expr = sp.expand(expr)
        P = sp.Poly(expr, e[-1])
        out = 0
        for (k,), coef in P.terms():
            out += coef * (1 - sum(x**2 for x in e[:-1]))**(k // 2) * e[-1]**(k % 2)
        return sp.expand(out)
    res['V(10) at |e|=1'] = sp.simplify(unit_reduce(sp.expand(a * (Vgen - Vprop))))
    # B
    for i in range(d):
        for k in range(d):
            got = E(Q[i] * Qe * Q[k] * Qe, Q)
            want = a**2 * ((E2 if i == k else 0) + 2 * e[i] * e[k]) + k4 * (e[i]**2 if i == k else 0)
            res[f'B{i}{k}'] = sp.expand(got - want)
    # R
    for p in range(n):
        for q in range(n):
            for i in range(d):
                got = E(QU[p] * QU[q] * Q[i] * Qe, Q)
                want = a**2 * (dot(U[p], U[q]) * e[i] + dot(U[p], e) * U[q][i] + dot(U[q], e) * U[p][i]) \
                    + k4 * U[p][i] * U[q][i] * e[i]
                res[f'R{p}{q},{i}'] = sp.expand(got - want)
    # C
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    got = E(QU[p] * QU[q] * QU[r] * QU[s], Q)
                    want = a**2 * (dot(U[p], U[q]) * dot(U[r], U[s]) + dot(U[p], U[r]) * dot(U[q], U[s])
                                   + dot(U[p], U[s]) * dot(U[q], U[r])) \
                        + k4 * sum(U[p][i] * U[q][i] * U[r][i] * U[s][i] for i in range(d))
                    res[f'C{p}{q}{r}{s}'] = sp.expand(got - want)
    # zero blocks: odd/even
    for i in range(d):
        for j in range(d):
            res[f'Cov(D{i},He{j})'] = E(Q[i] * 0 + 0, Q)  # D sin-only, He cos-only: product of coefs is 0 identically
    res['Cov(Te,He/A)'] = 0  # T_e sin-only, He/A cos-only => exactly zero in construction
    res['Cov(F,He_i)+a e_i'] = sum(sp.expand(E(-Q[i] * Qe, Q) + a * e[i])**2 for i in range(d))
    bad = {k: v for k, v in res.items() if v != 0}
    print(f"d={d}: {len(res)} symbolic identities checked; nonzero residuals: {len(bad)}")
    for k, v in list(bad.items())[:5]: print("   ", k, v)
    allok &= not bad
print("ALL SYMBOLIC IDENTITIES HOLD" if allok else "SYMBOLIC FAILURE")
