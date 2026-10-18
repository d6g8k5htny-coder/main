# Symbolic verification of Section 5 identities (8),(9),(10), |v|^2 identity and kappa defs
# for an abstract product (coordinatewise independent, sign-symmetric) spectral law.
import itertools, sys
import sympy as sp

a, b, c = sp.symbols('a b c', positive=True)
k4 = b - 3*a**2
k6 = c - 15*a*b + 30*a**3
mu = {0: sp.Integer(1), 2: a, 4: b, 6: c}

def mom(exps):
    """E prod Q_i^{n_i} for product sign-symmetric law."""
    r = sp.Integer(1)
    for n in exps:
        if n % 2: return sp.Integer(0)
        r *= mu[n]
    return r

ok = True
# --- 1D cumulant definitions: kappa_4, kappa_6 are the 4th/6th cumulants of a symmetric 1D law
t = sp.symbols('t')
m2, m4, m6 = sp.symbols('m2 m4 m6')
# moment generating function series of symmetric law: 1 + m2 t^2/2 + m4 t^4/24 + m6 t^6/720
M = 1 + m2*t**2/2 + m4*t**4/24 + m6*t**6/720
cg = sp.series(sp.log(M), t, 0, 7).removeO()
cum4 = sp.expand(cg.coeff(t, 4)*24); cum6 = sp.expand(cg.coeff(t, 6)*720)
c4ok = sp.simplify(cum4.subs({m2: a, m4: b}) - k4) == 0
c6ok = sp.simplify(cum6.subs({m2: a, m4: b, m6: c}) - k6) == 0
print('kappa_4 = 4th cumulant:', c4ok, ' kappa_6 = 6th cumulant:', c6ok)
ok &= c4ok and c6ok

for d in range(1, 8):
    e = sp.symbols('e1:%d' % (d+1))
    p2 = sum(x**2 for x in e); p4 = sum(x**4 for x in e); p6 = sum(x**6 for x in e)
    # (8): fourth moment tensor
    t8 = True
    for idx in itertools.product(range(d), repeat=4):
        i, j, k, l = idx
        exps = [idx.count(s) for s in range(d)]
        lhs = mom(exps)
        dl = lambda p, q: 1 if p == q else 0
        rhs = a**2*(dl(i,j)*dl(k,l) + dl(i,k)*dl(j,l) + dl(i,l)*dl(j,k)) + k4*sum(dl(i,s)*dl(j,s)*dl(k,s)*dl(l,s) for s in range(d))
        if sp.expand(lhs - rhs) != 0:
            t8 = False
    # Cov(D)=aI
    tD = all(sp.expand(mom([ (1 if s==i else 0)+(1 if s==j else 0) for s in range(d)]) - (a if i==j else 0)) == 0 for i in range(d) for j in range(d))
    # expectation of polynomial in Q via monomial expansion
    Q = sp.symbols('Q1:%d' % (d+1))
    def E(expr):
        P = sp.Poly(sp.expand(expr), *Q)
        return sp.expand(sum(coef*mom(list(mon)) for mon, coef in P.terms()))
    Qe = sum(ei*qi for ei, qi in zip(e, Q))
    # (9) homogeneous form (valid without |e|=1): v_i = 3a^2 e_i p2 + k4 e_i^3
    v = [E(Q[i]*Qe**3) for i in range(d)]
    t9 = all(sp.expand(v[i] - (3*a**2*e[i]*p2 + k4*e[i]**3)) == 0 for i in range(d))
    # sixth moment, homogeneous form: 15 a^3 p2^3 + 15 a k4 p4 p2 + k6 p6
    E6 = E(Qe**6)
    t6 = sp.expand(E6 - (15*a**3*p2**3 + 15*a*k4*p4*p2 + k6*p6)) == 0
    # |v|^2 identity (homog.): 9a^4 p2^3 + 6a^2 k4 p4 p2 + k4^2 p6
    v2 = sp.expand(sum(x**2 for x in v))
    tv = sp.expand(v2 - (9*a**4*p2**3 + 6*a**2*k4*p4*p2 + k4**2*p6)) == 0
    # (10): regression variance = Var(T) - Cov(T,D) Cov(D)^-1 Cov(D,T), Cov(D)=aI, Cov(T,D_i)=-v_i
    covD = sp.diag(*[a]*d)
    cTD = sp.Matrix([[-x for x in v]])
    Vreg = sp.expand(E6 - (cTD*covD.inv()*cTD.T)[0, 0])
    V10 = 6*a**3*p2**3 + 9*a*k4*p4*p2 + (k6 - k4**2/a)*p6
    t10 = sp.simplify(Vreg - V10) == 0
    # coordinate direction e=e_1: c - b^2/a
    tc = sp.simplify(V10.subs({x: 0 for x in e[1:]}).subs(e[0], 1) - (c - b**2/a)) == 0
    # V2 (M10) positive decomposition, expanded with homogeneous weights
    beta = b - a**2
    M10 = (c - b**2/a)*p6 + 9*a*beta*sum(e[i]**4*e[j]**2 for i in range(d) for j in range(d) if i != j)*p2**0 \
          + 36*a**3*sum(e[i]**2*e[j]**2*e[k]**2 for i in range(d) for j in range(i+1, d) for k in range(j+1, d))
    # M10 is homogeneous deg 6 in e; compare on sphere: V10 with p2=1 vs M10 -> check M10 == V10 when homogenized:
    # V10 homog uses p2 powers; M10 is already degree 6, compare exactly as homogeneous polynomials
    tM10 = sp.simplify(sp.expand(M10 - V10)) == 0
    print(f'd={d}: (8)tensor={t8} Cov(D)=aI={tD} (9)={t9} E6={t6} |v|^2={tv} (10)={t10} coord={tc} M10==(10)={tM10}')
    ok &= all([t8, tD, t9, t6, tv, t10, tc, tM10])

# General-d argument via monomial symmetric functions in x_i=e_i^2 (Newton identities, any number of variables):
# E(Q.e)^6 = c m_(3) + 15ab m_(2,1) + 90a^3 m_(1,1,1);  m_(3)=P3, m_(2,1)=P2 P1 - P3, m_(1,1,1)=(P1^3-3P1P2+2P3)/6
P1, P2, P3 = sp.symbols('P1 P2 P3')
gen = c*P3 + 15*a*b*(P2*P1 - P3) + 90*a**3*(P1**3 - 3*P1*P2 + 2*P3)/6
tg = sp.expand(gen - (15*a**3*P1**3 + 15*a*k4*P2*P1 + k6*P3)) == 0
# v_i = b e_i^3 + 3a^2 e_i (P1 - e_i^2)
ei = sp.symbols('ei')
tgv = sp.expand(b*ei**3 + 3*a**2*ei*(P1 - ei**2) - (3*a**2*ei*P1 + k4*ei**3)) == 0
# |v|^2 = sum_i (3a^2 e_i P1 + k4 e_i^3)^2 = 9a^4 P1^2 * P1 + 6 a^2 k4 P1 P2 + k4^2 P3
print('general-d multiplicity identity for E(Q.e)^6:', tg, ' v_i:', tgv)
ok &= tg and tgv
print('ALL SYMBOLIC CHECKS PASS' if ok else 'SOME SYMBOLIC CHECK FAILED')
