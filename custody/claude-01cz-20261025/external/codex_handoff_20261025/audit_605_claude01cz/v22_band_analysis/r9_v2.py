import sympy as sp, sys
exec(open(sys.argv[1]).read().split('# pin-parallel b=0')[0])
def negs(K):
    out={}
    for term in sp.Add.make_args(sp.expand(K)):
        e=term.as_coeff_exponent(s)[1]
        if e<0: out[e]=out.get(e,0)+term*s**(-e)
    return {k:sp.factor(v) for k,v in out.items() if sp.simplify(v)!=0}
for aa in (1,-1):
    U3,W1,W2,W3=rows(aa,0); W1h=W1+(q**2/24)*U3
    Q=(2*aa*W1h+(-c+eta/(2*aa))*W2-t*W3+(c*eta/2-eta**2/(4*aa))*U3)/s
    Qx=sp.expand(sp.expand(Q).subs(r,q*s)); ng=negs(Qx)
    lim=sp.simplify(Qx.subs({q:0,eta:0}).subs(s,0))
    print('v2 pin a=%+d: residual negative s-powers'%aa, ng, '; limit+(a/6)Kz =', sp.simplify(lim+sp.Rational(aa,6)*D4))
for lv in (1,-1):
    rr=sp.sqrt(c**2+t**2); A=lv*c/rr; Bb=lv*t/rr
    U3,W1,W2,W3=rows(A,Bb); W1h=W1+(q**2/24)*U3
    kap=2*Bb/t+eta*Bb**2/t**2
    Q=(W2-kap*W1h-(eta*A**2/2)*U3)/s
    Qx=sp.expand(sp.expand(Q).subs(r,q*s)); ng=negs(Qx)
    lim=sp.simplify(Qx.subs({q:0,eta:0}).subs(s,0))
    print('v2 rad lam=%+d: residual negative s-powers'%lv, ng, '; limit-(b/6t)Kz =', sp.simplify(lim-Bb/(6*t)*D4))
# determinants of row replacement (pin: W3 -> Q ; rad: W2 -> Q): coefficient of W3 in Q is -t/s ; of W2 in Qrad is 1/s
print('pin replacement factor |coef of W3| = |t|/s ; rad replacement factor coef of W2 = 1/s  (by construction)')
print('pin density power: r^-5 eta^-2 |t|^-1 s^-7 * |t|/s = r^-5 eta^-2 s^-8 -> e^-2 s^-6 ; rad: *1/s -> e^-2 |t|^-1 s^-6')
