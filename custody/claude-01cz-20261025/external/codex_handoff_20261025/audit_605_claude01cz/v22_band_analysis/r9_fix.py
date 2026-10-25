import sympy as sp, sys
exec(open(sys.argv[1]).read().split('# pin-parallel b=0')[0])
def negs(K):
    K=sp.expand(K); out={}
    for term in sp.Add.make_args(K):
        p=sp.degree(term.as_powers_dict().get(s,0)) if False else None
        e=term.as_coeff_exponent(s)[1]
        if e<0: out[e]=out.get(e,0)+term*s**(-e)
    return {k:sp.factor(v) for k,v in out.items()}
for aa in (1,-1):
    U3,W1,W2,W3=rows(aa,0); W1h=W1+(q**2/24)*U3
    base=(2*aa*W1h-c*W2-t*W3+(eta*c/2)*U3+(eta/(2*aa))*W2)/s
    for nm,K in [('as stated',base),('with -(eta^2/4)U3/s',base-(eta**2/4)*U3/s)]:
        Kx=sp.expand(sp.expand(K).subs(r,q*s))
        ng=negs(Kx); lim=sp.simplify(Kx.subs({q:0,eta:0}).subs(s,0)) if not ng else None
        print('pin a=%+d %s: negative s-powers ->'%(aa,nm), ng, '' if ng else '; limit+(a/6)D4 = %s'%sp.simplify(lim+sp.Rational(aa,6)*D4))
for lv in (1,-1):
    rr=sp.sqrt(c**2+t**2)
    U3,W1,W2,W3=rows(lv*c/rr,lv*t/rr); W1h=W1+(q**2/24)*U3
    K=(W2-(2*lv/rr)*W1h-(eta*c**2/(2*rr**2))*U3-(eta/rr**2)*W1h)/s
    Kx=sp.expand(sp.expand(K).subs(r,q*s)); ng=negs(Kx)
    print('rad lam=%+d negative s-powers ->'%lv, ng)
    if not ng:
        lim=sp.simplify(Kx.subs({q:0,eta:0}).subs(s,0)); print('   limit-(lam/6rho)D4 =',sp.simplify(lim-sp.Rational(lv,6)/rr*D4))
