import sympy as sp, numpy as np
al,s,a,b,k,A,bp,kp = sp.symbols('alpha s a b k A bp kp', positive=True)
# E[A^(-2/3)] with density alpha a^-alpha s^(alpha-1) on (0,a)
mom = sp.integrate(al*a**(-al)*s**(al-1)*s**sp.Rational(-2,3),(s,0,a), conds='none')
for alv0 in [sp.Rational(3,4), 1, sp.Rational(5,2)]:
    mv = sp.integrate((al*a**(-al)*s**(al-1)*s**sp.Rational(-2,3)).subs(al,alv0),(s,0,a))
    print('alpha=',alv0,' E[A^-2/3] =', sp.simplify(mv), ' claim:', sp.simplify((a**sp.Rational(-2,3)*al/(al-sp.Rational(2,3))).subs(al,alv0)), ' diff', sp.simplify(mv-(a**sp.Rational(-2,3)*al/(al-sp.Rational(2,3))).subs(al,alv0)))
# divergence for alpha<=2/3: exponent alpha-5/3 <= -1
for alv in [0.5, 2/3, 0.8]:
    f = lambda t: alv*t**(alv-1)*t**(-2/3)
    xs = np.logspace(-12,0,200001); print(alv, "numeric tail integral to 1e-12:", np.trapezoid(f(xs),xs))
# substitution: int k^-2/3 G(b/A,k/A) db dk = A^(4/3) int k'^-2/3 G(b',k') db'dk'  (with b=A b', k=A k')
G = sp.exp(-(bp**2+kp**2))*(1+kp)  # arbitrary test envelope-like function
lhs = sp.integrate(sp.integrate(k**sp.Rational(-2,3)*G.subs({bp:b/A,kp:k/A}),(b,-sp.oo,sp.oo)),(k,0,sp.oo))
rhs = A**sp.Rational(4,3)*sp.integrate(sp.integrate(kp**sp.Rational(-2,3)*G,(bp,-sp.oo,sp.oo)),(kp,0,sp.oo))
print("substitution check diff:", sp.simplify(lhs-rhs))
# homogeneity of v_r in (b,k)
r=sp.symbols('r',positive=True); S=sp.symbols('S',positive=True)
vr = lambda b,k: sp.Matrix([b-k*r**3/2, -k*r**2, 0, 12*k])
print("v_r(S b,S k) - S v_r(b,k):", (vr(S*b,S*k)-S*vr(b,k)).T)
# Monte Carlo: mixture identity pi^F(v) E[W|U^F=v] = E[A^-2 (pi^H E[W|U^H=.])(v/A)] for toy d=1:
# U^H ~ N(0,I_4) (n=2d+2=4), W = (U1*U2)^2 (homogeneous deg 2d=2).  Then pi^H(v)E[W|v]=phi(v) v1^2 v2^2.
rng=np.random.default_rng(0); alv=1.5; av=np.sqrt((alv+2)/alv)
Av = av*rng.random(2_000_000)**(1/alv)
v=np.array([0.7,-0.4,0.2,1.1]); n=4
phi=lambda x: np.exp(-np.sum(x**2,axis=-1)/2)/(2*np.pi)**(n/2)
rhs_mc = np.mean(Av**-2 * phi(v[None,:]/Av[:,None]) * (v[0]/Av)**2*(v[1]/Av)**2)
# direct: joint density of U^F = A U^H at v times E[W^F | U^F=v], W^F=A^2 (U1^H U2^H)^2 = (U1^F U2^F)^2 -> value (v1 v2)^2 * pi^F(v)
piF = np.mean(Av**-n * phi(v[None,:]/Av[:,None]))
print("mixture identity MC: lhs", piF*(v[0]*v[1])**2, "rhs", rhs_mc)
