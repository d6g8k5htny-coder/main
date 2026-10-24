import sympy as sp
a,h,th,s,u,v,tau,t,r = sp.symbols('a h theta s u v tau t r', positive=True)
P = lambda x,u,v: -x**4/4 + u*x**2/2 + v*x
U = 3*a**2+h**2; V = 2*a*(a**2-h**2)
print('stationary:', [sp.simplify(-x**3+U*x+V) for x in (-a-h,-a+h,2*a)])
gap = sp.simplify(P(-a-h,U,V)-P(-a+h,U,V)); print('gap P(-a-h)-P(-a+h) =', sp.factor(gap), ' == 4ah^3:', sp.simplify(gap-4*a*h**3)==0)
outer = sp.simplify(P(2*a,U,V)-P(-a-h,U,V)); print('P(2a)-P(-a-h) =', sp.factor(outer), ' == a^4(1-th)(3+th)^3/4 with h=a th:', sp.simplify(outer.subs(h,a*th) - a**4*(1-th)*(3+th)**3/4)==0)
# adversarial theta=2
A2 = {h:2*a}
hts=[sp.simplify(P(x,U,V).subs(A2)) for x in (-3*a,a,2*a)]; print('theta=2 heights/a^4:', [sp.simplify(x/a**4) for x in hts], 'younger-right gap', sp.simplify((hts[2]-hts[1])/a**4), 'fixed-left', sp.simplify((hts[0]-hts[1])/a**4))
# core monotonicity numerator
f = th*(9-th**2)*(3+th**2)**sp.Rational(-5,2)
num = sp.simplify(sp.diff(f,th)*(3+th**2)**sp.Rational(7,2)); print('core derivative numerator:', sp.expand(num))
# tail second derivative
C = sp.symbols('C', positive=True); rh = sp.symbols('rhi', positive=True)
g = C*(9*t**sp.Rational(-11,4) - rh**2*t**sp.Rational(-3,4))
print('tail f\'\':', sp.simplify(sp.diff(g,t,2)/C))
# closed-form CDF vs direct integral: d/dtau closed form should equal density (1/15) * int over theta>s of theta(9-theta^2) d/dtau (tau/(4 th^3))^(5/4)
# verify instead symbolically: direct(tau) = (1/15)[ int_0^s th(9-th^2)(3/(3+th^2))^(5/2) dth + (tau/4)^(5/4) int_s^1 th(9-th^2) th^(-15/4) dth ]
I1 = sp.integrate(th*(9-th**2)*(3/(3+th**2))**sp.Rational(5,2),(th,0,s))
I2 = sp.integrate((9-th**2)*th**sp.Rational(-11,4),(th,s,1))
direct = (I1 + (tau/4)**sp.Rational(5,4)*I2)/15
closed = (3 - 3*(1-s**2)/(1+s**2/3)**sp.Rational(3,2) + (tau/4)**sp.Rational(5,4)*(sp.Rational(36,7)*(s**sp.Rational(-7,4)-1) - 4*(1-s**sp.Rational(1,4))))/15
d = sp.simplify(direct-closed)
print('direct - closed (symbolic, s free):', d)
import random
for _ in range(3):
    sv = sp.Rational(random.randint(1,99),100); tv = sp.Rational(random.randint(1,200),100)
    print('  numeric diff at s=%s tau=%s:'%(sv,tv), sp.N((direct-closed).subs({s:sv,tau:tv}),30))
print('total mass I1(s=1)/15 =', sp.nsimplify(sp.simplify(I1.subs(s,1)/15)))
# measure normalization: (1/3) int z^4 th(9-th^2) over z<sqrt(3/(3+th^2)), th in (0,1)
tot = sp.integrate(sp.Rational(1,3)*th*(9-th**2)*sp.integrate(sp.Symbol('z')**4,(sp.Symbol('z'),0,sp.sqrt(3/(3+th**2)))),(th,0,1))
print('rectangle finite-bar mass =', sp.simplify(tot), float(tot))
# asymptotic coefficient: m(tau)/tau^(2/3) -> B ?
Bv = sp.Rational(9,28)/sp.cbrt(2); print('B =', Bv, sp.N(Bv,30))
