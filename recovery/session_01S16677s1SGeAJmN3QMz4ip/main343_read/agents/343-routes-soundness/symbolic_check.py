import sympy as sp
a,h,z,th,s,tau,A = sp.symbols('a h z theta s tau A', positive=True)
# --- chart: roots, u, v, stationarity, lifetime, outer-height identity
u = 3*a**2 + h**2; v = 2*a*(a**2 - h**2)
P = lambda x: -x**4/4 + u*x**2/2 + v*x
cub = lambda x: -x**3 + u*x + v
print('stationary roots:', [sp.simplify(cub(r)) for r in (-a-h, -a+h, 2*a)])
print('gap P(-a-h)-P(-a+h) - 4ah^3 =', sp.simplify(P(-a-h)-P(-a+h) - 4*a*h**3))
print('P(2a)-P(-a-h) - a^4(1-th)(3+th)^3/4 =', sp.simplify((P(2*a)-P(-a-h)).subs(h, a*th) - a**4*(1-th)*(3+th)**3/4))
# --- Jacobian of (z,theta) -> (u,v) with a = A z, h = a theta; c0 = 1/(24 A^5); two signs
U = A**2*z**2*(3+th**2); V = 2*A**3*z**3*(1-th**2)
J = sp.Matrix([[sp.diff(U,z), sp.diff(U,th)],[sp.diff(V,z), sp.diff(V,th)]]).det()
dens_two_signs = sp.simplify(2*sp.Abs(J)/(24*A**5))
print('two-sign control density in (z,theta):', dens_two_signs, ' expected (1/3) z^4 theta (9-theta^2):', sp.simplify(dens_two_signs - sp.Rational(1,3)*z**4*th*(9-th**2)))
print('tau = ell/A^4 =', sp.simplify(4*a*h**3/A**4).subs({h:a*th}).subs(a, A*z))
# --- total mass
core = th*(9-th**2)*(3/(3+th**2))**sp.Rational(5,2)
total = sp.integrate(core, (th,0,1))/15
print('total mass =', sp.nsimplify(sp.simplify(total)), float(total))
# --- closed-form antiderivatives used by the CDF route
Fcore = sp.integrate(core, (th,0,s))
claimed_core = 3 - 3*(1-s**2)/(1+s**2/3)**sp.Rational(3,2)
print('core antiderivative diff (should be 0):', sp.simplify(Fcore - claimed_core), ' numeric at s=0.3:', sp.N((Fcore-claimed_core).subs(s,sp.Rational(3,10)),30))
tail = 9*th**sp.Rational(-11,4) - th**sp.Rational(-3,4)
Ftail = sp.integrate(tail, (th,s,1))
claimed_tail = sp.Rational(36,7)*(s**sp.Rational(-7,4)-1) - 4*(1-s**sp.Rational(1,4))
print('tail antiderivative diff (should be 0):', sp.simplify(Ftail - claimed_tail))
# tail integrand equals theta(9-theta^2) theta^(-15/4)
print('tail integrand identity:', sp.simplify(th*(9-th**2)*th**sp.Rational(-15,4) - tail))
# split: (3/(3+s^2))^(5/2) = (tau/(4 s^3))^(5/4)  <=>  36 s^3 = tau (3+s^2)^2
print('split identity:', sp.simplify(((3/(3+s**2))**2 - tau/(4*s**3))*4*s**3*(3+s**2)**2))
# monotonicity numerator of core integrand (in theta^2)
w = sp.symbols('w', positive=True)
dcore = sp.simplify(sp.diff(core, th)*(3+th**2)**sp.Rational(7,2)/3**sp.Rational(5,2))
print('core derivative numerator:', sp.expand(dcore), ' roots in w=theta^2:', sp.solve(2*w**2-45*w+27, w))
# tail second derivative
C, r = sp.symbols('C r', positive=True)
t = sp.symbols('t', positive=True)
f = C*(9*t**sp.Rational(-11,4) - r**2*t**sp.Rational(-3,4))
print("f'' =", sp.simplify(sp.diff(f,t,2)*16/C))
# --- B from PROOF constants (7)-(8): (3/2) C_edge (tau A^4)^(2/3) = B tau^(2/3)
Kc = 4/sp.Integer(3)**sp.Rational(5,4)
u0 = 3*A**2; v0 = 2*A**3; c0 = 1/(4*u0*v0)
Cedge = sp.Rational(8,7)*c0*Kc**sp.Rational(-2,3)*u0**sp.Rational(7,6)
B_from_proof = sp.simplify(sp.Rational(3,2)*Cedge*A**sp.Rational(8,3))
B_contract = 9/(28*sp.cbrt(2))
print('B from PROOF (7)-(8):', B_from_proof, ' contract 9/(28 cbrt2) diff:', sp.simplify(B_from_proof - B_contract), float(B_contract))
print('c0 =', sp.simplify(c0), '  v0 - eta_c u0^(3/2) =', sp.simplify(v0 - 2/(3*sp.sqrt(3))*u0**sp.Rational(3,2)))
# kappa from chart vs K_c
kappa = 4*th**3/(3+th**2)**2; eta = 2*(1-th**2)/(3+th**2)**sp.Rational(3,2)
etac = 2/(3*sp.sqrt(3))
print('kappa(theta=1)=', kappa.subs(th,1), ' eta(0)-eta_c=', sp.simplify(eta.subs(th,0)-etac))
print('K_c limit kappa/(eta_c-eta)^(3/2) as theta->0:', sp.limit(kappa/(etac-eta)**sp.Rational(3,2), th, 0), ' PROOF K_c =', Kc, sp.N(Kc))
# leading coefficient of m(tau)/tau^(2/3) from the closed form, expand in s
M = (claimed_core + (tau/4)**sp.Rational(5,4)*claimed_tail)/15
