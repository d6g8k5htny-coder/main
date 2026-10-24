import sympy as sp
G=sp.gamma(sp.Rational(7,6)); pi=sp.pi
B3=3*G/(2**sp.Rational(4,3)*pi**sp.Rational(7,2))                  # LS-DER-056 (25)
# P (15.2) constant times p_G(0)p_V(0) normalizations and tau^(4/3)=(144 sigma^2)^(2/3)
Pc=G/(24**sp.Rational(1,3)*sp.sqrt(pi))*(2*pi)**sp.Rational(-3,2)*(2*pi)**sp.Rational(-3,2)*144**sp.Rational(2,3)
print('B3/Pconst =',sp.nsimplify(sp.simplify(B3/Pc)))
# chain (23) gamma integral check
k,s2=sp.symbols('kappa sigma2',positive=True)
lhs=sp.integrate((2*pi*s2)**sp.Rational(-1,2)*sp.exp(-k**2/(72*s2))*k**sp.Rational(4,3),(k,0,sp.oo))
rhs=G*72**sp.Rational(7,6)*s2**sp.Rational(2,3)/(2*sp.sqrt(2*pi))
print('(23) lhs/rhs =',sp.simplify(lhs/rhs))
# P scalar integral check: 144 int k^(4/3) phi_tau(12k) dk = Gamma(7/6) tau^(4/3)/(24^(1/3) sqrt(pi))
tau=sp.symbols('tau',positive=True)
l2=144*sp.integrate(k**sp.Rational(4,3)*sp.exp(-(12*k)**2/(2*tau**2))/(tau*sp.sqrt(2*pi)),(k,0,sp.oo))
print('P scalar lhs/rhs =',sp.simplify(l2/(G*tau**sp.Rational(4,3)/(24**sp.Rational(1,3)*sp.sqrt(pi)))))
# chain (24) at reference covariance: sigma_t^2=6/144, D=29/6-sqrt6, detSg=1, detSh=3, sphere 4pi
cref24=B3*4*pi*(sp.Rational(6,144))**sp.Rational(2,3)*(sp.Rational(29,6)-sp.sqrt(6))/sp.sqrt(3)
cinf=2**sp.Rational(-23,6)*3**sp.Rational(-8,3)*(29*sp.sqrt(6)-36)*sp.gamma(sp.Rational(1,6))*pi**sp.Rational(-5,2)
print('chain(24)_ref / cinf =',sp.N(cref24/cinf,60))
# chain kappa vs P k: kappa=6k. Kac-Rice 1/6 and lifetime 6^(2/3)/3 vs P's 4=12/3
print('d=2 chain LS-DER-042 (30) / PROOF (1) d=2:', sp.N((2**sp.Rational(2,3)*3**sp.Rational(5,6)*sp.gamma(sp.Rational(1,6))/(54*pi**sp.Rational(3,2)))/(G*sp.Rational(3,2)**sp.Rational(1,3)*sp.Rational(4,3)/(2*sp.sqrt(3)*pi*sp.sqrt(pi))),60))
