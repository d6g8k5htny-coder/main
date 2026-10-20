import sympy as sp
r,t,d = sp.symbols('r t delta', positive=True); a = sp.sqrt(15)/2; e = r/t
I_claim = sp.integrate(r**4*t**-6*t, (t, a*r, d))
print('int_{a r}^{delta} r^4 t^-6 t dt =', sp.simplify(I_claim), ' ; x(5/24) r^3 =', sp.expand(sp.Rational(5,24)*r**3*I_claim))
corr = r**-2*e**2*(e+t)**2*(e**2+t)
print('corrected envelope expanded:', sp.expand(corr))
for lo,hi,lab in ((a*r, r**sp.Rational(2,3),'t<=r^(2/3)'),(r**sp.Rational(2,3), r**sp.Rational(1,2),'r^(2/3)..r^(1/2)'),(r**sp.Rational(1,2), d,'t>=r^(1/2)')):
    I = sp.expand(sp.integrate(sp.expand(r**3*corr*t),(t,lo,hi)))
    lead = None
    for pw in (3, sp.Rational(13,3)):
        L = sp.limit(I/r**pw, r, 0)
        print(f'  {lab}: lim I/r^{pw} =', L)
# P2-geometry ledgers
print('Z^-1 r^-3 rho^-2 r^6 rho^2 =', sp.simplify(r**-2*r**-3*r**6))
print('P2 Region II exponent: ', -2-3+3-2-6)
c_ = sp.symbols('c', positive=True)
print('e^{-c/(2r^2)} <= 6!(2/c)^6 r^12  [6th series term of e^{x}, x=c/(2r^2)]:', sp.simplify(sp.factorial(6)/(c_/(2*r**2))**6 - sp.factorial(6)*(2/c_)**6*r**12)==0)
# Region-row overlap: window-integrated intensity per physical area at t=A r
A = sp.symbols('A', positive=True)
print('per-area window intensity at t=A r:', sp.simplify(r**3 * r**4*(A*r)**-6))
# P2-style column bounds in LS geometry
print('P2-style count power:', sp.limit(sp.integrate(r**3*r**-2*t**-6*(r*t)**2*t*t,(t,a*r,d))/r**2, r, 0), '* r^2')
