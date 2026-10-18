# Adversarial re-derivation (symbolic). Independent of lens scripts.
import sympy as sp
b, k, kap, ell, r, sig, tau, D, pG, pV = sp.symbols('b k kappa ell r sigma tau D p_G p_V', positive=True)

phi = lambda x, s: sp.exp(-x**2/(2*s**2))/sp.sqrt(2*sp.pi*s**2)

# --- P (13.6)->(15.2): c = 4 int k^(-2/3) pi_0 z_0 db dk ; pi_0 = p_(f,V)(b,0) pG phi_tau(12k);
#     z_0 = (6k)^2 E[..|f=b,V=0]; b-integration -> pV*D
P_k = sp.integrate(4*k**sp.Rational(-2,3)*pG*phi(12*k, tau)*(6*k)**2*pV*D, (k, 0, sp.oo))
P_152 = sp.gamma(sp.Rational(7,6))/(sp.cbrt(24)*sp.sqrt(sp.pi))*pG*pV*tau**sp.Rational(4,3)*D
print('P (13.6) k-integral / P (15.2):', sp.simplify(P_k/P_152))

# --- chain LS-DER-056: (1/6)*(6^(2/3)/3) * p_h D * int p_g phi_sigma(kappa/6) kappa^(4/3) dkappa
C_k = sp.Rational(1,6)*6**sp.Rational(2,3)/3*pV*D*sp.integrate(pG*phi(kap/6, sig)*kap**sp.Rational(4,3), (kap, 0, sp.oo))
C_23 = pG*sp.gamma(sp.Rational(7,6))*72**sp.Rational(7,6)*(sig**2)**sp.Rational(2,3)/(2*sp.sqrt(2*sp.pi))
print('chain kappa integral / 056 (23):', sp.simplify(sp.integrate(pG*phi(kap/6, sig)*kap**sp.Rational(4,3), (kap, 0, sp.oo))/C_23))
print('chain assembled / P(15.2) with tau=12 sigma:', sp.simplify((C_k/P_152).subs(tau, 12*sig)))

# --- 056 (24)-(25) constant vs P constant with pG pV = (2pi)^-3 (detg deth)^-1/2
B3 = 3*sp.gamma(sp.Rational(7,6))/(2**sp.Rational(4,3)*sp.pi**sp.Rational(7,2))
B3_from_P = sp.gamma(sp.Rational(7,6))/(sp.cbrt(24)*sp.sqrt(sp.pi))*(2*sp.pi)**-3*144**sp.Rational(2,3)
print('B_3 / P-derived constant:', sp.nsimplify(sp.simplify(B3/B3_from_P)))

# --- coordinate-free first principles: nu(ell) = int r^(d-1) dr dsigma db p_O(b,b-ell,0,0) E[W]
# with leading-order E[W] = r^2 * (f_tt(M)/r)(f_tt(S)/r) soft * E[det^2 1] and f_ttt fixed by the
# exact cubic model through the pins. Derive f_ttt and soft factors from the cubic model directly.
x, a0, a1, a2, a3 = sp.symbols('x a0 a1 a2 a3')
g = sp.symbols('g', positive=True)  # physical gap f(M)-f(S)
fx = a0 + a1*x + a2*x**2/2 + a3*x**3/6
sol = sp.solve([fx.subs(x, -r/2) - b, fx.subs(x, r/2) - (b - g),
                sp.diff(fx, x).subs(x, -r/2), sp.diff(fx, x).subs(x, r/2)], [a0, a1, a2, a3], dict=True)[0]
f3 = sol[a3]; fttM = sp.diff(fx, x, 2).subs(x, -r/2).subs(sol); fttS = sp.diff(fx, x, 2).subs(x, r/2).subs(sol)
print('cubic model: f_ttt =', sp.simplify(f3), ' f_tt(M)/r =', sp.simplify(fttM/r), ' f_tt(S)/r =', sp.simplify(fttS/r))
print('  P: gap=k r^3 -> f_ttt =', sp.simplify(f3.subs(g, k*r**3)), '; chain: gap=kappa r^3/6 -> f_ttt =', sp.simplify(f3.subs(g, kap*r**3/6)))
# P pin density: 12 r^-(d+3) * pi_0 with t-factor phi_tau(f_ttt) ; chain: r^-(d+3) p_Y with u=-f_ttt/12 ~ N(0,sigma^2)
d = sp.Integer(3)
f3l = sp.simplify(f3.subs(g, ell))   # = 12 ell / r^3
soft = sp.Abs(sp.simplify(fttM/r*fttS/r)).subs(g, ell)
P_int = r**(d-1)*12*r**(-(d+3))*pG*phi(f3l, tau)*pV*D*r**2*soft     # P coordinates
C_int = r**(d-1)*r**(-(d+3))*pG*phi(-f3l/12, sig)*pV*D*r**2*soft    # chain coordinates
print('P first-principles integrand / chain first-principles integrand (tau=12 sigma):', sp.simplify((P_int/C_int).subs(tau, 12*sig)))
nu_P = sp.integrate(sp.simplify(P_int), (r, 0, sp.oo))
print('first-principles nu(ell) * ell^(1/3) / P(15.2) integrand:', sp.simplify(nu_P*ell**sp.Rational(1,3)/P_152))

# --- reference values (K=exp(-|x|^2/2)), d=3
tau2 = 15 - 9; detV = 3; D2 = sp.Rational(29,6) - sp.sqrt(6)
cref = sp.gamma(sp.Rational(7,6))/(sp.cbrt(24)*sp.sqrt(sp.pi))*(2*sp.pi)**sp.Rational(-3,2)*(2*sp.pi)**sp.Rational(-3,2)/sp.sqrt(detV)*sp.Integer(tau2)**sp.Rational(2,3)*D2*4*sp.pi
proof1 = sp.gamma(sp.Rational(7,6))*sp.cbrt(sp.Rational(3,2))*D2/(2*sp.sqrt(3)*sp.pi**2*sp.sqrt(sp.pi))
cls077 = 2**sp.Rational(-23,6)*3**sp.Rational(-8,3)*(29*sp.sqrt(6)-36)*sp.gamma(sp.Rational(1,6))*sp.pi**sp.Rational(-5,2)
der056_38 = 2**sp.Rational(1,6)*3**sp.Rational(1,3)*(29*sp.sqrt(6)-36)*sp.gamma(sp.Rational(1,6))/(432*sp.pi**sp.Rational(5,2))
print('P(15.2)@BF / PROOF(1):', sp.N(cref/proof1, 40), ' /077:', sp.N(cref/cls077, 40), ' /056(38):', sp.N(cref/der056_38, 40))
print('c_3,ref =', sp.N(cref, 40))
# BF sigma_t^2 from 047 (4): Var(-f111/12)=15/144, Cov(.,f1)=3/12
print('BF sigma^2 =', sp.Rational(15,144) - sp.Rational(3,12)**2, ' 144*sigma^2 =', 144*(sp.Rational(15,144) - sp.Rational(1,16)))
