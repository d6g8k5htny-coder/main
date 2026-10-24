# Symbolic checks of PROOF.md Sections 1-5 identities
import sympy as sp
z,H,eta,w,u,v,s,rho = sp.symbols('z H eta w u v s rho', real=True)
eta_c = sp.Rational(2,3)/sp.sqrt(3)
zc = -1/sp.sqrt(3)
Phi = -z**4/4 + z**2/2 + eta*z
Phip = sp.diff(Phi, z)
# (21): Phi'_eta(z_c+H) = sqrt3 H^2 - H^3 - (eta_c - eta)
lhs = sp.expand(Phip.subs(z, zc+H))
rhs = sp.expand(sp.sqrt(3)*H**2 - H**3 - (eta_c - eta))
print("(21) identity:", sp.simplify(lhs - rhs) == 0)
# (22): K_c
a_c = 3**sp.Rational(-1,4)
Kc = sp.integrate(1 - sp.sqrt(3)*w**2, (w, -a_c, a_c))
print("(22) K_c =", sp.nsimplify(Kc), " 4/3^(5/4) =", 4/3**sp.Rational(5,4), " equal:", sp.simplify(Kc - 4/3**sp.Rational(5,4)) == 0, " (4/3)a_c equal:", sp.simplify(Kc - sp.Rational(4,3)*a_c)==0)
# (23): K_c^(1/3)/(2 a_c) == (2/3) K_c^(-2/3)
Qc_a = Kc**sp.Rational(1,3)/(2*a_c)
Qc_b = sp.Rational(2,3)*Kc**sp.Rational(-2,3)
print("(23) Q_c forms equal:", sp.simplify(Qc_a - Qc_b) == 0, " Q_c =", sp.N(Qc_b, 20))
# (7): C_edge = 2 c0 Q_c (6/7) u0^(7/6) == (8c0/7) K_c^(-2/3) u0^(7/6)
c0,u0 = sp.symbols('c0 u0', positive=True)
Ce_a = 2*c0*Qc_b*sp.Rational(6,7)*u0**sp.Rational(7,6)
Ce_b = sp.Rational(8,7)*c0*Kc**sp.Rational(-2,3)*u0**sp.Rational(7,6)
print("(7) C_edge forms equal:", sp.simplify(Ce_a-Ce_b)==0)
# eta_c from discriminant: 27 v^2 = 4 u^3 -> |v| = eta_c u^(3/2)
print("eta_c^2 == 4/27:", sp.simplify(eta_c**2 - sp.Rational(4,27))==0)
# Turning point values of g(s) = -s^3 + u s + v at s=+-sqrt(u/3)
g = -s**3 + u*s + v
tp = sp.sqrt(u/3)
print("g(+sqrt(u/3)) - (eta_c u^(3/2) + v) == 0:", sp.simplify(g.subs(s,tp) - (eta_c*u**sp.Rational(3,2) + v))==0)
print("g(-sqrt(u/3)) - (v - eta_c u^(3/2)) == 0:", sp.simplify(g.subs(s,-tp) - (v - eta_c*u**sp.Rational(3,2)))==0)
# Phi' at -1, -1/sqrt3, 0, 1
for pt in [-1, zc, 0, 1]:
    print("  Phi'(%s) = %s" % (pt, sp.simplify(Phip.subs(z,pt))))
# At eta_c: factorization of Phi'
print("Phi'_{eta_c} factor:", sp.factor(Phip.subs(eta, eta_c), extension=sp.sqrt(3)))
print("Phi_{eta_c}(z_c) =", sp.simplify(Phi.subs({eta:eta_c, z:zc})), " Phi_{eta_c}(2/sqrt3) =", sp.simplify(Phi.subs({eta:eta_c, z:2/sp.sqrt(3)})))
# kappa(0) = Phi_0(-1) - Phi_0(0)
print("kappa(0) =", Phi.subs({eta:0,z:-1}) - Phi.subs({eta:0,z:0}))
# (13) core expansion: F0 = b - s^2 c(r) - |zp|^2 with c(r)=r/4
zp = sp.symbols('zp', real=True); b = sp.symbols('b')
r = s**2 + zp**2
F0 = b - s**2*(r/4) - zp**2
print("(12) identity:", sp.simplify(F0 - (b - s**4/4 - (1+s**2/4)*zp**2))==0)
# Derivatives (F0)_s and (F0)_z with general c
c = sp.Function('c')
F0g = b - s**2*c(r) - zp**2
print("(F0)_s:", sp.simplify(sp.diff(F0g, s) - (-2*s*(c(r) + s**2*sp.Derivative(c(r), r) ))) )
