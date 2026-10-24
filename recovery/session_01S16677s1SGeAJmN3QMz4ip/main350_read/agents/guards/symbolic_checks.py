import sympy as sp
t, s, z, r, u, v = sp.symbols('t s z r u v', real=True)
q = sp.exp(-1/t)
qp = sp.diff(q, t); qpp = sp.diff(q, t, 2)
print('q\'  ==', sp.simplify(qp - t**-2*sp.exp(-1/t)) == 0)
print('q\'\' ==', sp.simplify(qpp - (1-2*t)*t**-4*sp.exp(-1/t)) == 0)
# q' max at 1/2 equals 4 e^-2; q'' sign
print('q\'(1/2) = 4e^-2:', sp.simplify(qp.subs(t, sp.Rational(1,2)) - 4*sp.exp(-2)) == 0)
print('q\'\'(1/2)=0:', sp.simplify(qpp.subs(t, sp.Rational(1,2))) == 0)
# (G2)
Q = sp.Function('q')
psi = Q(1-t)/(Q(t)+Q(1-t))
lhs = sp.diff(psi, t)
D1 = sp.Function('D1')  # derivative placeholder not needed; compare via Subs
qf = sp.Lambda(t, sp.exp(-1/t))
psi_c = qf(1-t)/(qf(t)+qf(1-t))
g2 = -(sp.diff(qf(t),t).subs(t,1-t)*qf(t) + qf(1-t)*sp.diff(qf(t),t))/(qf(t)+qf(1-t))**2
print('(G2) psi\' identity:', sp.simplify(sp.diff(psi_c, t) - g2) == 0)
print('psi(1/2)=1/2:', sp.simplify(psi_c.subs(t, sp.Rational(1,2)) - sp.Rational(1,2)) == 0)
print("psi'(1/2) = -2:", sp.simplify(sp.diff(psi_c,t).subs(t, sp.Rational(1,2)) + 2) == 0)
# Section 4: c'(r)
r1, R = sp.Rational(1,32), sp.Rational(1,8)
w = R**2/2 - r1**2
Psi = sp.Function('Psi')
c = 1 - (1 - r/4)*Psi((r - r1**2)/w)
# expected psi/4 - (1-r/4) psi'/w
# Compare by substituting a concrete Psi
arg = (r - r1**2)/w
Psi_c = sp.Lambda(t, sp.sin(t)**2)  # arbitrary smooth test function
c_c = 1 - (1 - r/4)*Psi_c(arg)
expected_c = Psi_c(arg)/4 - (1 - r/4)*sp.diff(Psi_c(t), t).subs(t, arg)/w
print("(G7) c'(r) identity (test Psi):", sp.simplify(sp.diff(c_c, r) - expected_c) == 0)
# F0 derivatives with generic c
rr = s**2+z**2
# use concrete test C to verify the formula shape
C_c = sp.Lambda(t, sp.cos(t) + t**3)
F0c = sp.Rational(4,9) - s**2*C_c(rr) - z**2
Cpc = sp.diff(C_c(t), t).subs(t, rr)
print('(F0)_s identity:', sp.simplify(sp.diff(F0c, s) + 2*s*(C_c(rr) + s**2*Cpc)) == 0)
print('(F0)_z identity:', sp.simplify(sp.diff(F0c, z) + 2*z*(1 + s**2*Cpc)) == 0)
# Annular core polynomial with c = r/4
F0core = sp.Rational(4,9) - s**2*(rr/4) - z**2
print('core F0 == b - s^4/4 - (1+s^2/4) z^2:', sp.expand(F0core - (sp.Rational(4,9) - s**4/4 - (1+s**2/4)*z**2)) == 0)
print('|(F0)_s| core == |s|(s^2+z^2/2):', sp.expand(sp.diff(F0core, s) + s*(s**2 + z**2/2)) == 0)
print('|(F0)_z| core == 2|z|(1+s^2/4):', sp.expand(sp.diff(F0core, z) + 2*z*(1 + s**2/4)) == 0)
# Core field with chi=1: F = b + P(s) - (1+s^2/4) z^2, P = -s^4/4 + u s^2/2 + v s
F = F0core + (u*s**2/2 + v*s)
print('F core == b + P - (1+s^2/4)z^2:', sp.expand(F - (sp.Rational(4,9) - s**4/4 + u*s**2/2 + v*s - (1+s**2/4)*z**2)) == 0)
print("F_s at z=0 == -s^3 + u s + v:", sp.expand(sp.diff(F, s).subs(z, 0) - (-s**3 + u*s + v)) == 0)
print("F_z == -2 z (1 + s^2/4):", sp.expand(sp.diff(F, z) + 2*z*(1+s**2/4)) == 0)
# Hessian (PROOF 15) at z=0
print('F_ss at z=0 == u - 3 s^2:', sp.expand(sp.diff(F, s, 2).subs(z, 0) - (u - 3*s**2)) == 0)
# seed h = b - r with w_i = 3^-i: cos(2 pi y) = 1 - 2 sin^2(pi y)
y1, y2 = sp.symbols('y1 y2', real=True)
h = sp.Rational(1,3)*sp.cos(2*sp.pi*y1) + sp.Rational(1,9)*sp.cos(2*sp.pi*y2)
ss = sp.sqrt(sp.Rational(2,3))*sp.sin(sp.pi*y1); zz = sp.sqrt(sp.Rational(2,9))*sp.sin(sp.pi*y2)
print('h == b - s^2 - z^2:', sp.simplify(h - (sp.Rational(4,9) - ss**2 - zz**2)) == 0)
# grad of chi inner argument
rho2 = s**2 + z**2
garg = (rho2 - r1**2/4)/(3*r1**2/4)
grad = sp.Matrix([sp.diff(garg, s), sp.diff(garg, z)])
print('||grad arg||^2 == (8 rho/(3 r1^2))^2:', sp.simplify(grad.dot(grad) - (8*sp.sqrt(rho2)/(3*r1**2))**2) == 0)
# wedge: (2/(3 sqrt 3)) (3A^2)^(3/2) == 2 A^3
Asym = sp.symbols('A', positive=True)
print('wedge symbolic:', sp.simplify(sp.Rational(2,3)/sp.sqrt(3)*(3*Asym**2)**sp.Rational(3,2) - 2*Asym**3) == 0)
