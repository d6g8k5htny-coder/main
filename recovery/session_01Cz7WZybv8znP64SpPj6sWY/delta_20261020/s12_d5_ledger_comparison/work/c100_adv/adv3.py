import sympy as sp, mpmath as mp
# (a) z0(6/5) = E[q^2; q<0], q~N(-6/5, 2), by direct quadrature
b = mp.mpf(6)/5
z0 = mp.quad(lambda q: q**2*mp.exp(-(q+b)**2/4)/mp.sqrt(4*mp.pi), [-mp.inf, 0])
print('z0(6/5) quad =', z0)
print('ledger r^-2 normalizers:', [0.00808417960496/0.05**2, 0.00201529242071/0.025**2, 0.000503621144509/0.0125**2])
# MC standard error for 300k samples of q^2 1{q<0}
m2 = z0; m4 = mp.quad(lambda q: q**4*mp.exp(-(q+b)**2/4)/mp.sqrt(4*mp.pi), [-mp.inf, 0])
print('MC SE (n=3e5) ~', mp.sqrt(m4 - m2**2)/mp.sqrt(300000))
# (b) Region II (P2 axis q=0): exact longitudinal quartic g = H + (A4/24) x^2 (x-r)^2 keeps all four pins;
#     third station at x = r p forces A4, and the Hessian xx entries differ from the cubic model at LEADING order.
x, r, k, p, A4, bb = sp.symbols('x r k p A4 b', real=True)
H = bb + k*(2*x**3 - 3*r*x**2)
g = H + A4/24*x**2*(x - r)**2
assert sp.simplify(g.subs(x,0)-bb)==0 and sp.simplify(g.subs(x,r)-(bb-k*r**3))==0
assert sp.simplify(sp.diff(g,x).subs(x,0))==0 and sp.simplify(sp.diff(g,x).subs(x,r))==0
A4s = sp.solve(sp.diff(g,x).subs(x, r*p), A4)
A4s = [s for s in A4s][0]
print('forced A4 =', sp.factor(A4s))
for nm, pt in (('M', 0), ('S', r), ('X', r*p)):
    true = sp.factor(sp.diff(g,x,2).subs(x,pt).subs(A4, A4s))
    cubic = sp.factor(sp.diff(H,x,2).subs(x,pt))
    print(f'g_xx({nm}) true = {true};  cubic-model = {cubic}')
# (c) heuristic: absolute O(r) error in each vanishing scaled det -> r^3 * int (rho + r)^2/rho^2 * rho d rho
rho, eps = sp.symbols('rho epsilon', positive=True)
I = sp.integrate((rho + r)**2/rho, (rho, eps, sp.Rational(1,4)))
print('area integral (rho+r)^2/rho from eps:', sp.simplify(I), ' -> log term coefficient r^2:', sp.limit(I/sp.log(1/eps), eps, 0))
# (d) the leading angular growth of C100's collision integrand: density s^-2 times product s^-6
c, s, w, z = sp.symbols('c s w z', real=True)
lead = -((-2*c**2 + s**2*w)*(-4*c**3 + 3*c*s**2*w + s**3*z)**2)/(4*s**6)
print('s^6*lead at s->0 (c->1):', sp.limit((s**6*lead).subs(c, sp.sqrt(1-s**2)), s, 0))
# P2 Region I polynomial: (1+t^2)^3 (1+t^6), t=c/s -> s^-12 ; C100: s^-8
