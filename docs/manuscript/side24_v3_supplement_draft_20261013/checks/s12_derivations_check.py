"""Exact checks for the derivations written out in draft 6, in Sections S12 and S8 (SymPy). Run: python3 -I s12_derivations_check.py

Finite algebra only; not a check of any analytic estimate.
"""
import sympy as sp

r, s, x, k, b = sp.symbols('r s x k b', positive=True)
a, c = -r/2, r/2
ok = []

def check(name, cond):
    ok.append((name, bool(cond)))
    print(('PASS ' if cond else 'FAIL ') + name)

# 1. Lemma S12.10: the axial block of T_r and its determinant, the transverse blocks, the target.
fM, fS, gM, gS = sp.symbols('fM fS gM gS')
rows = [ (fM+fS)/2, (fS-fM)/r, (gS-gM)/r, 6/r**2*(gM+gS-2*(fS-fM)/r) ]
J = sp.Matrix([[sp.diff(R_, v) for v in (fM, fS, gM, gS)] for R_ in rows])
check('axial determinant = -12 r^-4', sp.simplify(J.det() + 12/r**4) == 0)
yM, yS = sp.symbols('yM yS')
Jt = sp.Matrix([[sp.diff(R_, v) for v in (yM, yS)] for R_ in [(yM+yS)/2, (yS-yM)/r]])
check('transverse block determinant = 1/r', sp.simplify(Jt.det() - 1/r) == 0)
for d in range(2, 7):
    m = d-1
    check('|det T_r| = 12 r^-(d+3) for d=%d' % d, sp.simplify(12/r**4 * r**(-m) - 12*r**(-(d+3))) == 0)
tgt = [R_.subs({fM: b, fS: b-k*r**3, gM: 0, gS: 0}) for R_ in rows]
check('target v_r', [sp.simplify(t) for t in tgt] == [b-k*r**3/2, -k*r**2, 0, 12*k])

# 2. Lemma S12.11(a): averaging kernels.  Generic polynomial phi along the axis.
cs = sp.symbols('c0:8')
phi = sum(cs[i]*x**i for i in range(8))
d1, d2, d3 = [sp.diff(phi, x, n) for n in (1, 2, 3)]
U1 = (phi.subs(x, c) - phi.subs(x, a))/r
U2 = (d1.subs(x, c) - d1.subs(x, a))/r
U3 = 6/r**2*(d1.subs(x, a) + d1.subs(x, c) - 2*(phi.subs(x, c) - phi.subs(x, a))/r)
check('U1 = average of f_x', sp.expand(U1 - sp.integrate(d1, (x, a, c))/r) == 0)
check('U2 = average of f_xx', sp.expand(U2 - sp.integrate(d2, (x, a, c))/r) == 0)
kern = 6/r**3*(r**2/4 - x**2)
check('U3 kernel has mass one', sp.simplify(sp.integrate(kern, (x, a, c)) - 1) == 0)
check('U3 = kernel average of f_xxx', sp.expand(U3 - sp.integrate(kern*d3, (x, a, c))) == 0)

# 3. Lemma S12.14: endpoint identities under the gradient pins f_x(a) = f_x(c) = 0.
# Impose the two gradient pins on the generic polynomial by solving for c1, c2.
sol = sp.solve([d1.subs(x, a), d1.subs(x, c)], [cs[1], cs[2]], dict=True)[0]
P = phi.subs(sol); P1, P2, P3 = [sp.diff(P, x, n) for n in (1, 2, 3)]
check('f_xx(a) = -(1/r) int (c-x) f_xxx', sp.simplify(P2.subs(x, a) + sp.integrate((c-x)*P3, (x, a, c))/r) == 0)
check('f_xx(c) = (1/r) int (x-a) f_xxx', sp.simplify(P2.subs(x, c) - sp.integrate((x-a)*P3, (x, a, c))/r) == 0)
check('int f_xx over the pin interval = 0', sp.simplify(sp.integrate(P2, (x, a, c))) == 0)
check('Hermite: f(c)-f(a) = -(1/2) int (x-a)(c-x) f_xxx', sp.simplify(P.subs(x, c) - P.subs(x, a) + sp.integrate((x-a)*(c-x)*P3, (x, a, c))/2) == 0)
check('Hermite weight mass r^3/6', sp.simplify(sp.integrate((x-a)*(c-x), (x, a, c)) - r**3/6) == 0)
# With f_xxx = 12k constant (cubic with the pins), alpha_M = f_xx(a)/r = -6k and alpha_S = 6k.
check('alpha_M = -6k when f_xxx = 12k', sp.simplify(-sp.integrate((c-x)*12*k, (x, a, c))/r**2 + 6*k) == 0)
check('alpha_S = 6k when f_xxx = 12k', sp.simplify(sp.integrate((x-a)*12*k, (x, a, c))/r**2 - 6*k) == 0)
# Bounds: int (c-x) dx = r^2/2 and int |x - e| dx over the interval from an endpoint e = r^2/2.
check('int (c-x) = r^2/2', sp.simplify(sp.integrate(c-x, (x, a, c)) - r**2/2) == 0)
check('int |x-a| = r^2/2', sp.simplify(sp.integrate(x-a, (x, a, c)) - r**2/2) == 0)

# 4. Section S8, after eq:remainder-window: the completing-square identity and the bound on the support.
w_, h_, u_, v_ = sp.symbols('w h u v', real=True)
check('S8: (w-h+u)(w+h+v) = (w+(u+v)/2)^2 - (h+(v-u)/2)^2',
      sp.expand((w_-h_+u_)*(w_+h_+v_) - ((w_+(u_+v_)/2)**2 - (h_+(v_-u_)/2)**2)) == 0)

n_pass = sum(1 for _, v in ok if v)
print('ALL PASS (%d/%d)' % (n_pass, len(ok)) if n_pass == len(ok) else 'FAILURES (%d/%d pass)' % (n_pass, len(ok)))
