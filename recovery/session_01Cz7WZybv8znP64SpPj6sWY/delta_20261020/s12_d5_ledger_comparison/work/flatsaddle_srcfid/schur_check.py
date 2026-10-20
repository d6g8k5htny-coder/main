"""Tail-set relative-error check for carrier l.79-80 and l.131-132, and P2 (P15) pathwise version."""
import sympy as sp
r, kap, nu, eps, m, zs, zt, L = sp.symbols('r kappa nu epsilon m z_s z_t L', real=True)
# (F1): f_tt(S) = kappa r (1 + O_P(r)) -> kappa r + eps r^2 ; tail set mu_tilde_S = m r^2
mu = m*r**2 - r*nu**2/kap
detHS = (kap*r + eps*r**2)*(-mu) - (r*nu)**2
print('det H_S on tail set           :', sp.factor(sp.expand(detHS)))
print('carrier -kappa r mu~_S(1+O(r)) :', sp.expand(-kap*r*m*r**2), '  -> extra term eps*nu^2*r^3/kappa is same order r^3')
# exact Schur form: det = f_tt * (f_ss - f_ts^2/f_tt); f_s on f_t-null line at quadratic order
a, b, c, ys = sp.symbols('a b c y_s', real=True)
H = sp.Matrix([[a, b], [b, c]]); yv = sp.Matrix([-b*ys/a, ys])
print('H y on f_t-null line          :', list(sp.simplify(H*yv)), ' ; det/a*y_s =', sp.simplify(H.det()/a*ys))
# window offset with cross term
yt = -nu*zs/kap*r**2; ys_ = zs*r**2
mu0 = sp.symbols('mu', real=True)
kept = sp.Rational(1,2)*kap*r*yt**2
cross = r*nu*yt*ys_
fss = -sp.Rational(1,2)*mu0*ys_**2
print('kept, cross, f_ss terms       :', sp.simplify(kept), '|', sp.simplify(cross), '|', fss)
print('sum = -1/2 r^4 zs^2 mu~_S ?   :', sp.simplify(kept+cross+fss + sp.Rational(1,2)*r**4*zs**2*(mu0 + r*nu**2/kap)) == 0)
# literal l.159-162 decomposition both readings
print('l.159-162 literal (den as -2):', 4-3+2+2+1-2, ' (den as -(-2)):', 4-3+2+2+1+2)
# l.140 literal factor product vs l.141
cs, p0 = sp.symbols('c_s p0', positive=True)
lit = kap**2*r**2*p0*(cs*zs*r**2)*(r**2)*(cs*zs)*(cs*zs**2*r**4)**-1*(cs*zs*r**2)
print('l.140 literal                  :', sp.simplify(lit), ' vs l.141: kappa^2 r^4 c_s p0 C1')
