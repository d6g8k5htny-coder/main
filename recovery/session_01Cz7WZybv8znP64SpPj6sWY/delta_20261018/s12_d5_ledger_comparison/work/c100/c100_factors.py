import sympy as sp
x, y, X, Y = sp.symbols('x y X Y', real=True)
a, Q, w, z, rho, c, s = sp.symbols('a Q w z rho c s', real=True)
P = (x**3/3 - x/4 - sp.Rational(1, 12) + a/2*(x**2 - sp.Rational(1, 4))*y + Q/2*y**2 + w/2*x*y**2 + z/6*y**3)
Px, Py = sp.diff(P, x), sp.diff(P, y)
sol = sp.solve([Px.subs({x: X, y: Y}), Py.subs({x: X, y: Y})], [a, Q], dict=True)[0]
H = sp.hessian(P, (x, y))
pts = {'M': (-sp.Rational(1, 2), 0), 'S': (sp.Rational(1, 2), 0), 'y': (X, Y)}
for pin, x0 in (('M', -sp.Rational(1, 2)), ('S', sp.Rational(1, 2))):
    sub = {X: x0 + rho*c, Y: rho*s}
    for nm, (px, py) in pts.items():
        d = sp.together(H.subs({x: px, y: py}).det().subs(sol).subs(sub))
        d = sp.factor(d)
        num, den = sp.fraction(d)
        on = min(m[0] for m in sp.Poly(sp.expand(num), rho).monoms())
        od = min(m[0] for m in sp.Poly(sp.expand(den), rho).monoms()) if den.has(rho) else 0
        lead = sp.factor(sp.limit(d / rho**(on - od), rho, 0))
        print(f'chart {pin}: det D2P({nm}) ~ rho^{on-od} * [{lead}]')
