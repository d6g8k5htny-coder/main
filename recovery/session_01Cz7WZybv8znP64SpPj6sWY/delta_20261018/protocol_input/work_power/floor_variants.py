import sympy as sp, math
eta, x = sp.symbols('eta x', positive=True)
def floor(c, rho_of):
    f = x**2/(2*rho_of**2)
    xs = [z for z in sp.solve(sp.diff(sp.log(f), x), x) if z.is_positive is not False]
    return sp.factor(sp.simplify(f.subs(x, xs[0]))), xs[0]
# P22 guaranteed: (1-eta)^2 m/(2Q) + 2 r/m <= eta
fA, xA = floor((1-eta)**2/2, (eta - (1-eta)**2/(2*x))/2)
# optimistic typical outcome with one-sided t: (1-eta)^2 m/(4Q) + r/m <= eta
fT, xT = floor((1-eta)**2/4, (eta - (1-eta)**2/(4*x)))
for name, f, xx in (("P22", fA, xA), ("typical one-sided", fT, xT)):
    v = f.subs(eta, sp.Rational(1,20))
    print(name, "n/ln(2J/alpha) >=", f, "=", v, " Q/m* =", sp.simplify(xx.subs(eta, sp.Rational(1,20))),
          " min over J>=2,alpha<1:", round(float(v)*math.log(4)), " J=8,alpha=.05:", round(float(v)*math.log(320)))
