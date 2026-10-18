import sympy as sp
x=sp.symbols('x')
for t in [sp.Rational(1,8), sp.Rational(1,2), sp.Integer(1), sp.Rational(3,16), sp.Rational(1,1024)]:
    for j in range(0,4):
        a=2**j*t; b=2**(j+1)*t
        w=sp.Rational(3,2)*(b**sp.Rational(2,3)-a**sp.Rational(2,3))
        mp=sp.minimal_polynomial(w,x)
        print(t,j,sp.degree(mp,x), sp.N(w,20))
