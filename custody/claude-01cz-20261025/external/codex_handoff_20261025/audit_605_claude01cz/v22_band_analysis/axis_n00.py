import sympy as sp, sys
src=open(sys.argv[1]).read().split("print('B J21")[0]; exec(src)
Nx=sub(N)
num,den=sp.fraction(sp.together(Nx))
num=sp.expand(num)
P=sp.Poly(num,s,eta)
c00=sp.factor(P.coeff_monomial(1))
print('denominator:',sp.factor(den)); print('numerator constant term (s^0 eta^0):',c00)
