from fractions import Fraction as F
Q=F(5)
print((F(1),-4*Q,4*Q*Q),(F(1),-4*Q,(2*Q)**2))
# tautology for any Q
import random
print(all((F(1),-4*q,4*q*q)==(F(1),-4*q,(2*q)**2) for q in [F(random.randint(-10**6,10**6),random.randint(1,10**6)) for _ in range(1000)]))
try:
    import sympy as sp
    x,q=sp.symbols('x q')
    print(sp.expand(x**2-4*q*(x-q)-(x-2*q)**2))
    print(sp.Poly(x**2-4*q*(x-q),x).all_coeffs(), sp.Poly((x-2*q)**2,x).all_coeffs())
except ImportError as e:
    print("nosympy",e)
