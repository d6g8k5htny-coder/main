import mpmath as mp, sys
sys.argv = ['x', '40', 'none']
exec(open('indep_ht.py').read().split("if __name__")[0])
mp.mp.dps = 40
a = 2*mp.zetazero(1).imag; c = 2*mp.zetazero(2).imag
Fz = lambda z: mp.quad(lambda u: Phi(u)*mp.sin(z*u)/u, pts(z))
lhs = Fz(c) - Fz(a)
# rhs: 1/2 int (x-a)(c-x) dH/dt dx, dH/dt = -H_zz ; use closed form via u-integral of polynomial weight
rhs = mp.quad(lambda x: (x-a)*(c-x)*(-Hder(0, x, 2))/2, [a, (a+c)/2, c])
print(mp.nstr(lhs, 15), mp.nstr(rhs, 15))
