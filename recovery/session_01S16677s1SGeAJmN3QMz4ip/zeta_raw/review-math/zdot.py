import mpmath as mp, sys
sys.argv = ['x', '40', 'none']
exec(open('indep_ht.py').read().split("if __name__")[0])
mp.mp.dps = 40
for k in [1, 2, 5]:
    z = 2*mp.zetazero(k).imag
    print(k, mp.nstr(Hder(0, z, 2)/Hder(0, z, 1), 12))
