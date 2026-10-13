import sympy as sp, pickle, mpmath as mp
mp.mp.dps = 30
r, b, k = sp.symbols('r b k', real=True)
D = pickle.load(open('a2_d2.pkl', 'rb'))
A0, A2 = D['A0'], D['A2']
f0 = sp.lambdify((b, k), A0, 'mpmath'); f2 = sp.lambdify((b, k), A2, 'mpmath')
c = mp.mpf(2) * mp.pi / 3 * mp.quad(lambda bb: mp.quad(lambda kk: f0(bb, kk) * kk ** (-mp.mpf(2)/3), [0, 0.5, 2, mp.inf]), [-mp.inf, 0, mp.inf])
print('c =', c, ' vs c_2,inf 0.073406919306034271030')
c2 = mp.mpf(2) * mp.pi / 3 * mp.quad(lambda bb: mp.quad(lambda kk: (f2(bb, kk) - f2(bb, 0)) * kk ** (-mp.mpf(4)/3), [0, 0.5, 2, mp.inf]), [-mp.inf, 0, mp.inf])
print('c2 =', c2, ' vs #216 0.22152441')
