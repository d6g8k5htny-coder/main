import mpmath as mp
mp.mp.dps = 40
p = mp.mpf(7)/4
Kp = mp.pi/(2*mp.gamma(1+p)*mp.sin(mp.pi*p/2))
C0 = (mp.mpf(3)/16)**(-mp.mpf(5)/8)
I = mp.mpc(0, 1)
beta = lambda w: mp.mpf(3)/16 - I*w/16 + 19*w*w/192
h = lambda w: (1 + I*w)**(-mp.mpf(1)/2)*beta(w)**(-mp.mpf(5)/8)
f = lambda u: 4*u**(-8)*(C0 - mp.re(h(u**4)))
u0 = mp.mpf('0.01')
U = mp.mpf(40)
J = u0*f(mp.mpf('0.003')) + mp.quad(f, mp.linspace(u0, U, 400)) + mp.mpf(4)/7*C0*U**(-7)
pref = -(mp.mpf(192)/7)*2**mp.mpf(0.25)*2*mp.pi*(2*mp.pi)**-3*12**-0.5*mp.gamma(mp.mpf(5)/8)/(4*Kp)
print('J =', J); print('c1(d=2) =', pref*J)
