import mpmath as mp
mp.mp.dps = 40
p = mp.mpf(7)/4
Kp = mp.pi/(2*mp.gamma(1+p)*mp.sin(mp.pi*p/2))
C0 = (mp.mpf(3)/16)**(-mp.mpf(5)/8)
beta = lambda w: mp.mpf(3)/16 - 1j*w/16 + 19*w*w/192
h = lambda w: (1+1j*w)**(-0.5)*beta(w)**(-mp.mpf(5)/8)
# integrate in x = log w to tame both ends
g = lambda x: mp.exp(x)**(-p)*(C0 - mp.re(h(mp.exp(x))))
pts = list(range(-40, 41, 2))
J = mp.quad(g, pts)
# tail beyond w = e^40: C0 * int w^{-1-p} = C0 e^{-40p}/p, plus the h-part ~ w^{-1-p-7/4}
J += C0*mp.exp(-40*p)/p
# below w = e^-40: integrand ~ w^{1-p}*const -> negligible
c1 = -(mp.mpf(192)/7)*2**mp.mpf(0.25)*2*mp.pi*(2*mp.pi)**-3*12**-0.5*mp.gamma(mp.mpf(5)/8)/(4*Kp)*J
print('J =', J); print('c1(d=2) =', c1)
