import mpmath as mp
mp.mp.dps = 40
p = mp.mpf(7)/4
Kp = mp.pi/(2*mp.gamma(1+p)*mp.sin(mp.pi*p/2))
C0 = (mp.mpf(3)/16)**(-mp.mpf(5)/8)
beta = lambda w: mp.mpf(3)/16 - mp.mpc(0, 1)*w/16 + 19*w*w/192
h = lambda w: (1 + mp.mpc(0, 1)*w)**(-mp.mpf(1)/2)*beta(w)**(-mp.mpf(5)/8)
# omega = u^4: omega^{-1-p} d omega = 4 u^{-8} du
def f(u):
    if u == 0:
        return mp.mpf(0)
    w = u**4
    return 4*u**(-8)*(C0 - mp.re(h(w)))
U = mp.mpf(40)
J = mp.quad(f, mp.linspace(0, U, 161)) + mp.mpf(4)/7*C0*U**(-7)
# the h-part tail beyond U: |h| <= (1+w^2)^{-1/4} |beta|^{-5/8} ~ w^{-7/4}: bounded by int 4u^-8 u^-7 ~ tiny
c1 = -(mp.mpf(192)/7)*2**mp.mpf(0.25)*2*mp.pi*(2*mp.pi)**-3*12**-0.5*mp.gamma(mp.mpf(5)/8)/(4*Kp)*J
print('J =', J); print('c1(d=2) =', c1)
print('f near 0:', f(mp.mpf('1e-3')), f(mp.mpf('0.1')))
