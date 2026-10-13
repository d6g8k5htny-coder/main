import mpmath as mp
mp.mp.dps = 30
p = mp.mpf(7)/4
Kp = mp.pi/(2*mp.gamma(1+p)*mp.sin(mp.pi*p/2))
# check K_p numerically
Kchk = mp.quad(lambda u: (1-mp.cos(u))*u**(-1-p), [0, 1, mp.inf])
print('K_p', Kp, 'check', Kchk)
beta = lambda w: mp.mpf(3)/16 - 1j*w/16 + 19*w*w/192
f = lambda w: w**(-1-p)*((mp.mpf(3)/16)**(-mp.mpf(5)/8) - mp.re((1+1j*w)**(-0.5)*beta(w)**(-mp.mpf(5)/8)))
J = mp.quad(f, [0, 0.01, 0.1, 1, 10, 100, mp.inf])
c1 = -(mp.mpf(192)/7)*2**mp.mpf(0.25)*2*mp.pi*(2*mp.pi)**-3*12**-0.5*mp.gamma(mp.mpf(5)/8)/(4*Kp)*J
print('J =', J)
print('c1(d=2) =', c1)
# d = 1 control: c1^(1) = -(192/7) 2^(1/4) * 2 * int pi0(b) E|f4/12|^p db, pi0 = e^{-3b^2/4}/(2 pi sqrt2) (2pi)^-1/2 (12 pi)^-1/2
Mp0 = 2**(p/2)*mp.gamma((p+1)/2)/mp.sqrt(mp.pi)
c11 = -(mp.mpf(192)/7)*2**mp.mpf(0.25)*2*12**(-p)/(2*mp.pi*mp.sqrt(2))*(2*mp.pi)**-0.5*(12*mp.pi)**-0.5*mp.sqrt(4*mp.pi/3)*30**(p/2)*Mp0
print('c1(d=1) =', c11, '  (8/21)24^(1/4) * ? :', mp.mpf(8)/21*24**mp.mpf(0.25))
