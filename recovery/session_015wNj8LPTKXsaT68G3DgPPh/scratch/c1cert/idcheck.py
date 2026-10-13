import mpmath as mp
mp.mp.dps = 30
p = mp.mpf(7)/4
Kp = mp.pi/(2*mp.gamma(1+p)*mp.sin(mp.pi*p/2))
Mp0 = 2**(p/2)*mp.gamma((p+1)/2)/mp.sqrt(mp.pi)
M = lambda t: Mp0*mp.hyp1f1(-p/2, mp.mpf(1)/2, -t*t/2)
for mu, s in ((mp.mpf('1.3'), mp.mpf('0.5')), (mp.mpf(0), mp.mpf(1)), (mp.mpf(2), mp.mpf('0.2'))):
    exact = s**p*M(mu/s)
    f = lambda w: w**(-1-p)*(1 - mp.exp(-w*w*s*s/2)*mp.cos(w*mu))
    # split: [0, W] numerically, tail of the '1' exactly; the damped cos tail beyond W is < e^{-W^2 s^2/2}
    W = 60/s
    I = mp.quad(f, mp.linspace(0, W, 400)) + W**(-p)/p
    print(mu, s, exact, I/Kp, 'Kp', Kp)
# Kp by its own: int_0^inf (1-cos u) u^{-1-p} du via damped limit
