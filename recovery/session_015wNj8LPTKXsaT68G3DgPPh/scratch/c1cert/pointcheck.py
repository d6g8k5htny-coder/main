import mpmath as mp
mp.mp.dps = 30
p = mp.mpf(7)/4
Kp = mp.pi/(2*mp.gamma(1+p)*mp.sin(mp.pi*p/2))
Mp0 = 2**(p/2)*mp.gamma((p+1)/2)/mp.sqrt(mp.pi)
def M(t):
    return Mp0*mp.hyp1f1(-p/2, mp.mpf(1)/2, -t*t/2)
def G2D(a):
    c = mp.sqrt(mp.mpf(6)/19)
    f = lambda g: mp.npdf(g, 0, mp.sqrt(2))*M(c*(a/4 - g*g/a))
    E = 2*mp.quad(f, [0, mp.sqrt(a)/4, mp.sqrt(a), 4*mp.sqrt(a), 1, 3, 10])
    return (mp.mpf(19)/16)**(p/2)*(a/mp.sqrt(6))**p*E
def G1D(a):
    h = lambda w: mp.re((1+1j*w)**(-0.5)*mp.exp(1j*w*a*a/16 - w*w*a*a*mp.mpf(19)/192))
    g = lambda x: mp.exp(x)**(-p)*(1 - h(mp.exp(x)))
    J = mp.quad(g, list(range(-30, 31, 1)))
    J += mp.exp(-30*p)/p
    return J/Kp
for a in (mp.mpf('0.3'), mp.mpf(1), mp.mpf(2)):
    print(a, G2D(a), G1D(a))
# |y|^p identity
y = mp.mpf('1.3')
g = lambda x: mp.exp(x)**(-p)*(1 - mp.cos(mp.exp(x)*y))
print('identity', abs(y)**p, (mp.quad(g, list(range(-30, 31, 1))) + mp.exp(-30*p)/p)/Kp)
