# Float prototype: c1 in d = 2 from the reduction (f4, A, gamma independent given U_0 = (b,0,0,0,0,0)).
import mpmath as mp
mp.mp.dps = 20
p = mp.mpf(7)/4
def M(t):
    # E|Z + t|^p, Z ~ N(0,1)
    return 2**(p/2)*mp.gamma((p+1)/2)/mp.sqrt(mp.pi)*mp.hyp1f1(-p/2, mp.mpf(1)/2, -t*t/2)
def inner(b):
    # E[|Y|^{7/4} |A|^{1/4} 1{A<0}], A = -a, a>0; f4 ~ N(-3b,24) -> Y ~ N(mu, sig^2), mu = (a b - g^2)/4, sig = a/sqrt6
    def fa(a):
        dens_a = mp.npdf(-a, -b, mp.sqrt(2))
        def fg(g):
            mu = (a*b - g*g)/4
            sig = a/mp.sqrt(6)
            return mp.npdf(g, 0, mp.sqrt(2)) * sig**p * M(mu/sig)
        Eg = 2*mp.quad(fg, [0, 1, 3, 8])
        return dens_a * a**(mp.mpf(1)/4) * Eg
    return mp.quad(fa, [0, 0.5, 2, 5, 12 + abs(b)])
def pi0(b):
    return (2*mp.pi)**-3 * 12**-0.5 * mp.exp(-3*b*b/4)
I = mp.quad(lambda b: pi0(b)*inner(b), [-8, -3, 0, 3, 8])
c1 = -(mp.mpf(192)/7) * 2**0.25 * 2*mp.pi * I
print('I =', I, ' c1(d=2) =', c1)
