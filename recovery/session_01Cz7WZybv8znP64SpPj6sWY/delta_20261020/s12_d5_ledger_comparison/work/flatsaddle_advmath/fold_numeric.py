"""Numeric sanity check of B3/B5: exact cubic model at S, third critical point at distance r^2."""
import mpmath as mp
mp.mp.dps = 50
def check(r, kap, nu, m, zs, Tttt, Ttts, Ttss):
    A = kap*r; B = r*nu; C = -m*r**2 + B**2/A          # muhat = m r^2 exactly
    ys = r**2*zs
    # f_t = A yt + B ys + 1/2(Tttt yt^2 + 2 Ttts yt ys + Ttss ys^2) = 0 -> root near -B ys/A
    yt = mp.findroot(lambda u: A*u + B*ys + (Tttt*u**2 + 2*Ttts*u*ys + Ttss*ys**2)/2, -B*ys/A)
    # f_s = B yt + C ys + 1/2(Ttts yt^2 + 2 Ttss yt ys + Tsss ys^2) = 0 -> Tsss
    Tsss = -(B*yt + C*ys + (Ttts*yt**2 + 2*Ttss*yt*ys)/2)*2/ys**2
    Htt = A + Tttt*yt + Ttts*ys; Hts = B + Ttts*yt + Ttss*ys; Hss = C + Ttss*yt + Tsss*ys
    detS = A*C - B**2; detY = Htt*Hss - Hts**2
    f = (A*yt**2 + 2*B*yt*ys + C*ys**2)/2 + (Tttt*yt**3 + 3*Ttts*yt**2*ys + 3*Ttss*yt*ys**2 + Tsss*ys**3)/6
    return detY/detS, f/(-(m*r**2)*ys**2/6), Tsss
for r in [mp.mpf('1e-2'), mp.mpf('1e-3'), mp.mpf('1e-4')]:
    print(mp.nstr(r,2), [mp.nstr(x, 8) for x in check(r, mp.mpf(1.3), mp.mpf(0.7), mp.mpf(0.5), mp.mpf(0.8),
                                                  mp.mpf(-0.4), mp.mpf(0.9), mp.mpf(1.1))])
