"""Check ORIGINAL K_L (spectral cosine sum, full lattice incl. zero mode) equals side24's
normalized periodized Gaussian, and side24 closed form c_2,ref equals ORIGINAL (32)."""
import mpmath as mp
mp.mp.dps = 60
def K_spec(L, x):
    h = 2*mp.pi/L
    one = lambda xi: mp.fsum(mp.e**(-(h*m)**2/2)*mp.cos(h*m*xi) for m in range(-400, 401))
    z = mp.fsum(mp.e**(-(h*m)**2/2) for m in range(-400, 401))
    return mp.fprod(one(xi)/z for xi in x)
def K_per(L, x):
    one = lambda xi: mp.fsum(mp.e**(-(xi+n*L)**2/2) for n in range(-30, 31))
    s = mp.fsum(mp.e**(-(n*L)**2/2) for n in range(-30, 31))
    return mp.fprod(one(xi)/s for xi in x)
for L, x in [(3, (mp.mpf('0.7'), mp.mpf('1.1'))), (24, (mp.mpf('11.3'), mp.mpf('5.2'))), (2, (mp.mpf('0.9'), mp.mpf('0.95')))]:
    a, b = K_spec(mp.mpf(L), x), K_per(mp.mpf(L), x)
    print("L=%s K_spec=%s K_per=%s reldiff=%s" % (L, mp.nstr(a, 20), mp.nstr(b, 20), mp.nstr(abs(a-b)/abs(b), 3)))
g = mp.gamma(mp.mpf(7)/6)
side24 = g*mp.cbrt(mp.mpf(3)/2)*(mp.mpf(4)/3)/(2*mp.sqrt(3)*mp.pi*mp.sqrt(mp.pi))
orig32 = 2/(3*mp.pi*mp.sqrt(3))*mp.cbrt(mp.mpf(3)/2)*g/mp.sqrt(mp.pi)
print("side24 c2ref", mp.nstr(side24, 40)); print("ORIG (32)   ", mp.nstr(orig32, 40))
print("legacy literal minus c2inf:", mp.nstr(mp.mpf('0.07340691930603427103') - orig32, 5))
print("side24 upper minus c2inf:", mp.nstr(mp.mpf('0.07340691930603427104') - orig32, 5))
print("first bin per-field leading count coefficient 576*C*(3/2)(2^(2/3)-1) =", mp.nstr(576*orig32*mp.mpf(3)/2*(mp.cbrt(4)-1), 10))
