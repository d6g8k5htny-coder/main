# Independent check of nu^F_far(ell) = E[A^-1 nu^H_far(ell/A)] with A = a R, R ~ alpha r^(alpha-1) on (0,1),
# a = sqrt((alpha+2)/alpha). Toy far density nu^H_far = 1 on (0,1], 0 elsewhere. Compare the direct
# amplitude-mixture quadrature against the closed Mellin form alpha a^-alpha ell^(alpha-1) int_{ell/a}^inf s^-alpha nu ds.
import math
def a_of(al): return math.sqrt((al+2)/al)
def nu(s): return 1.0 if 0 < s <= 1 else 0.0
def mixture(al, ell, n=400000):
    a = a_of(al)
    # integrate over u in (0,a): density alpha a^-alpha u^(alpha-1); substitute u = a v^(1/alpha), v~U(0,1) -> exact pushforward
    tot = 0.0
    for i in range(n):
        v = (i+0.5)/n
        u = a*v**(1.0/al)
        tot += nu(ell/u)/u
    return tot/n
def mellin(al, ell):
    a = a_of(al)
    lo = ell/a
    if lo >= 1: return 0.0
    if abs(al-1) < 1e-12: I = math.log(1/lo)
    else: I = (1 - lo**(1-al))/(1-al)
    return al*a**(-al)*ell**(al-1)*I
for al in (0.5, 2/3, 0.8, 1.0, 1.5):
    print("alpha=%.3f a=%.4f" % (al, a_of(al)))
    for ell in (1e-2, 1e-4, 1e-6, 1e-8):
        m = mellin(al, ell)
        q = mixture(al, ell, 200000) if ell >= 1e-4 else float('nan')
        print("  ell=%g  mellin=%.6g  quad=%.6g  ell^(1/3)*nu=%.6g" % (ell, m, q, ell**(1/3)*m))
print("predicted alpha=2/3 constant 2^(1/3) =", 2**(1/3), " alpha a^-alpha M =", (2/3)*2**(-2/3)*3)
print("alpha>1 bound alpha*C/(a(alpha-1)) at alpha=1.5, C=1:", 1.5/(a_of(1.5)*0.5), " = E[A^-1] check:", 1.5*a_of(1.5)**(-1)/(0.5))
