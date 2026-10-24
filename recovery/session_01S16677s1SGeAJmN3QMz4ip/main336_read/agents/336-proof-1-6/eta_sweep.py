# Section 4 claims across the whole interval (1): root bounds, (14), Phi(z+)>0, sign of Phi(z-), kappa>0; exact kappa via trig roots (mpmath).
import mpmath as mp
mp.mp.dps = 40
eta_end = 2/(3*mp.sqrt(3))
def roots(eta):
    # z^3 - z - eta = 0, three real roots via trigonometric formula (p=-1,q=-eta): z_k = 2 sqrt(1/3) cos( (1/3) acos( (3 eta/2) sqrt(3) ) - 2 pi k/3 )
    th = mp.acos(mp.mpf(3)*eta/2*mp.sqrt(3))/3
    zs = sorted([2*mp.sqrt(mp.mpf(1)/3)*mp.cos(th - 2*mp.pi*k/3) for k in range(3)])
    return zs
Phi = lambda z, eta: -z**4/4 + z**2/2 + eta*z
ok = True; first_neg = None
N = 2000
for i in range(1, N):
    eta = eta_end*mp.mpf(i)/N
    zm, z0, zp = roots(eta)
    for zz in (zm, z0, zp): assert abs(-zz**3 + zz + eta) < mp.mpf(10)**-30
    b1 = -1 < zm < -1/mp.sqrt(3); b2 = -1/mp.sqrt(3) < z0 < 0; b3 = zp > 1
    o = Phi(zp, eta) > Phi(zm, eta) > Phi(z0, eta); pos = Phi(zp, eta) > 0; kap = Phi(zm, eta) - Phi(z0, eta) > 0
    if not (b1 and b2 and b3 and o and pos and kap): ok = False; print("FAIL at eta=", eta, b1, b2, b3, o, pos, kap)
    if first_neg is None and Phi(zm, eta) < 0: first_neg = eta
print("all section-4 claims hold on 1999 eta samples:", ok)
print("Phi_eta(z_-) first negative at eta ~", mp.nstr(first_neg, 6), "(endpoint 2/(3sqrt3) =", mp.nstr(eta_end, 6), ")")
# locate the exact threshold Phi_eta(z_-)=0
g = lambda eta: Phi(roots(eta)[0], eta)
th = mp.findroot(g, (first_neg*mp.mpf('0.95'), first_neg))
print("threshold eta* with Phi(z_-)=0:", mp.nstr(th, 15), " z_- there:", mp.nstr(roots(th)[0], 12))
# closed-form check at eta=0.2 against instance.py numbers
zm, z0, zp = roots(mp.mpf('0.2'))
print("eta=0.2 roots:", mp.nstr(zm, 12), mp.nstr(z0, 12), mp.nstr(zp, 12), " kappa=", mp.nstr(Phi(zm, mp.mpf('0.2')) - Phi(z0, mp.mpf('0.2')), 12))
# derivative identity for (14): d/deta [Phi(z+)-Phi(z-)] = z+ - z-
eta = mp.mpf('0.2'); h = mp.mpf('1e-10')
num = (Phi(roots(eta+h)[2], eta+h) - Phi(roots(eta+h)[0], eta+h) - (Phi(roots(eta-h)[2], eta-h) - Phi(roots(eta-h)[0], eta-h)))/(2*h)
print("d/deta[Phi(z+)-Phi(z-)] numeric:", mp.nstr(num, 12), " z+-z-:", mp.nstr(zp - zm, 12))
