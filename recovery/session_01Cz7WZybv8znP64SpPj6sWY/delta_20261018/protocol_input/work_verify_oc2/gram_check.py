"""OC-2 verification: (1) full-spectrum Lagrange/Gram double sum for a*nu3=ac-b^2 at L=24,
truncated |m|<=M (all terms >=0, so a certified-style lower bound), vs true value;
(2) Poisson cumulant closed forms kappa4=2 e^{-L^2/2} L^4, kappa6=-2 e^{-L^2/2} L^6;
(3) conditioning of Gamma_e, detB, V at L=24 from raw a,b,c. Analysis only."""
import mpmath as mp
mp.mp.dps = 300
L = mp.mpf(24); h = 2*mp.pi/L; al = h*h/2
Mfull = 200
ms = range(-Mfull, Mfull+1)
z = mp.fsum(mp.e**(-al*m*m) for m in ms)
p = {m: mp.e**(-al*m*m)/z for m in ms}
a = mp.fsum(p[m]*(h*m)**2 for m in ms); b = mp.fsum(p[m]*(h*m)**4 for m in ms); c = mp.fsum(p[m]*(h*m)**6 for m in ms)
true = a*c - b*b
print("a,b,c", mp.nstr(a,20), mp.nstr(b,20), mp.nstr(c,20), " ac-b^2", mp.nstr(true, 20))
for M in (8, 16, 24, 40, 64):
    # p restricted but with the FULL z normalizer: lower bound since all terms nonnegative
    qs = [(p[m], (h*m)**2) for m in range(-M, M+1)]
    s = mp.fsum(pi*pj*qi*qj*(qj-qi)**2 for (pi,qi) in qs for (pj,qj) in qs)/2
    print("M=%d Gram double-sum lower bound %s  rel deficit %s" % (M, mp.nstr(s, 15), mp.nstr((true-s)/true, 5)))
k4 = b-3*a*a; k6 = c-15*a*b+30*a**3
e1 = mp.e**(-L*L/2)
print("k4", mp.nstr(k4,12), "closed", mp.nstr(2*e1*L**4,12)); print("k6", mp.nstr(k6,12), "closed", mp.nstr(-2*e1*L**6,12))
# conditioning at theta=0.3 of Gamma_e computed from raw beta form
th = mp.mpf('0.3'); e = (mp.cos(th), mp.sin(th)); v = (-e[1], e[0])
beta = b-a*a
B = [[a*a+beta*e[0]**2, 2*a*a*e[0]*e[1]], [2*a*a*e[0]*e[1], a*a+beta*e[1]**2]]
detB = B[0][0]*B[1][1]-B[0][1]**2
print("detB", mp.nstr(detB, 20), " 3a^4", mp.nstr(3*a**4, 20), " diff", mp.nstr(detB-3*a**4, 5))
