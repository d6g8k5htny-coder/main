from fractions import Fraction as F
import json

def check(ok, label):
    if not ok:
        raise ValueError(label)

def A(u):
    return 2*(u+F(1,2))**2*(u-1)

# R=0: tau=(2/3+|c|/(24*kappa))/sqrt(3).
psi, c, eta = F(1), F(49,100), F(1,1000)
ks, km = (psi+c)/48, (psi-c)/48
ts2 = (F(2,3)+abs(c)/(24*ks))**2/3
tm2 = (F(2,3)+abs(c)/(24*km))**2/3
ms, em = 2/(125*ts2), min(1/(8*tm2), F(1))
mu = A(-psi/(2*c))+1
den = 1-F(256,125)*c/psi
bad = F(277,250)/den
fm = (1+eta)/(psi-c)
check(0 < eta < min(-mu, ms, em), "H for denominator example")
check(den == -F(11,3125) and bad < 0 < psi*fm, "unsafe reciprocal")
a, b, d = F(599,250), F(277,250), F(256,125)
K, w0 = b+d*a, a/(b+d*a)
check(K == F(187969,31250) and K < F(121,20), "piecewise constant")
check(0 < w0 < 1/d and b/(1-d*w0) == a/w0 == K, "branch intersection")
for w in (F(1,100), F(1,4), w0, F(49,100)):
    bound = b/(1-d*w) if w <= w0 else a/w
    check(bound <= K, "representative repaired branch")
check(144*F(121,20) < F(148,5)**2, "29.6 height")
check(48*F(121,20) < F(171,10)**2, "17.1 height")

# c=0, psi=1: tau=(4+R)/(6*sqrt(3)).
R = F(399,100)
m = 16/R**2-1
eta2 = m/2
ts2 = (4+R)**2/108
ms2, em2 = 2/(125*ts2), min(1/(8*ts2), F(1))
py, level = -(1+m), -1-eta2
check(0 < eta2 < m < min(ms2, em2), "H for excluded saddle")
check(py < level, "saddle outside trap")
# The universal root/limit claim is analytic, not proved by these endpoints.
q = lambda t: 3*t*t-2*t*t*t
check(q(F(0)) == 0 and q(F(1)) == 1, "sharpness endpoints")
result = {
    "denominator": str(den), "invalid_bound": str(bad),
    "mu": str(mu), "mS": str(ms), "epsilonM": str(em),
    "psiF_at_M": str(psi*fm), "K": str(K), "omega0": str(w0),
    "sharpness_m": str(m), "sharpness_eta": str(eta2),
    "sharpness_mS": str(ms2), "sharpness_epsilonM": str(em2),
    "P_saddle": str(py), "level": str(level),
    "scope": "exact examples and coefficient checks; universal proofs analytic"
}
print(json.dumps(result, sort_keys=True, separators=(",",":")))
