import sys, time
sys.argv = ["x"]
sys.path.insert(0, "<SCRATCH>/wtd4cf/coefficients/side24_d5_closed_form_20260930")
import closed_form_d4 as X
from fractions import Fraction as Fr
t0 = time.time()
m = 5
alpha, beta, PE, PR, expE, rest = X.a_integrated_terms(m)
print("a-integration done", round(time.time() - t0, 1), "s; alpha", alpha, "terms PE", len(PE), "PR", len(PR))
# variables after a-integration: (p1, p2, p3, p4); nested coordinates T = 4p1+3p2+2p3+p4, w = 3p2+2p3+p4, v = 2p3+p4, q = p4
T, w, v, q = (X.pvar(i, 4) for i in range(4))
p4 = q
p3 = X.pscale(X.padd(v, X.pscale(q, Fr(-1))), Fr(1, 2))
p2 = X.pscale(X.padd(w, X.pscale(v, Fr(-1))), Fr(1, 3))
p1 = X.pscale(X.padd(T, X.pscale(w, Fr(-1))), Fr(1, 4))
img = [p1, p2, p3, p4]
eE, eR, b = X.psubst(expE, img, 4), X.psubst(rest, img, 4), X.psubst(beta, img, 4)
print("expE:", eE); print("rest:", eR); print("beta:", b)
PE2, PR2 = X.psubst(PE, img, 4), X.psubst(PR, img, 4)
print("monomials E", len(PE2), "R", len(PR2), round(time.time() - t0, 1), "s")
kE = sorted({k[3] for k in PE2}); kR = sorted({k[3] for k in PR2})
print("q-powers E", kE, "R", kR)
print("T-powers E", sorted({k[0] for k in PE2}), "w-deg", max(k[1] for k in PE2), "v-deg", max(k[2] for k in PE2))
