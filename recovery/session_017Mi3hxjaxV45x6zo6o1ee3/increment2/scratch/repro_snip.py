# save the merged checker as pjc.py next to this file
import sys; sys.argv = ['x']
import pjc
from pjc import D, PI, jet_data, q_derivatives, coefficient_d2, stirling_gamma
def kum(L):
    q2, q4, q6, _ = q_derivatives(L); m2, m4, m6 = -q2, q4, -q6
    return {2: m2, 4: m4 - 3*m2*m2, 6: m6 - 15*m4*m2 + 30*m2**3}
def integrand(u, k):
    j = jet_data(u, k)
    return j['tau2']**(D(2)/3) * j['S_AgV'][0][0] / 2 / (4*PI*PI*(j['det_G']*j['det_V']).sqrt())
pref = stirling_gamma(D(7)/6) / (D(24)**(D(1)/3) * PI.sqrt())
cref = coefficient_d2(kum(None), 4)
for L in (D(4), D(3), D(2)):
    k = kum(L); full = coefficient_d2(k, 4096)
    s = 1/D(2).sqrt()
    print(L, full/cref, pref*2*PI*integrand([D(1), D(0)], k)/full,
          pref*2*PI*integrand([s, s], k)/full, coefficient_d2({2: k[2], 4: D(0), 6: D(0)}, 16)/full)
