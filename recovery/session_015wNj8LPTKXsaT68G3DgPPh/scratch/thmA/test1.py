import time
from fractions import Fraction as Fr
import theorem_a as ta
t0 = time.time()
law = ta.band_law(Fr(1, 1024), Fr(1, 512))
print('band law time', round(time.time() - t0, 1))
for n in ('w', 'v', 'om', 'Y', 'T', 'X40', 'X04'):
    print(n, 'alpha', float(law.alpha[n].mid()), 'beta', float(law.beta[n].mid()), 'C', float(law.C[(n, n)].lo), float(law.C[(n, n)].hi), 'width', float(law.C[(n, n)].width()))
print('bw', {n: (float(law.bw[n].lo), float(law.bw[n].hi)) for n in law.bw})
print('L3 diag', [(float(law.L3[i][i].lo), float(law.L3[i][i].hi)) for i in range(3)])
print('S', {n: float(law.S[n]) for n in law.S})
print('energy coeffs', float(law.e_bb.mid()), float(law.e_bk.mid()), float(law.e_kk.mid()), 'widths', float(law.e_kk.width()))
t0 = time.time()
res = ta.box_bound(law, (Fr(0), Fr(1, 64)), (Fr(2) - Fr(1, 512), Fr(2)), Fr(1, 1024), Fr(1, 512))
print('box time', round(time.time() - t0, 2))
print({k: float(v) for k, v in res.items()})
print('c_G(0,2) = 5324360.44 ; ratio', float(res['total']) / 5324360.44)
