from fractions import Fraction as Fr
import theorem_a as ta
bl = ta.bands()
law = ta.band_law(*bl[0])
res = ta.box_bound(law, (Fr(0), Fr(1, 64)), (Fr(63, 32), Fr(2)), bl[0][0], bl[0][1], last_band=True)
for k in ('qA2', 'qfar', 'qx', 'tail', 'zt'):
    print(k, float(res[k]))
print('bw', {n: float(ta.absup(law.bw[n])) for n in law.bw})
print('S', {n: float(law.S[n]) for n in law.S})
