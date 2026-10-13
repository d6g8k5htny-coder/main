import sys; sys.argv=['x']
import pjc, decimal
from pjc import D, PI, jet_data, directions, q_derivatives
decimal.getcontext().prec = 120
worst = D(0); band_ok = True; floor_ok = True; n = 0
for L in (None, D(24), D(8), 2*PI, D(4), D(3), D(2)):
    q2, q4, q6, _ = q_derivatives(L); m2, m4, m6 = -q2, q4, -q6
    k = {2: m2, 4: m4-3*m2*m2, 6: m6-15*m4*m2+30*m2**3}
    Dl = m6 - m4*m4/m2; V2 = m4 - m2*m2
    lo, hi = min(Dl, 3*m2*V2, 6*m2**3), max(Dl, 3*m2*V2, 6*m2**3)
    for d in (2, 3, 4):
        for u in directions(d):
            t = [x*x for x in u]
            s3 = sum(x**3 for x in t)
            s21 = sum(t[i]**2*t[j] for i in range(d) for j in range(d) if i != j)
            s111 = sum(t[i]*t[j]*t[l] for i in range(d) for j in range(i+1, d) for l in range(j+1, d))
            f = Dl*s3 + 9*m2*V2*s21 + 36*m2**3*s111
            tau = jet_data(u, k)['tau2']
            worst = max(worst, abs(f-tau)/tau); n += 1
            band_ok &= (lo - D('1e-100') <= tau <= hi + D('1e-100'))
            floor_ok &= tau >= Dl/(d*d) - D('1e-100')
    print('L', L if L is not None else 'ref', 'Delta %.10e' % Dl, '3m2V2 %.10e' % (3*m2*V2), '6m2^3 %.10e' % (6*m2**3))
print('cases', n, 'max rel dev %.2e' % worst, 'band', band_ok, 'floor', floor_ok)
