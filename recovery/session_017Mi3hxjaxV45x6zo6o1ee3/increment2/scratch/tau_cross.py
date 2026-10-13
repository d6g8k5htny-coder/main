import sys; sys.argv=['x']
import pjc
from decimal import Decimal as D
import decimal
decimal.getcontext().prec = 120
worst = D(0); rows = []
for L in (None, 24, 8, '2pi', 4, 3):
    LL = 2*pjc.PI if L == '2pi' else L
    q2,q4,q6,_ = pjc.q_derivatives(LL)
    m2,m4,m6 = -q2, q4, -q6
    k = {2: m2, 4: m4-3*m2*m2, 6: m6-15*m4*m2+30*m2**3}
    a,b,c = k[2],k[4],k[6]
    for d in (2,3,4):
        for u in pjc.directions(d):
            x = [ui*ui for ui in u]
            f = 6*a**3 + 9*a*b*sum(t*t for t in x) + (c-b*b/a)*sum(t**3 for t in x)
            t2 = pjc.jet_data(u,k)['tau2']
            e = abs(f-t2)/abs(t2); worst = max(worst,e)
        rows.append((str(L), d))
    print(L, 'tau2 range check ok; e_rel max so far', '%.2e' % worst,
          ' u=e1:', '%.12f' % (6*a**3+9*a*b+c-b*b/a), ' u=diag2:', '%.12f' % (6*a**3+9*a*b/2+(c-b*b/a)/4))
print('MAX REL DEV', '%.3e' % worst)
