exec(open('t3.py').read())
import sys
r0, r1 = Fr(15,1024), Fr(1,64)
law=band_law(r0, r1)
# instrument: copy of the relevant computations
bn=law.bnorm
for n in ('v1','v2','o11','o22','o12','y111','y112','y122','y222','T','tau','nu1'):
    print(n, 'bnorm %.3g'%float(bn[n]), 'sdg %.3g'%float(law.sdg[n]))
print('S', {k: '%.3g'%float(v) for k,v in law.S.items()})
print('sB_lo', float(law.sB_lo), 'sB_hi2', float(law.sB_hi2))
Lf=law.Lfree
print('Lfree', [['%.3g'%float(Lf[i][j].mid()) for j in range(i+1)] for i in range(9)])
