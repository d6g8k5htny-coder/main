import time, math
exec(open('t3.py').read())
from math import sqrt
print('m3(0)', float(m3(0).lo), float(m3(0).hi), (7-4*sqrt(2))/2, float(m3(0).width()))
for (a,c) in ((1,1),(1,2),(4,0),(2,0)):
    for m in (0, Fr(1,2), 1):
        v=contact_moment(a,c,m); print(a,c,m, float(v.mid()), float(v.width()))
# floating quadrature check of contact moments
def quad(a,c,m,N=500):
    h=14.0/N; tot=0
    for i in range(N):
        l1=(i+0.5)*h-1+float(m)*0
        if l1<=0: continue
        for j in range(N):
            l2=(j+0.5)*h-1
            if l2<=l1: continue
            tot+=(l1*l2)**a*(l1+l2)**c*(l2-l1)*math.exp(-((l1-float(m))**2+(l2-float(m))**2)/4)
    return tot*h*h/(4*sqrt(2*math.pi))
for (a,c) in ((1,1),(1,2),(4,0)):
    print('quad',a,c,[round(quad(a,c,m),5) for m in (0,0.5,1)])
t=time.time(); law=band_law(Fr(1,2**30), Fr(1,2**29)); print('band_law', time.time()-t)
C=law.C
for n in ('a11','a22','a12','v1','v2','o11','o22','o12','y111','y112','y122','y222'):
    print(n, float(C[(n,n)].mid()), float(C[(n,n)].width()))
print('Cb', [[float(x.mid()) for x in row] for row in law.Cb])
print('det', float(law.detCb.mid()), 'eig', float(law.sB_lo2), float(law.sB_hi2))
print('alpha a11', law.alpha['a11'], 'beta a11', law.beta['a11'])
for n in ('v1','o11','o12','y111','T','X040','X220','X400','X202','X022'):
    print('bt', n, [float(x.mid()) for x in law.bt[n]], float(law.bnorm[n]))
print('Lfree diag', [float(law.Lfree[i][i].mid()) for i in range(9)])
print('S', {k: float(v) for k,v in law.S.items()})
print('M1', float(law.M1), 'th1', float(law.th1))
