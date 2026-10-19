# Sample check of the six §6 conditions on the closed box |alpha-6|,|beta-22|,|q-24| <= eps (11^3 grid; a sample, not a proof).
import itertools, numpy as np
def check(al,be,q):
    r=np.roots([q/24,-6.0,be/2,-al])           # P'(xi)/xi
    if np.max(abs(r.imag))>1e-12: return ['complex root']
    r=np.sort(r.real); P=lambda x:-al*x**2/2+be*x**3/6-1.5*x**4+q*x**5/120
    bad=[]
    if not (0.75<r[0]<1.25): bad.append('xi1')
    if not (1.75<r[1]<2.25): bad.append('xi2')
    if not (2.75<r[2]<3.25): bad.append('xi3')
    gaps=np.diff([P(r[2]),P(r[0]),P(r[1]),0.0])
    if min(gaps)<=1/8: bad.append('gap<=1/8')
    if not 0.5<-P(r[2])<1.5: bad.append('L0')
    return bad
for eps in (0.01,0.015,0.02,0.05):
    g=np.linspace(-eps,eps,11); fails=set()
    for a,b,c in itertools.product(g,g,g): fails.update(check(6+a,22+b,24+c))
    print(f'eps={eps}: 1331 grid points, failing conditions: {sorted(fails) if fails else "none"}')
