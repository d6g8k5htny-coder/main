import re, sys
from mpmath import mp, mpf, sqrt, gamma, pi, exp, log10, fabs
mp.dps = 450
D=sys.argv[1]
def txt(n): return open(D+'/'+n+'.txt','rb').read().decode('utf-8-sig').replace('\r\n','\n')
def eq(t, tag):
    # find the number immediately preceding "(tag)" on same line
    m = re.findall(r'(-?\d\.\d+(?:e-\d+)?)\s*[,.]?\s*\('+re.escape(tag)+r'\)', t)
    return m
t61=txt('LS-DER-061-v1.0'); t63=txt('LS-DER-063-v1.0'); a62=txt('CL-AUD-LSDER062-20260728-01')
e12=eq(t61,'12')[0]; e13=eq(t61,'13')[0]; e14=eq(t61,'14')[0]; e15=eq(t61,'15')[0]
d1=eq(t63,'1')[0]; d2=eq(t63,'2')[0]; d3=eq(t63,'3')[0]
cinf = mpf(2)**(mpf(-23)/6)*mpf(3)**(mpf(-8)/3)*(29*sqrt(6)-36)*gamma(mpf(1)/6)*pi**(mpf(-5)/2)
def sig(s): return len(s.split('e')[0].replace('-','').replace('.','').lstrip('0'))
print('061(14) digits after point', len(e14.split('.')[1]), ' |cinf-(14)| =', mp.nstr(fabs(cinf-mpf(e14)),5))
print('061(15) digits after point', len(e15.split('.')[1]))
print('063(1) sig digits', sig(d1), '063(2) sig', sig(d2), '063(3) sig', sig(d3))
r1=mpf(d1); r2=mpf(d2); r3=mpf(d3)
print('|cinf(1+063(1)) - 061(15)| =', mp.nstr(fabs(cinf*(1+r1)-mpf(e15)),5))
print('rel |cinf*063(1) - 063(2)|/|063(2)| =', mp.nstr(fabs(cinf*r1-r2)/fabs(r2),5))
P3=mpf(-620813376)/35
print('063(1) - P3 e^-288 =', mp.nstr(r1-P3*exp(-288),25), ' vs 063(3) =', d3[:30])
print('P3(24)e^-288 =', mp.nstr(P3*exp(-288),20))
print('P3 exact:', -24**2*(10*24**4-147*24**2+315), '/105 ->', (-24**2*(10*24**4-147*24**2+315))*35 == -620813376*105)
# CL-AUD-062 strings
L=a62.split('\n')
r2s=re.findall(r'-1\.4862\d{200,}e-118', a62)
print('CL-AUD-062 280-digit strings:', [sig(x) for x in r2s], ' == 063(1):', [x==d1 for x in r2s])
for tag in ('(12)','(13)','(17)'):
    m=re.search(r'^\s*'+re.escape(tag)+r'\s+(-?\d\.\d+e-\d+)', a62, re.M)
    s=m.group(1); tgt={'(12)':d1,'(13)':d2,'(17)':d3}[tag]
    print('R1',tag,'sig',sig(s),'prefix-of-063:', tgt.split('e')[0].startswith(s.split('e')[0]) and tgt.split('e')[1]==s.split('e')[1])
x=e12.split('e')[0]; y=d1.split('e')[0]
k=next(i for i in range(min(len(x),len(y))) if x[i]!=y[i]); print('061(12) vs 063(1) first diff at significant digit', k-1)  # "-1." -> index 0='-',1='1',2='.'
# PROOF enclosure
lo,hi=mpf('0.04177593184059834334'),mpf('0.04177593184059834335')
print('c24 chain in [lo,hi]:', lo<cinf*(1+r1)<hi, ' 061(15) in:', lo<mpf(e15)<hi)
