import re,sys
from mpmath import mp,mpf,sqrt,gamma,pi,exp,nstr,log10,fabs
mp.dps=450
D=sys.argv[1]
def txt(n): return open(D+'/'+n+'.txt','rb').read().decode('utf-8-sig').replace('\r\n','\n')
t61=txt('LS-DER-061-v1.0'); t63=txt('LS-DER-063-v1.0')
def grab(t,tag):
    m=re.search(r'=\s*(-?[0-9.]+(?:e-?\d+)?)[.,]?\s*\('+re.escape(tag)+r'\)',t); return m.group(1).rstrip('.')
s12=grab(t61,'12'); s13=grab(t61,'13'); s14=grab(t61,'14'); s15=grab(t61,'15')
r1=re.search(r'=(-1\.4862[0-9]+e-118)',t63).group(1)
a2=re.search(r'=(-6\.2088[0-9]+e-120)',t63).group(1)
r3=re.search(r'(-7\.5200[0-9]+e-237)',t63).group(1)
cinf=mpf(2)**(mpf(-23)/6)*mpf(3)**(mpf(-8)/3)*(29*sqrt(6)-36)*gamma(mpf(1)/6)*pi**(mpf(-5)/2)
# PROOF formula (1) at d=3
D2=mpf(29)/6-sqrt(6)
cref=gamma(mpf(7)/6)*(mpf(3)/2)**(mpf(1)/3)*D2/(2*sqrt(3)*pi**2*sqrt(pi))
# LS-DER-055 (28)
c55=mpf(2)**(mpf(1)/6)*mpf(3)**(mpf(1)/3)*(29*sqrt(6)-36)*gamma(mpf(1)/6)/(432*pi**(mpf(5)/2))
print('decimals in (14),(15):',len(s14.split('.')[1]),len(s15.split('.')[1]))
print('|cinf-cref|',nstr(fabs(cinf-cref),3),' |cinf-c55|',nstr(fabs(cinf-c55),3))
print('|cinf-(14)|',nstr(fabs(cinf-mpf(s14)),3))
c24=cinf*(1+mpf(r1))
print('|cinf(1+063(1))-(15)|',nstr(fabs(c24-mpf(s15)),3))
print('rel |cinf*063(1)-063(2)|/|063(2)|',nstr(fabs(cinf*mpf(r1)-mpf(a2))/fabs(mpf(a2)),3))
P3=mpf(-620813376)/35
print('P3 e^-288 =',nstr(P3*exp(-288),25))
print('063(1)-P3e^-288 =',nstr(mpf(r1)-P3*exp(-288),25),' printed (3):',r3[:25])
lo,hi=mpf('0.04177593184059834334'),mpf('0.04177593184059834335')
print('c24 in PROOF interval:',lo<c24<hi,' (15) in interval:',lo<mpf(s15)<hi, ' cref in:',lo<cref<hi)
print('margins: c24-lo',nstr(c24-lo,5),' hi-c24',nstr(hi-c24,5))
print('cref 30 digits',nstr(cref,30))
# 061(12) vs 063(1) digit divergence
x=s12.split('e')[0].lstrip('-').replace('.',''); y=r1.split('e')[0].lstrip('-').replace('.','')
k=next(i for i in range(min(len(x),len(y))) if x[i]!=y[i]); print('061(12) vs 063(1) first differing significant digit:',k+1)
print('061(12) len sig',len(x),' 063(1) len sig',len(y))
# second-shell estimate: next images at |n|^2=2 (12 vectors, weight e^{-L^2}) -> order L^12 e^{-576}
print('e^-576*24^12 ~',nstr(exp(-576)*mpf(24)**12,5), ' (3)/that',nstr(mpf(r3)/(exp(-576)*mpf(24)**12),5))
