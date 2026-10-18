import sys, re, json
from fractions import Fraction as F
import mpmath as mp
S='<SCRATCH>/s9'
mp.mp.dps=460
# P3
L=24
P3a=F(-L**2*(10*L**4-147*L**2+315),105)
P3b=F(-3*L**2)+F(7*L**4,5)-F(10*L**6,3*5*7)
print('P3(24)',P3a,P3a==P3b==F(-620813376,35))
# e^-288 enclosure, N=900
N=900
SN=F(0); t=F(1)
for n in range(N+1):
    if n>0: t=t*288/n
    SN+=t
tN1=t*288/(N+1)
rho=F(288,N+2)
lo=1/(SN+tN1/(1-rho)); hi=1/SN
e=mp.e**(-288)
print('lo<e<hi',mp.mpf(lo.numerator)/lo.denominator< e < mp.mpf(hi.numerator)/hi.denominator)
txt=open(S+'/named/LS-DER-062-v1.0.txt',encoding='utf-8-sig').read()
nums=re.findall(r'8\.378942533819[0-9]+e-126',txt)
print('printed e-288 endpoints',len(nums))
def sig(a,b):
    # count agreeing significant digits of mp numbers via strings
    sa=mp.nstr(a,300,strip_zeros=False).replace('-','').replace('.','')
    sb=mp.nstr(b,300,strip_zeros=False).replace('-','').replace('.','')
    sa=sa.split('e')[0].lstrip('0'); sb=sb.split('e')[0].lstrip('0')
    k=0
    while k<min(len(sa),len(sb)) and sa[k]==sb[k]: k+=1
    return k
def strsig(s, x):
    m=s.split('e')[0].replace('-','').replace('.','').lstrip('0')
    xs=mp.nstr(abs(x),len(m)+30,strip_zeros=False).split('e')[0].replace('.','').lstrip('0')
    k=0
    while k<len(m) and m[k]==xs[k]: k+=1
    return k,len(m)
lof=mp.mpf(lo.numerator)/lo.denominator; hif=mp.mpf(hi.numerator)/hi.denominator
print('e lo printed vs exact', strsig(nums[0],lof), ' hi', strsig(nums[1],hif))
print('width', mp.nstr(hif-lof,6))
# R1 interval, C_rel 1e-170 and new 1e-180 endpoints
R1lo=P3a*hi; R1hi=P3a*lo   # negative multiplier reverses
R1=mp.mpf(P3a.numerator)/P3a.denominator*e
print('R1',mp.nstr(R1,40))
# check printed R1-,R1+, C-,C+
pr=re.findall(r'-1\.4862170290658[0-9]+e-118',txt)
print('n printed -1.486..e-118 strings in 062:',len(pr))
for s in pr: print(' ', s[:30],'... len',len(s.split('e')[0])-3)
R1lof=mp.mpf(R1lo.numerator)/R1lo.denominator; R1hif=mp.mpf(R1hi.numerator)/R1hi.denominator
print('R1- printed vs exact', strsig(pr[0],R1lof),' R1+ ',strsig(pr[1],R1hif))
Cm=R1lo-F(1,10**170); Cp=R1hi+F(1,10**170)
print('C- printed', strsig(pr[2],mp.mpf(Cm.numerator)/Cm.denominator),' C+',strsig(pr[3],mp.mpf(Cp.numerator)/Cp.denominator))
# c_ref closed forms
g=mp.gamma
cref=g(mp.mpf(7)/6)*mp.cbrt(mp.mpf(3)/2)*(mp.mpf(29)/6-mp.sqrt(6))/(2*mp.sqrt(3)*mp.pi**2*mp.sqrt(mp.pi))
c059=mp.mpf(2)**(mp.mpf(1)/6)*mp.cbrt(3)*(29*mp.sqrt(6)-36)*g(mp.mpf(1)/6)/(432*mp.pi**mp.mpf(2.5))
c077=mp.mpf(2)**(mp.mpf(-23)/6)*mp.mpf(3)**(mp.mpf(-8)/3)*(29*mp.sqrt(6)-36)*g(mp.mpf(1)/6)*mp.pi**(-mp.mpf(5)/2)
print('cref-c059',mp.nstr(cref-c059,5),'cref-c077',mp.nstr(cref-c077,5))
c24=cref*(1+R1)
s1=mp.nstr(cref,200,strip_zeros=False); s2=mp.nstr(c24,200,strip_zeros=False)
k=0
while s1[k]==s2[k]: k+=1
print('first differing char index',k,'-> decimal place',k-1, s1[k-3:k+5], s2[k-3:k+5])
lo3=mp.mpf('0.04177593184059834334'); hi3=mp.mpf('0.04177593184059834335')
print('cref-lo',mp.nstr(cref-lo3,6),'hi-cref',mp.nstr(hi3-cref,6))
print('relative margins down/up', mp.nstr((cref-lo3)/cref,6), mp.nstr((hi3-cref)/cref,6))
for eps in [mp.mpf(10)**-180, -mp.mpf(10)**-180]:
    print('c24 band inside', lo3< cref*(1+R1+eps) < hi3)
print('|c24-cref|',mp.nstr(abs(c24-cref),6),' ratio to width',mp.nstr(abs(c24-cref)/mp.mpf('1e-20'),4))
# PROOF E and ledger
E=F(1458*(76*24**6+15),10**125)
print('E',E.numerator,'e-125 ->',float(E))
enc=json.load(open(S+'/side24/ENCLOSURE.json'))
db=F(enc['image_ledger']['derivative_bound']); cb=F(enc['image_ledger']['covariance_relative_bound'])
print('derivative_bound == E', db==E, ' cov bound/E', cb/E, float(cb))
print('60E<1e-108', 60*E<F(1,10**108), '32e-108<1e-106', True)
print('e^-288<1e-125', hi< F(1,10**125))
# jet: delta m6 * q (1-D), second-order remainder 1-D and 3-D max
def He(n,x):
    return [1,0,x**2-1,0,x**4-6*x**2+3,0,x**6-15*x**4+45*x**2-15][n]
q=mp.e**(-mp.mpf(288))
def kn(n):
    num=sum(He(n,mp.mpf(24*m))*mp.e**(-mp.mpf(288)*m*m) for m in range(-4,5))
    den=sum(mp.e**(-mp.mpf(288)*m*m) for m in range(-4,5))
    return num/den
k0={n:He(n,0) for n in (0,2,4,6)}
kex={n:kn(n) for n in (0,2,4,6)}
k1={n:k0[n]+2*(He(n,mp.mpf(24))-He(n,0))*q for n in (0,2,4,6)}
for n in (2,4,6): print('1D n',n,'delta',mp.nstr(kex[n]-k0[n],8),' R1D',mp.nstr(kex[n]-k1[n],6))
# 3-D max over multi-indices with even entries summing<=6 of exact product minus first-order product
import itertools
best=(0,None)
for a in itertools.product((0,2,4,6),repeat=3):
    if sum(a)>6 or sum(a)==0: continue
    ex=kex[a[0]]*kex[a[1]]*kex[a[2]]
    p0=k0[a[0]]*k0[a[1]]*k0[a[2]]
    fo=p0+sum((k1[a[i]]-k0[a[i]])*k0[a[(i+1)%3]]*k0[a[(i+2)%3]] for i in range(3))
    r=abs(ex-fo)
    if r>best[0]: best=(r,a)
print('3D R_jet max',mp.nstr(best[0],6),best[1])
