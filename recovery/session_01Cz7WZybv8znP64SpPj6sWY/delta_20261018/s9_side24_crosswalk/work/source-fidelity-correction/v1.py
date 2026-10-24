import re, sys, json
from fractions import Fraction as F
from math import factorial
import mpmath as mp
mp.mp.dps = 450
S='<SCRATCH>/s9'
def rd(p): return open(p,encoding='utf-8').read()
def lines(p): return rd(p).split('\n')
N=S+'/named/'
# closed forms
g16=mp.gamma(mp.mpf(1)/6); g76=mp.gamma(mp.mpf(7)/6)
c059 = 2**(mp.mpf(1)/6)*3**(mp.mpf(1)/3)*(29*mp.sqrt(6)-36)*g16/(432*mp.pi**2.5)
c077 = 2**(mp.mpf(-23)/6)*3**(mp.mpf(-8)/3)*(29*mp.sqrt(6)-36)*g16*mp.pi**(-2.5)
D2 = mp.mpf(29)/6-mp.sqrt(6)
cP = g76*(mp.mpf(3)/2)**(mp.mpf(1)/3)*D2/(2*mp.sqrt(3)*mp.pi**2*mp.sqrt(mp.pi))
print('c059-cP',mp.nstr(c059-cP,5),' c077-cP',mp.nstr(c077-cP,5))
P3 = lambda L: F(-L**2*(10*L**4-147*L**2+315),105)
print('P3(24)=',P3(24), P3(24)==F(-620813376,35), F(-576*3233403,105)==P3(24))
print('P3 alt', F(-2,21)*24**6+F(7,5)*24**4-3*24**2)
q=mp.e**(-288)
R1=mp.mpf(P3(24).numerator)/P3(24).denominator*q
c24=cP*(1+R1)
def dec(x,a,b):
    s=mp.nstr(x,300,strip_zeros=False) # 0.0417...
    # get decimal digits after point
    s=mp.nstr(x, 300)
    return s
sref=mp.nstr(cP,250); s24=mp.nstr(c24,250)
fr=sref.split('.')[1]; f24=s24.split('.')[1]
print('places110-125 ref',fr[109:125]); print('places110-125 c24',f24[109:125])
fd=next(i for i in range(len(fr)) if fr[i]!=f24[i])+1
print('first differing decimal place',fd)
print('c24-cref',mp.nstr(c24-cP,8))
lo=mp.mpf('0.04177593184059834334'); hi=mp.mpf('0.04177593184059834335')
print('cref-lo',mp.nstr(cP-lo,6),'hi-cref',mp.nstr(hi-cP,6))
print('band inside', lo< cP*(1+R1-mp.mpf(10)**-180) and cP*(1+R1+mp.mpf(10)**-180)<hi)
print('R1',mp.nstr(R1,25))
# e^-288 rational enclosure N=900
Nn=900
SN=sum(F(288**n,factorial(n)) for n in range(Nn+1))
t=F(288**(Nn+1),factorial(Nn+1)); rho=F(288,Nn+2)
lowr=1/(SN+t/(1-rho)); upr=1/SN
L062=lines(N+'LS-DER-062-v1.0.txt')
def grab(lineno):
    m=re.search(r'([-+]?[0-9]\.[0-9]+e[-+]?[0-9]+)',L062[lineno-1]); return m.group(1)
e_lo=grab(217); e_hi=grab(221)
def toF(s):
    m,e=s.split('e'); return F(m.replace('.',''))*F(10)**(int(e)-(len(m.split('.')[1])))
print('e_lo printed digits', len(e_lo.split('e')[0].replace('.','').lstrip('-')))
print('printed lo <= rational lo:', toF(e_lo)<=lowr, ' printed hi >= rational hi:', toF(e_hi)>=upr)
print('rational lo - printed lo', float((lowr-toF(e_lo))/F(10)**-336), 'e-336 units ; printed hi - rational hi', float((toF(e_hi)-upr)/F(10)**-336))
print('width rational', float((upr-lowr)*F(10)**307),'e-307')
# R1 endpoints, C_rel endpoints
r1m=grab(247); r1p=grab(251); cm=grab(283); cp=grab(287)
P=P3(24)
print('R1- printed vs P*hi_rat: printed<=exact?', toF(r1m)<=P*upr, ' R1+ printed >= P*lo_rat?', toF(r1p)>=P*lowr)
print('C- = R1- - 1e-170 exactly?', toF(cm)==toF(r1m)-F(1,10**170), ' C+ = R1+ + 1e-170?', toF(cp)==toF(r1p)+F(1,10**170))
print('C width', float((toF(cp)-toF(cm))*F(10)**170))
ex_lo=P*upr; ex_hi=P*lowr
print('R1- printed - exact lower', float((toF(r1m)-ex_lo)/F(10)**-329),'e-329 ; R1+ printed - exact upper', float((toF(r1p)-ex_hi)/F(10)**-329),'e-329')
print('P*printed e_hi vs printed R1-:', float((toF(r1m)-P*toF(e_hi))/F(10)**-329),'e-329')
print('R1- digits',len(r1m.split('e')[0].replace('.','').lstrip('-')), 'ulp exp', -118-210)
import math
def digits_at(x, nsig, mode):
    # x positive Fraction; return integer mantissa with nsig digits, floor/ceil
    e=math.floor(math.log10(float(x)))
    scale=F(10)**(nsig-1-e)
    y=x*scale
    m = (y.numerator//y.denominator) if mode=='floor' else -((-y.numerator)//y.denominator)
    return m
pm=lambda s: int(s.split('e')[0].replace('.','').replace('-',''))
print('e_lo printed == floor(rational lo)?', pm(e_lo)==digits_at(lowr,211,'floor'))
print('e_hi printed == ceil(rational hi)?', pm(e_hi)==digits_at(upr,211,'ceil'), ' == floor?', pm(e_hi)==digits_at(upr,211,'floor'))
print('|R1-| printed == floor(|exact lower|)?', pm(r1m)==digits_at(-ex_lo,211,'floor'), ' ceil?', pm(r1m)==digits_at(-ex_lo,211,'ceil'))
print('|R1+| printed == floor(|exact upper|)?', pm(r1p)==digits_at(-ex_hi,211,'floor'))
