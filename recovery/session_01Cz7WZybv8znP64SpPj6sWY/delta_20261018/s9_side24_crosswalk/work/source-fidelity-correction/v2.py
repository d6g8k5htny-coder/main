import re
from fractions import Fraction as F
import mpmath as mp
mp.mp.dps=500
S='<SCRATCH>/s9'
N=S+'/named/'
def L(p): return open(N+p,encoding='utf-8').read().split('\n')
def num(line):
    m=re.search(r'([-+]?[0-9]+\.[0-9]+(?:e[-+]?[0-9]+)?)',line); return m.group(1)
def sig(s):
    m=s.split('e')[0].lstrip('-').replace('.','').lstrip('0'); return m
def agree(a,b):
    A=sig(a);B=sig(b);n=0
    for x,y in zip(A,B):
        if x!=y: break
        n+=1
    return n,len(A),len(B)
l61=L('LS-DER-061-v1.0.txt'); l63=L('LS-DER-063-v1.0.txt'); la=L('CL-AUD-LSDER062-20260728-01.txt')
e={}
for k,ln in [('61_12',195),('61_13',204),('61_14',211),('61_15',218),('61_17',238)]: e[k]=num(l61[ln-1])
for k,ln in [('63_1',79),('63_2',86),('63_3',92)]: e[k]=num(l63[ln-1])
e['A280']=num(la[139-1]); e['A162']=num(la[45-1]); e['A12']=num(la[124-1]); e['A13']=num(la[127-1]); e['A17']=num(la[130-1])
for k,v in e.items(): print(k,'sig digits printed',len(sig(v)))
print('61_12 vs 63_1',agree(e['61_12'],e['63_1']))
print('63_1 vs A280',agree(e['63_1'],e['A280']))
print('61_17 vs 63_3',agree(e['61_17'],e['63_3']))
print('61_13 vs 63_2',agree(e['61_13'],e['63_2']))
print('A12 vs 63_1',agree(e['A12'],e['63_1']),'A13 vs 63_2',agree(e['A13'],e['63_2']),'A17 vs 63_3',agree(e['A17'],e['63_3']))
g16=mp.gamma(mp.mpf(1)/6)
c=2**(mp.mpf(-23)/6)*3**(mp.mpf(-8)/3)*(29*mp.sqrt(6)-36)*g16*mp.pi**(-2.5)
P=mp.mpf(-620813376)/35; R1=P*mp.e**(-288)
def cmpnum(s,x):
    xs=mp.nstr(x,len(sig(s))+20)
    return agree(s,xs)
print('61_14 vs closed form',cmpnum(e['61_14'],c))
print('61_15 vs c(1+R1)',cmpnum(e['61_15'],c*(1+R1)))
r63=mp.mpf(e['63_1'])
print('63_2 vs 63_1*c',cmpnum(e['63_2'],r63*c))
print('63_3 vs 63_1-R1',cmpnum(e['63_3'],r63-R1))
print('61_15-61_14 - 61_13', mp.nstr(mp.mpf(e['61_15'])-mp.mpf(e['61_14'])-mp.mpf(e['61_13']),5))
print('63_1 - R1', mp.nstr(r63-R1,10))
print('c(63_1) i.e. 63_2 - c*63_1',mp.nstr(mp.mpf(e['63_2'])-c*r63,5))
