import re, mpmath as mp
mp.mp.dps=500
S='<SCRATCH>/s9/named/'
t61=open(S+'LS-DER-061-v1.0.txt',encoding='utf-8-sig').read().splitlines()
t63=open(S+'LS-DER-063-v1.0.txt',encoding='utf-8-sig').read().splitlines()
ta=open(S+'CL-AUD-LSDER062-20260728-01.txt',encoding='utf-8-sig').read().splitlines()
def num(line):
    m=re.search(r'-?[0-9]\.[0-9]+e-[0-9]+|0\.0[0-9]+',line); return m.group(0)
def mant(s):
    return s.split('e')[0].replace('-','').replace('.','').lstrip('0')
def agree(a,b):
    a=mant(a);b=mant(b);k=0
    while k<min(len(a),len(b)) and a[k]==b[k]: k+=1
    return k,len(a),len(b)
L=lambda T,n: num(T[n-1])
e61={12:L(t61,195),13:L(t61,204),14:L(t61,211),15:L(t61,218),17:L(t61,238)}
e63={1:L(t63,79),2:L(t63,86),3:L(t63,92)}
aud280=L(ta,139); aud162=L(ta,45); aud17=L(ta,130)
print('061(12) vs 063(1)',agree(e61[12],e63[1]))
print('063(1) vs AUD l139',agree(e63[1],aud280))
print('061(12) vs AUD l45 162',agree(e61[12],aud162))
print('061(17) vs 063(3)',agree(e61[17],e63[3]))
print('AUD R1 (17) vs 063(3)',agree(aud17,e63[3]))
print('063(2) length',len(mant(e63[2])),' 063(1) length',len(mant(e63[1])))
g=mp.gamma
cref=mp.mpf(2)**(mp.mpf(1)/6)*mp.cbrt(3)*(29*mp.sqrt(6)-36)*g(mp.mpf(1)/6)/(432*mp.pi**mp.mpf(2.5))
R1=mp.mpf(-620813376)/35*mp.e**(-288)
f=lambda x,n: mp.nstr(x,n,strip_zeros=False)
print('061(14) vs cref',agree(e61[14],f(cref,300)))
print('061(15) vs cref(1+R1)',agree(e61[15],f(cref*(1+R1),300)))
r1=mp.mpf(e63[1])
print('063(3) vs 063(1)-R1',agree(e63[3],f(r1-R1,200)))
print('063(2) vs 063(1)*cref',agree(e63[2],f(r1*cref,300)))
print('061(15)-(14) - (13):',mp.nstr(mp.mpf(e61[15])-mp.mpf(e61[14])-mp.mpf(e61[13]),5))
print('063(1)-R1',mp.nstr(r1-R1,10),' |063(1)-R1|<1e-180',abs(r1-R1)<mp.mpf(10)**-180)
