from probe_m3 import *
from fractions import Fraction as Fr
r0=R3R(0)
def boxval(blo,bhi):
    return exp(-blo*blo/4)*I(3,bhi)/(4*sqrt(2*pi))/m3(blo)/r0
for cand in ([0,1/1024,1/256,1/64,1/16,1/4,1],
             [0,1/512,3/512,3/256,3/128,3/64,3/32,3/16,3/8,3/4,1],
             [0,1/512,1/256,1/128,1/64,1/32,1/16,1/8,1/4,1/2,1],
             [0,1/1024,3/1024,3/512,3/256,3/128,3/64,3/32,3/16,3/8,3/4,1]):
    vals=[boxval(cand[i],cand[i+1]) for i in range(len(cand)-1)]
    print(len(vals), ['%.4f'%v for v in vals])
