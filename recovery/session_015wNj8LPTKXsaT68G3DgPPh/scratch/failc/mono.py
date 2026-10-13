from float_jfail import s0_root
import math
def H(B,T):
    s0=s0_root(B,T)
    return abs(B)**3-4*s0**3+3*B*B*s0
bad=0; worst=None
Ts=[0,1e-6,1e-3,0.01,0.1,0.5,1,2,5,10,30,100,1000]
for T in Ts:
    prev=None
    for i in range(-2000,2001):
        B=i/200.0
        h=H(B,T)
        if prev is not None and h>prev+1e-9*max(1,abs(prev)):
            bad+=1; worst=(T,B,h,prev)
        prev=h
print('violations of decreasing-in-B:',bad,worst)
# increasing in T
bad2=0
for i in range(-200,201):
    B=i/20.0; prev=None
    for T in [x/10 for x in range(0,2000)]:
        h=H(B,T)
        if prev is not None and h<prev-1e-9*max(1,abs(prev)): bad2+=1
        prev=h
print('violations of increasing-in-T:',bad2)
