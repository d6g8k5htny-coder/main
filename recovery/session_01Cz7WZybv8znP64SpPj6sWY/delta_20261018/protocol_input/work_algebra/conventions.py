from fractions import Fraction as F
delta,a,b=F(1,10),F(1),F(2)
fl=[F(3,4),F(17,20),F(1),F(6,5),F(2),F(11,5)]
pl=[F(19,20),F(21,20),F(1),F(1),F(9,5),F(2)]
conv={'[a,b)':lambda x,lo,hi:lo<=x<hi,'[a,b]':lambda x,lo,hi:lo<=x<=hi,'(a,b]':lambda x,lo,hi:lo<x<=hi,'(a,b)':lambda x,lo,hi:lo<x<hi}
for name,f in conv.items():
    c=lambda xs,lo,hi: sum(f(x,lo,hi) for x in xs) if lo<hi else 0
    for k,lab in [(2,'2delta'),(1,'1delta')]:
        lo_=c(fl,a+k*delta,b-k*delta); mid=c(pl,a,b); up=c(fl,a-k*delta,b+k*delta)
        print(name,lab,lo_,mid,up,'PASS' if lo_<=mid<=up else 'FAIL')
# vacuous checks: show they hold for arbitrary values
import random
ok=all(F(x,y)*y==x for x in range(1,50) for y in range(1,50))
print('exposure tautology for all x,y:',ok)
ok=all((F(r)/576)/(F(c)/576)==F(r)/c for r in range(1,30) for c in range(1,30))
print('normalization tautology:',ok)
print('tail_square tuple check reduces to 4Q^2==(2Q)^2:', all((1,-4*Q,4*Q*Q)==(1,-4*Q,(2*Q)**2) for Q in range(1,100)))
