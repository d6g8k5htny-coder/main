from fractions import Fraction as F
import itertools
delta,a,b=F(1,10),F(1),F(2)
fl=[F(3,4),F(17,20),F(1),F(6,5),F(2),F(11,5)]
pl=[F(19,20),F(21,20),F(1),F(1),F(9,5),F(2)]
print("pair diffs/delta:",[abs(x-y)/delta for x,y in zip(fl,pl)])
conv={'[)':lambda x,lo,hi:lo<=x<hi,'[]':lambda x,lo,hi:lo<=x<=hi,'(]':lambda x,lo,hi:lo<x<=hi,'()':lambda x,lo,hi:lo<x<hi}
def c(f,xs,lo,hi): return sum(f(x,lo,hi) for x in xs) if lo<hi else 0
print("consistent conventions, shift k*delta:")
for name,f in conv.items():
  for k in (2,1):
    lo_=c(f,fl,a+k*delta,b-k*delta); mid=c(f,pl,a,b); up=c(f,fl,a-k*delta,b+k*delta)
    print(name,k,lo_,mid,up,'PASS' if lo_<=mid<=up else 'FAIL')
print("mixed: P convention vs F convention (2delta):")
for (pn,pf),(fn,ff) in itertools.product(conv.items(),conv.items()):
    lo_=c(ff,fl,a+2*delta,b-2*delta); mid=c(pf,pl,a,b); up=c(ff,fl,a-2*delta,b+2*delta)
    if not (lo_<=mid<=up): print(" P",pn,"F",fn,lo_,mid,up,"FAIL")
# wrong guard: does check detect a>2delta violation? n/a since a fixed.
