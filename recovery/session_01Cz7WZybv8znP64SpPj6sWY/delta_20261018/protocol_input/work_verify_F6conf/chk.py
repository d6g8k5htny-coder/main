from fractions import Fraction as F
import random, math
random.seed(6)
bad=0
for trial in range(5000):
    Q=random.randint(1,12)
    k=random.randint(1,8)
    vals=[random.randint(0,60) for _ in range(k)]
    w=[random.randint(1,9) for _ in range(k)]; s=sum(w); p=[F(x,s) for x in w]
    EN=sum(pi*x for pi,x in zip(p,vals))
    Ec=sum(pi*min(x,Q) for pi,x in zip(p,vals))
    T=sum(pi*max(x-Q,0) for pi,x in zip(p,vals))
    S2=sum(pi*x*x for pi,x in zip(p,vals))
    if EN!=Ec+T: bad+=1
    if EN> Q+T: bad+=1
    if EN*EN> S2: bad+=1
    if T> S2/(4*Q): bad+=1
print("violations",bad)
# heavy-tail truncation: P(N=m^2)=1/m-ish  -> E N diverges while Ec<=Q; T absorbs everything
Q=5
for M in (10,100,1000,10000):
    # P(N=2^k)=2^-k, k=1..M-1 normalized remainder at 0 -> E N = M-1 partial sums
    EN=sum(F(1,2**k)*2**k for k in range(1,M if M<60 else 60))
    print(M, float(EN))
