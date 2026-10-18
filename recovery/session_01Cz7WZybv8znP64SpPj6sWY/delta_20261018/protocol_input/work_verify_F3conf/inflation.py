# Illustrative: inflation of n/ln(2J/alpha) from common (Q,r) vs per-bin (Q_j,r_j)
# Best case: T_j=0, H=beta=grid=0, EN_j=m_j exactly. Tail t_j = E(N-Q)_+ with N ~ Poisson(m_j) (oracle tail)
# or P13 tail S2/(4Q) with S2=m^2+m (Poisson second moment).
import mpmath as mp
mp.mp.dps=30
eta=mp.mpf(1)/20; J=8
def pois_tail(m,Q):
    # E(N-Q)_+ = E N - E min(N,Q) ; E min(N,Q)= sum_{k<Q} P(N>k)
    s=mp.mpf(0); p=mp.e**(-m); cdf=p
    for k in range(0,Q):
        s+= 1-cdf
        p=p*m/(k+1); cdf+=p
    return max(m - s, mp.mpf(0))
def p13_tail(m,Q): return (m*m+m)/(4*Q)
def solve(m0,tailf):
    ms=[m0*mp.mpf(2)**(mp.mpf(2*j)/3) for j in range(J)]
    # per-bin
    per=[]
    for m in ms:
        best=mp.inf
        Qmax=int(max(60, 60*m)) if tailf is p13_tail else int(m+40*mp.sqrt(m)+60)
        step=max(1,Qmax//4000)
        for Q in range(1,Qmax,step):
            r=(eta*m-2*tailf(m,Q))/2
            if r<=0: continue
            v=(Q/r)**2/2
            if v<best: best=v
        per.append(best)
    perbin=max(per)
    # common
    bestc=mp.inf
    Qmax=int(max(60,60*ms[-1])) if tailf is p13_tail else int(ms[-1]+40*mp.sqrt(ms[-1])+60)
    step=max(1,Qmax//4000)
    for Q in range(1,Qmax,step):
        r=min((eta*m-2*tailf(m,Q))/2 for m in ms)
        if r<=0: continue
        v=(Q/r)**2/2
        if v<bestc: bestc=v
    return perbin,bestc
for name,tf in [("poisson-exact-tail",pois_tail),("P13 S2=m^2+m",p13_tail)]:
    for m0 in [0.5,1,3,10,100,1000]:
        a,b=solve(mp.mpf(m0),tf)
        print(name,"m0=",m0,"per-bin n/ln=",mp.nstr(a,6),"common n/ln=",mp.nstr(b,6),"ratio=",mp.nstr(b/a,5))
print("(m7/m0)^2=",mp.nstr(mp.mpf(2)**(mp.mpf(28)/3),6))
