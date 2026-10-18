import math
eta=0.05
def kl(p,q):
    s=0.0
    if p>0: s+=p*math.log(p/q)
    if p<1: s+=(1-p)*math.log((1-p)/(1-q))
    return s
def coeff(mu_frac):
    best=(math.inf,None)
    m=1.0; A=(1-eta)**2
    for i in range(200,20000):
        Q=i/100.0
        r=(eta*m - A/(2*Q))/2
        if r<=0: continue
        p=mu_frac*m/Q
        k=min(kl(p,min(p+r/Q,1-1e-15)), kl(p,max(p-r/Q,1e-15)))
        n=1.0/k
        if n<best[0]: best=(n,Q)
    return best
for mf in (1-eta,1.0):
    print("mu/m",mf,"KL route n/ln >=",coeff(mf))
print("Gaussian approx 27(1-eta)^2/eta^3 =",27*(1-eta)**2/eta**3)
