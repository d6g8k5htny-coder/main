import math, sys
def hermgauss(n):
    # Golub-Welsch via simple Newton on Hermite polynomials (physicists'), returns nodes, weights for int e^{-x^2} f
    xs=[];ws=[]
    for i in range(1,n+1):
        # initial guess
        if i==1: z=math.sqrt(2*n+1)-1.85575*(2*n+1)**(-0.16667)
        elif i==2: z=z-1.14*n**0.426/z
        elif i==3: z=1.86*z-0.86*xs[0]
        elif i==4: z=1.91*z-0.91*xs[1]
        else: z=2.0*z-xs[i-3]
        for _ in range(100):
            p1=math.pi**-0.25; p2=0.0
            for j in range(1,n+1):
                p3=p2; p2=p1
                p1=z*math.sqrt(2.0/j)*p2-math.sqrt((j-1)/j)*p3
            pp=math.sqrt(2.0*n)*p2
            z1=z; z=z1-p1/pp
            if abs(z-z1)<1e-15: break
        xs.append(z); ws.append(2.0/(pp*pp))
    X=[];W=[]
    for x,w in zip(xs,ws):
        X+= [x,-x]; W+=[w,w]
    if n%2==1:
        pass
    return X[:n],W[:n]
def s0_root(B,T):
    # root of (s-B)^2(-2s-B)=T on s<=min(B,-|B|/2), decreasing branch; use bisection on x=-s
    hi_s=min(B,-abs(B)/2)
    if T<=0: return hi_s
    lo_s=hi_s-1.0
    f=lambda s:(s-B)**2*(-2*s-B)-T
    while f(lo_s)<0: lo_s=hi_s-2*(hi_s-lo_s)
    a,b=lo_s,hi_s
    for _ in range(80):
        m=(a+b)/2
        if f(m)>0: a=m
        else: b=m
    return (a+b)/2
def Ifail(a,beta,c,k):
    B=beta-a*a/(12*k); D=(c-a*beta/(4*k)+a**3/(72*k*k))/2
    s0=s0_root(B,12*k*D*D)
    return 3*k*k*(abs(B)**3-4*s0**3+3*B*B*s0)
def J(k,n=40):
    X,W=hermgauss(n)
    # N(0,v): E f = sum w/sqrt(pi) f(sqrt(2v) x)
    tot=0.0
    for xa,wa in zip(X,W):
        a=math.sqrt(4)*xa
        for xb,wb in zip(X,W):
            beta=math.sqrt(4)*xb
            for xc,wc in zip(X,W):
                c=math.sqrt(12)*xc
                tot+=wa*wb*wc*Ifail(a,beta,c,k)
    return tot/math.pi**1.5
if __name__ == "__main__":
  for k in (0.5,1.0,2.0):
    for n in (20,30):
        print(k,n,J(k,n)); sys.stdout.flush()
