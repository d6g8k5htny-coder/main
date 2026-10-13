import math, heapq, time, sys
SQ2PI=math.sqrt(2*math.pi)
def Phi(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def xroot(B,T):
    lo=max(-B,B/2)
    if T<=0: return lo
    f=lambda x:2*x**3+3*B*x*x-B**3-T
    # Newton from a safe start
    x=max(lo+1e-300, (T/2)**(1/3) if B>=0 else lo+math.sqrt(T/(3*abs(B))) if abs(B)>0 else (T/2)**(1/3))
    x=max(x,lo)
    for _ in range(60):
        fx=f(x); d=6*x*(x+B)
        if d<=0: break
        nx=x-fx/d
        if nx<=lo: nx=(x+lo)/2
        if abs(nx-x)<=1e-15*max(1,abs(x)): x=nx; break
        x=nx
    return x
def Vx(B,x): return -3*B*(2*x-B)*(x+B)
def V(B,T): return Vx(B,xroot(B,T))
def moments(x0,x1,s,nmax):
    # unnormalized int_{x0}^{x1} x^n phi_s(x) dx
    p0=math.exp(-x0*x0/(2*s*s))/(s*SQ2PI); p1=math.exp(-x1*x1/(2*s*s))/(s*SQ2PI)
    M=[Phi(x1/s)-Phi(x0/s), s*s*(p0-p1)]
    for n in range(2,nmax+1):
        M.append((n-1)*s*s*M[n-2]+s*s*(x0**(n-1)*p0-x1**(n-1)*p1))
    return M
def d0(a,b): return -a*b/4+a**3/72
def d0range(a0,a1,b0,b1):
    cand=[d0(a,b) for a in (a0,a1) for b in (b0,b1)]
    for b in (b0,b1):
        if b>0:
            r=math.sqrt(6*b)
            for a in (r,-r):
                if a0<a<a1: cand.append(d0(a,b))
    return min(cand),max(cand)
def imul(a,b):
    p=[a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]]; return (min(p),max(p))
def iadd(a,b): return (a[0]+b[0],a[1]+b[1])
def isub(a,b): return (a[0]-b[1],a[1]-b[0])
def idiv(a,b):
    assert b[0]>0 or b[1]<0
    return imul(a,(1/b[1],1/b[0]))
def isc(c,a): return (c*a[0],c*a[1]) if c>=0 else (c*a[1],c*a[0])
def ipow(a,n):
    if n%2==1: return (a[0]**n,a[1]**n)
    lo=0.0 if a[0]<=0<=a[1] else min(abs(a[0]),abs(a[1]))**n
    return (lo,max(abs(a[0]),abs(a[1]))**n)
def iabs_max(a): return max(abs(a[0]),abs(a[1]))
def box_integral(box,sd,cache):
    (a0,a1),(b0,b1),(c0,c1)=box
    Ma=cache_get(cache,0,a0,a1,sd[0],6); Mb=cache_get(cache,1,b0,b1,sd[1],2); Mc=cache_get(cache,2,c0,c1,sd[2],2)
    m=Ma[0]*Mb[0]*Mc[0]
    A2lo=0.0 if a0<=0<=a1 else min(a0*a0,a1*a1); A2hi=max(a0*a0,a1*a1)
    Bl=b0-A2hi/12; Bh=b1-A2lo/12
    dl,dh=d0range(a0,a1,b0,b1)
    ul,uh=c0+dl,c1+dh
    U2l=0.0 if ul<=0<=uh else min(ul*ul,uh*uh); U2h=max(ul*ul,uh*uh)
    Tl,Th=3*U2l,3*U2h
    vmin=V(Bh,Th if Bh>=0 else Tl); vmax=V(Bl,Th if Bl<=0 else Tl)
    lo1,hi1=m*vmin,m*vmax
    if Tl<=0 or m==0:
        return lo1,hi1
    ac,bc,cc=(a0+a1)/2,(b0+b1)/2,(c0+c1)/2
    Bc=bc-ac*ac/12; uc=cc+d0(ac,bc); Tc=3*uc*uc
    xc=xroot(Bc,Tc)
    Vc=Vx(Bc,xc); VT=-Bc*(4*xc+Bc)/(2*xc*(xc+Bc)); VB=-6*xc*xc+4.5*Bc*Bc-1.5*Bc**3/xc
    E1B=(Mb[1]-bc*Mb[0])*Ma[0]*Mc[0]-(Ma[2]-ac*ac*Ma[0])*Mb[0]*Mc[0]/12
    Id0=-Ma[1]*Mb[1]/4+Ma[3]*Mb[0]/72
    Id02=Ma[2]*Mb[2]/16-Ma[4]*Mb[1]/144+Ma[6]*Mb[0]/5184
    Eu2=Mc[2]*Ma[0]*Mb[0]+2*Mc[1]*Id0+Mc[0]*Id02
    E1T=3*(Eu2-uc*uc*m)
    # x-range over rectangle
    xs=[xroot(Bl,Th),xroot(Bh,Th),xroot(Bl,Tl),xroot(Bh,Tl)]
    xmax=max(xroot(Bl,Th),xroot(Bh,Th))
    xmin=min(xroot(Bl,Tl),xroot(Bh,Tl))
    for TT in (Tl,):
        Bs=(TT/4)**(1/3)
        if Bl<Bs<Bh: xmin=min(xmin,Bs)
    X=(xmin,xmax); Bi=(Bl,Bh)
    Y=(xroot(Bl,Tl)+Bl, xroot(Bh,Th)+Bh)
    if Y[0]<=0 or X[0]<=0: return lo1,hi1
    Q=iadd(iadd(isc(4,ipow(X,2)),isc(2,imul(Bi,X))),ipow(Bi,2))
    VTT=idiv(imul(Bi,Q),isc(12,imul(ipow(X,3),ipow(Y,3))))
    VBT=idiv(imul(Q,isub(isc(2,X),Bi)),isc(4,imul(ipow(X,3),Y)))
    VBB=iadd(iadd(isc(3,Bi),isc(6,X)), iadd(isc(-4.5,idiv(ipow(Bi,2),X)), iadd(isc(-0.75,idiv(ipow(Bi,3),ipow(X,2))), isc(0.75,idiv(ipow(Bi,4),ipow(X,3))))))
    SBB,SBT,STT=iabs_max(VBB),iabs_max(VBT),iabs_max(VTT)
    dB=max(abs(Bl-Bc),abs(Bh-Bc)); dT=max(abs(Tl-Tc),abs(Th-Tc))
    R=0.5*(SBB*dB*dB+2*SBT*dB*dT+STT*dT*dT)
    mid=Vc*m+VB*E1B+VT*E1T
    lo2,hi2=mid-R*m,mid+R*m
    return max(lo1,lo2),min(hi1,hi2)
def cache_get(cache,d,x0,x1,s,n):
    key=(d,x0,x1)
    r=cache.get(key)
    if r is None:
        r=moments(x0,x1,s,n); cache[key]=r
    return r
def run(k,budget):
    sd=(math.sqrt(2/k),math.sqrt(2),math.sqrt(6*k))
    A=[7*s for s in sd]; n0=8
    cache={}; heap=[]; tot=[0.0,0.0]
    def add(box):
        lo,hi=box_integral(box,sd,cache)
        tot[0]+=lo; tot[1]+=hi
        heapq.heappush(heap,(-(hi-lo),box,lo,hi))
    e=[[-A[d]+2*A[d]*i/n0 for i in range(n0+1)] for d in range(3)]
    for i in range(n0):
        for j in range(n0):
            for l in range(n0):
                add(((e[0][i],e[0][i+1]),(e[1][j],e[1][j+1]),(e[2][l],e[2][l+1])))
    nb=len(heap); t=time.time(); rep=[3000,10000,30000,100000,300000,1000000]
    while nb<budget:
        _,box,lo,hi=heapq.heappop(heap)
        tot[0]-=lo; tot[1]-=hi
        wid=[(box[i][1]-box[i][0])/sd[i] for i in range(3)]
        dmax=wid.index(max(wid))
        l_,h_=box[dmax]; mid=(l_+h_)/2
        for half in ((l_,mid),(mid,h_)):
            b=list(box); b[dmax]=half; add(tuple(b))
        nb+=1
        if rep and nb>=rep[0]:
            rep.pop(0)
            s=3*k*k
            print('k=%g boxes=%d 3k^2 EV in [%.5f, %.5f] width=%.5f t=%.0fs'%(k,nb,s*tot[0],s*tot[1],s*(tot[1]-tot[0]),time.time()-t)); sys.stdout.flush()
if __name__=='__main__':
    run(float(sys.argv[1]),int(sys.argv[2]))
