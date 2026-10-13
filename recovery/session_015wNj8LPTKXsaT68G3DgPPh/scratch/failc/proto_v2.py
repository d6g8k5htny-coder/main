import math, heapq, time, sys
from float_jfail import s0_root
SQ2=math.sqrt(2)
def Phi(x): return 0.5*(1+math.erf(x/SQ2))
def mass1(lo,hi,sd): return Phi(hi/sd)-Phi(lo/sd)
def V(B,T):
    x=-s0_root(B,T)
    return -3*B*(2*x-B)*(x+B)
def d0range(a0,a1,b0,b1):
    # d = -a*b/4 + a^3/72 over rectangle; extremes at corners or where d/da = -b/4 + a^2/24 = 0 (a = +-sqrt(6b)), b at endpoints (linear in b)
    cand=[]
    for b in (b0,b1):
        for a in (a0,a1):
            cand.append(-a*b/4+a**3/72)
        if b>0:
            r=math.sqrt(6*b)
            for a in (r,-r):
                if a0<a<a1: cand.append(-a*b/4+a**3/72)
    return min(cand),max(cand)
def ranges(box):
    (a0,a1),(b0,b1),(c0,c1)=box
    A2lo=0.0 if a0<=0<=a1 else min(a0*a0,a1*a1); A2hi=max(a0*a0,a1*a1)
    Blo=b0-A2hi/12; Bhi=b1-A2lo/12
    dlo,dhi=d0range(a0,a1,b0,b1)
    ulo,uhi=c0+dlo,c1+dhi
    U2lo=0.0 if ulo<=0<=uhi else min(ulo*ulo,uhi*uhi); U2hi=max(ulo*ulo,uhi*uhi)
    return Blo,Bhi,3*U2lo,3*U2hi
def vb(box):
    Blo,Bhi,Tlo,Thi=ranges(box)
    vmin=V(Bhi,Thi if Bhi>=0 else Tlo)
    vmax=V(Blo,Thi if Blo<=0 else Tlo)
    return vmin,vmax
def run(k,budget,smart):
    sd=(math.sqrt(2/k),SQ2,math.sqrt(6*k))
    A=[6.5*s for s in sd]
    n0=6
    heap=[]; tot=[0.0,0.0]
    def add(box):
        m=mass1(*box[0],sd[0])*mass1(*box[1],sd[1])*mass1(*box[2],sd[2])
        lo,hi=vb(box)
        tot[0]+=m*lo; tot[1]+=m*hi
        heapq.heappush(heap,(-(m*(hi-lo)),box,m,lo,hi))
    e=[[-A[d]+2*A[d]*i/n0 for i in range(n0+1)] for d in range(3)]
    for i in range(n0):
        for j in range(n0):
            for l in range(n0):
                add(((e[0][i],e[0][i+1]),(e[1][j],e[1][j+1]),(e[2][l],e[2][l+1])))
    nbox=len(heap); t=time.time(); report=[100000,300000,1000000]
    while nbox<budget:
        negw,box,m,lo,hi=heapq.heappop(heap)
        tot[0]-=m*lo; tot[1]-=m*hi
        if smart:
            best=None
            for d in range(3):
                l_,h_=box[d]; mid=(l_+h_)/2
                nb=list(box); nb[d]=(mid,mid)
                a,b=vb(tuple(nb))
                red=(hi-lo)-(b-a)
                if best is None or red>best[0]: best=(red,d)
            dmax=best[1]
        else:
            wid=[(box[i][1]-box[i][0])/sd[i] for i in range(3)]
            dmax=wid.index(max(wid))
        l_,h_=box[dmax]; mid=(l_+h_)/2
        for half in ((l_,mid),(mid,h_)):
            nb=list(box); nb[dmax]=half; add(tuple(nb))
        nbox+=1
        if report and nbox>=report[0]:
            report.pop(0)
            print('k=%g smart=%d boxes=%d EV*3k^2 in [%.4f, %.4f] width=%.4f t=%.0fs'%(k,smart,nbox,3*k*k*tot[0],3*k*k*tot[1],3*k*k*(tot[1]-tot[0]),time.time()-t)); sys.stdout.flush()
if __name__=='__main__':
    run(float(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
