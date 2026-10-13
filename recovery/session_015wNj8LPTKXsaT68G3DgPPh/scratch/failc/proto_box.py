import math, heapq, time, sys
from float_jfail import s0_root
SQ2=math.sqrt(2)
def Phi(x): return 0.5*(1+math.erf(x/SQ2))
def mass1(lo,hi,sd): return Phi(hi/sd)-Phi(lo/sd)
def H(B,T):
    s=s0_root(B,T)
    return abs(B)**3-4*s**3+3*B*B*s
def sqr_range(lo,hi):
    if lo<=0<=hi: return 0.0, max(lo*lo,hi*hi)
    return min(lo*lo,hi*hi), max(lo*lo,hi*hi)
def mul_range(a0,a1,b0,b1):
    p=[a0*b0,a0*b1,a1*b0,a1*b1]; return min(p),max(p)
def cube_range(lo,hi): return lo**3, hi**3
def box_bounds(box,k):
    (a0,a1),(b0,b1),(c0,c1)=box
    A2lo,A2hi=sqr_range(a0,a1)
    Blo=b0-A2hi/(12*k); Bhi=b1-A2lo/(12*k)
    ab0,ab1=mul_range(a0,a1,b0,b1)
    a3lo,a3hi=cube_range(a0,a1)
    d0lo=-ab1/(4*k)+a3lo/(72*k*k); d0hi=-ab0/(4*k)+a3hi/(72*k*k)
    ulo,uhi=c0+d0lo,c1+d0hi
    U2lo,U2hi=sqr_range(ulo,uhi)
    Tlo,Thi=3*k*U2lo,3*k*U2hi
    return 3*k*k*H(Bhi,Tlo), 3*k*k*H(Blo,Thi)
def run(k,budget,A=9.0,C=15.0):
    sd=(SQ2,SQ2,math.sqrt(6))
    init=[]
    n0=6
    ea=[-A+2*A*i/n0 for i in range(n0+1)]; ec=[-C+2*C*i/n0 for i in range(n0+1)]
    heap=[]
    lo_tot=hi_tot=0.0
    def add(box):
        nonlocal lo_tot,hi_tot
        m=mass1(*box[0],sd[0])*mass1(*box[1],sd[1])*mass1(*box[2],sd[2])
        flo,fhi=box_bounds(box,k)
        lo_tot+=m*flo; hi_tot+=m*fhi
        heapq.heappush(heap,(-(m*(fhi-flo)),box,m,flo,fhi))
    for i in range(n0):
        for j in range(n0):
            for l in range(n0):
                add(((ea[i],ea[i+1]),(ea[j],ea[j+1]),(ec[l],ec[l+1])))
    nbox=len(heap); t=time.time()
    report=[1000,10000,100000,300000,1000000,3000000]
    while nbox<budget:
        negw,box,m,flo,fhi=heapq.heappop(heap)
        lo_tot-=m*flo; hi_tot-=m*fhi
        # split along widest scaled dim
        wid=[(box[i][1]-box[i][0])/sd[i] for i in range(3)]
        dmax=wid.index(max(wid))
        lo_,hi_=box[dmax]; mid=(lo_+hi_)/2
        for half in ((lo_,mid),(mid,hi_)):
            nb=list(box); nb[dmax]=half; add(tuple(nb))
        nbox+=1
        if report and nbox>=report[0]:
            report.pop(0)
            print('k=%g boxes=%d lo=%.5f hi=%.5f width=%.3f%% t=%.0fs'%(k,nbox,lo_tot,hi_tot,100*(hi_tot-lo_tot)/lo_tot,time.time()-t)); sys.stdout.flush()
    return lo_tot,hi_tot
if __name__=='__main__':
    k=float(sys.argv[1]); budget=int(sys.argv[2])
    run(k,budget)
