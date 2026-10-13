import math, heapq, time, sys
from float_jfail import s0_root
from proto_box import mass1, sqr_range, mul_range, cube_range, SQ2
def V(B,T):
    x=-s0_root(B,T)
    return -3*B*(2*x-B)*(x+B)
def ranges(box,k):
    (a0,a1),(b0,b1),(c0,c1)=box
    A2lo,A2hi=sqr_range(a0,a1)
    Blo=b0-A2hi/(12*k); Bhi=b1-A2lo/(12*k)
    ab0,ab1=mul_range(a0,a1,b0,b1)
    a3lo,a3hi=cube_range(a0,a1)
    d0lo=-ab1/(4*k)+a3lo/(72*k*k); d0hi=-ab0/(4*k)+a3hi/(72*k*k)
    U2lo,U2hi=sqr_range(c0+d0lo,c1+d0hi)
    return Blo,Bhi,3*k*U2lo,3*k*U2hi
def vbounds(box,k):
    Blo,Bhi,Tlo,Thi=ranges(box,k)
    vmin=V(Bhi,Thi if Bhi>=0 else Tlo)
    vmax=V(Blo,Thi if Blo<=0 else Tlo)
    return 3*k*k*vmin,3*k*k*vmax
def run(k,budget,A=9.0,C=15.0):
    sd=(SQ2,SQ2,math.sqrt(6))
    n0=6
    ea=[-A+2*A*i/n0 for i in range(n0+1)]; ec=[-C+2*C*i/n0 for i in range(n0+1)]
    heap=[]; tot=[0.0,0.0]
    def add(box):
        m=mass1(*box[0],sd[0])*mass1(*box[1],sd[1])*mass1(*box[2],sd[2])
        lo,hi=vbounds(box,k)
        tot[0]+=m*lo; tot[1]+=m*hi
        heapq.heappush(heap,(-(m*(hi-lo)),box,m,lo,hi))
    for i in range(n0):
        for j in range(n0):
            for l in range(n0):
                add(((ea[i],ea[i+1]),(ea[j],ea[j+1]),(ec[l],ec[l+1])))
    nbox=len(heap); t=time.time()
    report=[10000,100000,300000,1000000,3000000]
    # exact parts
    while nbox<budget:
        negw,box,m,lo,hi=heapq.heappop(heap)
        tot[0]-=m*lo; tot[1]-=m*hi
        wid=[(box[i][1]-box[i][0])/sd[i] for i in range(3)]
        dmax=wid.index(max(wid))
        l_,h_=box[dmax]; mid=(l_+h_)/2
        for half in ((l_,mid),(mid,h_)):
            nb=list(box); nb[dmax]=half; add(tuple(nb))
        nbox+=1
        if report and nbox>=report[0]:
            report.pop(0)
            print('k=%g boxes=%d EV in [%.5f, %.5f] width=%.4f t=%.0fs'%(k,nbox,tot[0],tot[1],tot[1]-tot[0],time.time()-t)); sys.stdout.flush()
    return tot
if __name__=='__main__':
    k=float(sys.argv[1]); budget=int(sys.argv[2])
    # exact parts (float): E[2T]*3k^2 = 3k^2*6k(E d0^2+6); E d0^2 = 1/(4k^2)+5/(216k^4)
    Ed02=1/(4*k*k)+5/(216*k**4)
    print('3k^2 * E[2T] =', 3*k*k*6*k*(Ed02+6))
    run(k,budget)
