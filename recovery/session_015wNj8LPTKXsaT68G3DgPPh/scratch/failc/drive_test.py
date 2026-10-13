import sys, math, time
sys.path.insert(0,'.')
import fcore
from ia import isqrt, dn, up
def run(k, tol, L=8.0, n0=8, second=True, maxdepth=40):
    s2a=(2.0/k,2.0/k) if (2.0/k)*k==2.0 else None
    sa=isqrt(s2a); sb=isqrt((2.0,2.0)); sc=isqrt((6.0*k,6.0*k))
    Ma=fcore.Moments(sa,s2a,6); Mb=fcore.Moments(sb,(2.0,2.0),2); Mc=fcore.Moments(sc,(6.0*k,6.0*k),2)
    La,Lb,Lc=L*sa[1],L*sb[1],L*sc[1]
    edges=[[-La+2*La*i/n0 for i in range(n0+1)],[-Lb+2*Lb*i/n0 for i in range(n0+1)],[-Lc+2*Lc*i/n0 for i in range(n0+1)]]
    sd=(sa[1],sb[1],sc[1])
    stack=[((edges[0][i],edges[0][i+1]),(edges[1][j],edges[1][j+1]),(edges[2][l],edges[2][l+1]),0) for i in range(n0) for j in range(n0) for l in range(n0)]
    lo=hi=0.0; nleaf=0; t=time.time()
    while stack:
        a,b,c,dep=stack.pop()
        box=(a,b,c)
        I=fcore.V_box_integral(box,Ma,Mb,Mc,second)
        if I[1]-I[0]>tol and dep<maxdepth:
            w=[(box[i][1]-box[i][0])/sd[i] for i in range(3)]
            d=w.index(max(w)); l_,h_=box[d]; mid=0.5*(l_+h_)
            for half in ((l_,mid),(mid,h_)):
                nb=list(box); nb[d]=half; stack.append((nb[0],nb[1],nb[2],dep+1))
        else:
            lo=dn(lo+I[0]); hi=up(hi+I[1]); nleaf+=1
    s=3*k*k
    print('k=%g tol=%g leaves=%d  3k^2 E[V] in [%.6f, %.6f] width %.6f  time %.0fs'%(k,tol,nleaf,s*lo,s*hi,s*(hi-lo),time.time()-t)); sys.stdout.flush()
if __name__=='__main__':
    run(float(sys.argv[1]),float(sys.argv[2]))
