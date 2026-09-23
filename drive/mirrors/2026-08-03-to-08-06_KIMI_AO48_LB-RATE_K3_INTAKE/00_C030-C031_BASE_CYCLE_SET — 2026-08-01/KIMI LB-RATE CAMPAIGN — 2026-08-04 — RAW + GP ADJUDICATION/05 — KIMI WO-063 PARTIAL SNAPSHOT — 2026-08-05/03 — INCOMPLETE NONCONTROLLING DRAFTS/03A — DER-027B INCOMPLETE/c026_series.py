"""C026 exact-series engine: Fraction Laurent series in r; regularizing divided-difference frame;
exact leading coefficients of the S-quantities. Freeze 6ac9a33c..."""
from fractions import Fraction as Fr
NM = 17
class LS:
    __slots__=('off','c')
    def __init__(self, off=0, c=None):
        self.c = [Fr(x) for x in (c or [])]; self.off=off; self._trim()
    def _trim(self):
        while self.c and self.c[0]==0: self.c.pop(0); self.off+=1
        keep = NM - self.off + 1
        if keep <= 0: self.c=[]; self.off=NM+1
        else: self.c=self.c[:keep]
        if not self.c: self.off=NM+1
    def copy(self): return LS(self.off, self.c)
    def is0(self): return not self.c
    def __add__(a,b):
        if a.is0(): return b.copy()
        if b.is0(): return a.copy()
        off=min(a.off,b.off); end=max(a.off+len(a.c), b.off+len(b.c))
        c=[Fr(0)]*(end-off)
        for i,x in enumerate(a.c): c[a.off-off+i]+=x
        for i,x in enumerate(b.c): c[b.off-off+i]+=x
        return LS(off,c)
    def __neg__(self): return LS(self.off,[-x for x in self.c])
    def __sub__(a,b): return a+(-b)
    def __mul__(a,b):
        if isinstance(b,(int,Fr)):
            return LS(a.off,[x*b for x in a.c])
        if a.is0() or b.is0(): return LS()
        off=a.off+b.off; n=NM-off+1
        if n<=0: return LS()
        c=[Fr(0)]*n
        for i,x in enumerate(a.c):
            if i>=n: break
            for j,y in enumerate(b.c):
                if i+j>=n: break
                c[i+j]+=x*y
        return LS(off,c)
    __rmul__=__mul__
    def shift(self,k): return LS(self.off+k, self.c)
    def inv(self):
        assert self.c and self.c[0]!=0, "series inverse needs nonzero leading"
        n=NM-(-self.off)+1
        a=self.c; inv=[Fr(1)/a[0]]
        for k in range(1, NM+ -self.off+1+len(a)):
            if k>=len(a)+40: break
            s=Fr(0)
            for j in range(1,min(k,len(a)-0)):
                if j<len(a): s+=a[j]*inv[k-j] if k-j<len(inv) else 0
            if k-0<len(a): pass
            inv.append(-s/a[0])
            if len(inv)>NM+abs(self.off)+2: break
        return LS(-self.off, inv)
    def ev(self,r):
        return float(sum(float(x)*r**(self.off+i) for i,x in enumerate(self.c)))
    def lead(self): return (self.off, self.c[0] if self.c else Fr(0))
def LSc(x): return LS(0,[x])
HE=[[1],[0,1],[-1,0,1],[0,-3,0,1],[3,0,-6,0,1]]
def cser(a,b,du):
    n1=a[0]+b[0]; n2=a[1]+b[1]
    s=(-1)**((b[0]+b[1])+(n1+n2))
    u1,u2=Fr(du[0]),Fr(du[1])
    h1=LS(0,[HE[n1][k]*u1**k for k in range(n1+1)])
    h2=LS(0,[HE[n2][k]*u2**k for k in range(n2+1)])
    cc=(u1*u1+u2*u2)/2
    ex=[Fr(0)]*(NM+1); m=0; term=Fr(1)
    while 2*m<=NM:
        ex[2*m]=term; m+=1; term=term*(-cc)/m
    return (h1*h2*LS(0,ex))*s
def mat(n,m=None): m=m or n; return [[LS() for _ in range(m)] for _ in range(n)]
def mmul(A,B):
    n=len(A); k=len(B); m=len(B[0])
    C=mat(n,m)
    for i in range(n):
        for j in range(m):
            s=LS()
            for t in range(k): s=s+A[i][t]*B[t][j]
            C[i][j]=s
    return C
def madd(A,B): return [[A[i][j]+B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
def msub(A,B): return [[A[i][j]-B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
def mT(A): return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
def frmat_inv(M):
    n=len(M); A=[[Fr(M[i][j]) for j in range(n)]+[Fr(int(i==j)) for j in range(n)] for i in range(n)]
    for col in range(n):
        p=next(r for r in range(col,n) if A[r][col]!=0)
        A[col],A[p]=A[p],A[col]
        pv=A[col][col]
        A[col]=[x/pv for x in A[col]]
        for r in range(n):
            if r!=col and A[r][col]!=0:
                f=A[r][col]; A[r]=[A[r][j]-f*A[col][j] for j in range(2*n)]
    return [[A[i][n+j] for j in range(n)] for i in range(n)]
def frmat_det(M):
    n=len(M); A=[[Fr(M[i][j]) for j in range(n)] for i in range(n)]; d=Fr(1)
    for col in range(n):
        p=next((r for r in range(col,n) if A[r][col]!=0),None)
        if p is None: return Fr(0)
        if p!=col: A[col],A[p]=A[p],A[col]; d=-d
        d*=A[col][col]; pv=A[col][col]
        for r in range(col+1,n):
            f=A[r][col]/pv
            if f: A[r]=[A[r][j]-f*A[col][j] for j in range(n)]
    return d
def const_congr(W, A):
    n=len(A)
    B=mat(n)
    for i in range(n):
        for j in range(n):
            s=LS()
            for p in range(n):
                if W[i][p]==0: continue
                for q in range(n):
                    if W[j][q]==0: continue
                    s=s+A[p][q]*(W[i][p]*W[j][q])
            B[i][j]=s
    return B
def build_frame(FUN, MON, gexp, cexp):
    n=len(FUN)
    Spp=mat(n)
    for i,(ai,pi) in enumerate(FUN):
        for j,(aj,pj) in enumerate(FUN):
            Spp[i][j]=cser(ai,aj,(Fr(pi[0])-Fr(pj[0]),Fr(pi[1])-Fr(pj[1])))
    Vh=[[None]*n for _ in range(n)]
    for i,(ai,pi) in enumerate(FUN):
        for a_idx,(px,py) in enumerate(MON):
            x,y=Fr(pi[0]),Fr(pi[1])
            if ai==(0,0): v=x**px*y**py
            elif ai==(1,0): v=px*x**(px-1)*y**py if px>0 else Fr(0)
            elif ai==(0,1): v=py*x**px*y**(py-1) if py>0 else Fr(0)
            Vh[i][a_idx]=v
    W=frmat_inv(Vh)          # C = W . (Dg^-1 p)  in the scaled sense
    A=[[Spp[i][j].shift(gexp[i]+gexp[j]) for j in range(n)] for i in range(n)]
    B=const_congr(W,A)
    G=[[B[i][j].shift(-(cexp[i]+cexp[j])) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            assert G[i][j].is0() or G[i][j].off>=0, "G-B1 regularity violated at (%d,%d)"%(i,j)
    G0=[[ (G[i][j].c[0] if (not G[i][j].is0() and G[i][j].off==0) else Fr(0)) for j in range(n)] for i in range(n)]
    return Spp, Vh, W, G, G0
def neumann_inv(G, G0, K=11):
    n=len(G)
    G0i=frmat_inv(G0)
    Gk=[]
    for k in range(K+1):
        Gk.append([[ (G[i][j].c[k-G[i][j].off] if (not G[i][j].is0() and 0<=k-G[i][j].off<len(G[i][j].c)) else Fr(0)) for j in range(n)] for i in range(n)])
    def fm(Aa,Bb):
        return [[sum(Aa[i][t]*Bb[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
    X=[G0i]
    for k in range(1,K+1):
        S=[[Fr(0)]*n for _ in range(n)]
        for j in range(1,k+1):
            P=fm(Gk[j],X[k-j])
            S=[[S[i][t]+P[i][t] for t in range(n)] for i in range(n)]
        X.append([[-sum(G0i[i][t]*S[t][j] for t in range(n)) for j in range(n)] for i in range(n)])
    Xs=mat(n)
    for i in range(n):
        for j in range(n):
            Xs[i][j]=LS(0,[X[k][i][j] for k in range(K+1)])
    return Xs
