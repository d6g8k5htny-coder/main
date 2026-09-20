"""PRCP planar-limit proxy law: exact conditional law of the nine free jets
J9=(q,a,w,z,c40,c31,c22,c13,c04)=(f_yy,f_xxy,f_xyy,f_yyy,f_xxxx,f_xxxy,f_xxyy,f_xyyy,f_yyyy)
given six pins P=(f,f_x,f_y,f_xx,f_xy,f_xxx)=(b,0,0,0,0,2) under planar
Bargmann-Fock K=exp(-(x^2+y^2)/2). Computed in EXACT RATIONALS from the kernel."""
from fractions import Fraction as F

def dblfact(n):
    r=1
    while n>1: r*=n; n-=2
    return r
def gk(k):     # d^k/dt^k exp(-t^2/2) at 0
    if k%2==1: return F(0)
    j=k//2
    return F((-1)**j*dblfact(2*j-1)) if j>0 else F(1)
def covd(a,b): # Cov(d^a f, d^b f) = (-1)^{|b|} d^{a+b}K(0)
    return F((-1)**(b[0]+b[1]))*gk(a[0]+b[0])*gk(a[1]+b[1])

P=[(0,0),(1,0),(0,1),(2,0),(1,1),(3,0)]
J=[(0,2),(2,1),(1,2),(0,3),(4,0),(3,1),(2,2),(1,3),(0,4)]
JN=['q','a','w','z','c40','c31','c22','c13','c04']

SPP=[[covd(P[i],P[j]) for j in range(6)] for i in range(6)]
SJJ=[[covd(J[i],J[j]) for j in range(9)] for i in range(9)]
SJP=[[covd(J[i],P[j]) for j in range(6)] for i in range(9)]

def inv(Araw):
    n=len(Araw)
    A=[[F(Araw[i][j]) for j in range(n)]+[F(1) if i==j else F(0) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv=next(r for r in range(col,n) if A[r][col]!=0)
        A[col],A[piv]=A[piv],A[col]
        pv=A[col][col]; A[col]=[x/pv for x in A[col]]
        for r2 in range(n):
            if r2!=col and A[r2][col]!=0:
                f2=A[r2][col]; A[r2]=[A[r2][k]-f2*A[col][k] for k in range(2*n)]
    return [[A[i][j+n] for j in range(n)] for i in range(n)]

SPPi=inv(SPP)
R9=[[sum(SJP[i][k]*SPPi[k][j] for k in range(6)) for j in range(6)] for i in range(9)]
b=F(6,5)
p=[b,F(0),F(0),F(0),F(0),F(2)]
MEAN=[sum(R9[i][j]*p[j] for j in range(6)) for i in range(9)]
COV=[[SJJ[i][j]-sum(R9[i][k]*SJP[j][k] for k in range(6)) for j in range(9)] for i in range(9)]

if __name__=='__main__':
    print("conditional means:")
    for n2,m in zip(JN,MEAN): print(f"  E[{n2}|pins] = {m}")
    print("conditional covariance (exact rationals):")
    for i in range(9):
        print("  ",JN[i],[str(COV[i][j]) for j in range(9)])
    # anchors from published record:
    ok=[]
    ok.append(('E[q]=-b',            MEAN[0]==-b))
    ok.append(('E[a,w,z]=0',         MEAN[1]==0 and MEAN[2]==0 and MEAN[3]==0))
    ok.append(('Cov(qawz)=diag2226', all(COV[i][j]==(F([2,2,2,6][i]) if i==j else F(0)) for i in range(4) for j in range(4))))
    ok.append(('Var(c40)=24',        COV[4][4]==24))
    ok.append(('Var(c31)=6',         COV[5][5]==6))
    ok.append(('Cov(q,c22)=-2',      COV[0][6]==-2))
    ok.append(('Cov(q,c04)=-12',     COV[0][8]==-12))
    ok.append(('Cov(c22,c04)=12',    COV[6][8]==12))
    for name,v in ok: print(("PASS " if v else "FAIL ")+name)
    print("ALL ANCHORS:", "PASS" if all(v for _,v in ok) else "FAIL")
