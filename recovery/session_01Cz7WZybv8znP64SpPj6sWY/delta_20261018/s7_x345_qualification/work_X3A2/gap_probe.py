# Illustrative probe only (not part of any qualification): d=2, m=1, enlarged covariance of (U_r, A_r).
import mpmath as mp, itertools, math
mp.mp.dps=40
L=mp.mpf(3); N=10
ns=list(itertools.product(range(-N,N+1),repeat=2))
a=[mp.e**(-2*mp.pi**2*(n1*n1+n2*n2)/L**2) for n1,n2 in ns]; s=sum(a); a=[x/s for x in a]
w=2*mp.pi/L
def cov(r,th):
    u=(mp.cos(th),mp.sin(th)); y=(-mp.sin(th),mp.cos(th))
    rows=[]
    for (n1,n2) in ns:
        nu=1j*w*(n1*u[0]+n2*u[1]); ny=1j*w*(n1*y[0]+n2*y[1])
        pa=-r/2*(n1*u[0]+n2*u[1]); pc=r/2*(n1*u[0]+n2*u[1])
        ea=mp.expj(w*pa); ec=mp.expj(w*pc)
        rows.append([(ea+ec)/2,(ec-ea)/r,nu*(ec-ea)/r,6/r**2*(nu*ea+nu*ec-2*(ec-ea)/r),
                     ny*(ea+ec)/2,ny*(ec-ea)/r,ny*ny*ea])
    k=7
    S=mp.matrix(k,k)
    for i in range(k):
        for j in range(k):
            S[i,j]=mp.re(sum(a[t]*rows[t][i]*mp.conj(rows[t][j]) for t in range(len(ns))))
    return S
for r in [mp.mpf('0.3'),mp.mpf('0.1'),mp.mpf('0.03')]:
    vals=[]
    for th in [mp.pi*q/12 for q in range(13)]:
        E=mp.eigsy(cov(r,th))[0]
        vals.append(min(E))
    print('r',r,'min_frames lambda_min',mp.nstr(min(vals),6),'max_frames lambda_min',mp.nstr(max(vals),6))
