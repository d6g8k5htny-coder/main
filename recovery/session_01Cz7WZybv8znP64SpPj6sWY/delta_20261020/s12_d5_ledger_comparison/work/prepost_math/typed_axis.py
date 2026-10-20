# Typed vs untyped conditional triple-determinant expectation on the pin axis, X=M+r(p,0).
# Planar BF kernel exp(-|h|^2/2), kappa=1 (k=1/6), b=1.2; same conditioning as the 024 lens/adv scripts.
import mpmath as mp, random
mp.mp.dps=60
def He(n,x):
    a,b=mp.mpf(1),x
    if n==0: return a
    for m in range(1,n): a,b=b,x*b-m*a
    return b
def kc(P,a,R,c):
    h1=P[0]-R[0]; h2=P[1]-R[1]
    return (-1)**(a[0]+a[1])*He(a[0]+c[0],h1)*He(a[1]+c[1],h2)*mp.exp(-(h1*h1+h2*h2)/2)
def cond(obs,tgt,qry):
    n,m=len(obs),len(qry)
    S=mp.matrix(n,n);C=mp.matrix(m,n);Q=mp.matrix(m,m)
    for i in range(n):
        for j in range(n): S[i,j]=kc(obs[i][0],obs[i][1],obs[j][0],obs[j][1])
    for i in range(m):
        for j in range(n): C[i,j]=kc(qry[i][0],qry[i][1],obs[j][0],obs[j][1])
        for j in range(m): Q[i,j]=kc(qry[i][0],qry[i][1],qry[j][0],qry[j][1])
    Si=mp.inverse(S)
    return C*(Si*mp.matrix(tgt)), Q-C*Si*C.T
def sq(c):
    m=c.rows;c=(c+c.T)/2;E,V=mp.eigsy(c)
    return [[float(V[i,j]*mp.sqrt(max(E[j],0))) for j in range(m)] for i in range(m)]
HS=[(2,0),(1,1),(0,2)]
b=mp.mpf('1.2'); kap=mp.mpf(1)
for p in [mp.mpf('-0.15')]:
  for r in [mp.mpf('0.04'),mp.mpf('0.02'),mp.mpf('0.01')]:
    M=(-r/2,mp.mpf(0));S=(r/2,mp.mpf(0));X=(M[0]+r*p,mp.mpf(0))
    pins=[(M,(0,0)),(M,(1,0)),(M,(0,1)),(S,(0,0)),(S,(1,0)),(S,(0,1))]
    tg=[b,0,0,b-kap*r**3/6,0,0]
    obs=pins+[(X,(1,0)),(X,(0,1))]
    mH,cH=cond(obs,tg+[0,0],[(M,a) for a in HS]+[(S,a) for a in HS]+[(X,a) for a in HS])
    L=sq(cH); m=[float(v) for v in mH]
    rnd=random.Random(5); N=20000; aU=0.0; aT=0.0; nT=0
    for _ in range(N):
        g=[rnd.gauss(0,1) for _ in range(9)]
        u=[m[i]+sum(L[i][j]*g[j] for j in range(9)) for i in range(9)]
        eM=u[0]*u[2]-u[1]**2; eS=u[3]*u[5]-u[4]**2; eX=u[6]*u[8]-u[7]**2
        tr=abs(eM*eS*eX); aU+=tr
        if eM>0 and u[0]<0 and eS<0: aT+=tr; nT+=1
    den=float(r)**3*float(p*p)
    print(f'p={float(p):+.2f} r={float(r):.3f}: untyped E/(r^3rho^2)={aU/N/den:.3f}  typed E[.;types]/(r^3rho^2)={aT/N/den:.3f}  P(types)={nT/N:.4f}',flush=True)
