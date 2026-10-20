# Exact-Gaussian (planar BF kernel exp(-|h|^2/2); side-24 periodization corrections ~e^-288 ignored)
# conditional laws via mpmath; expectations of determinant products by Monte Carlo (common random numbers).
import mpmath as mp, random, math, sys
mp.mp.dps=90
def He(n,x):
    a,b=mp.mpf(1),x
    if n==0: return a
    for m in range(1,n): a,b=b,x*b-m*a
    return b
def cov(P1,a1,P2,a2):
    h1=P1[0]-P2[0]; h2=P1[1]-P2[1]
    return (-1)**(a1[0]+a1[1])*He(a1[0]+a2[0],h1)*He(a1[1]+a2[1],h2)*mp.e**(-(h1*h1+h2*h2)/2)
def condlaw(obs,tgt,qry):
    So=mp.matrix(len(obs),len(obs)); Sqo=mp.matrix(len(qry),len(obs)); Sqq=mp.matrix(len(qry),len(qry))
    for i,(P,a) in enumerate(obs):
        for j,(R,b) in enumerate(obs): So[i,j]=cov(P,a,R,b)
    for i,(P,a) in enumerate(qry):
        for j,(R,b) in enumerate(obs): Sqo[i,j]=cov(P,a,R,b)
        for j,(R,b) in enumerate(qry): Sqq[i,j]=cov(P,a,R,b)
    G=mp.lu_solve(So, mp.matrix(tgt))
    mean=Sqo*G
    K=Sqo*mp.inverse(So)*Sqo.T
    C=Sqq-K
    return mean,C,So
def factor(C):
    n=C.rows; C=(C+C.T)/2
    E,Q=mp.eigsy(C)
    L=[[0.0]*n for _ in range(n)]
    for j in range(n):
        lam=max(E[j],0); sq=mp.sqrt(lam)
        for i in range(n): L[i][j]=float(Q[i,j]*sq)
    return L
def gdens(mean,C,x):
    n=C.rows; d=mp.matrix([x[i]-mean[i] for i in range(n)])
    return mp.e**(-(d.T*mp.lu_solve(C,d))[0]/2)/mp.sqrt((2*mp.pi)**n*mp.det(C))
H=[(2,0),(1,1),(0,2)]
def mc(mean,L,N,fn,seed):
    rnd=random.Random(seed); n=len(L); m=[float(v) for v in mean]; acc=0.0
    for _ in range(N):
        g=[rnd.gauss(0,1) for _ in range(n)]
        v=[m[i]+sum(L[i][j]*g[j] for j in range(n)) for i in range(n)]
        acc+=fn(v)
    return acc/N
def dets(v):
    out=[]
    for k in range(0,len(v),3):
        a,b_,c=v[k],v[k+1],v[k+2]; out.append((a*c-b_*b_,a))
    return out
def Wfn(v):  # endpoint weight F2(H_M)F1(H_S)
    (dM,aM),(dS,aS)=dets(v[:6])
    return abs(dM*dS) if (dM>0 and aM<0 and dS<0) else 0.0
b=mp.mpf('1.2'); kap=mp.mpf(1)
def pins(r):
    M=(-r/2,mp.mpf(0)); S=(r/2,mp.mpf(0))
    obs=[(M,(0,0)),(M,(1,0)),(M,(0,1)),(S,(0,0)),(S,(1,0)),(S,(0,1))]
    tgt=[b,0,0,b-kap*r**3/6,0,0]
    return M,S,obs,tgt
def Zr(r,N):
    M,S,obs,tgt=pins(r)
    mean,C,_=condlaw(obs,tgt,[(M,a) for a in H]+[(S,a) for a in H])
    return mc(mean,factor(C),N,Wfn,1)
mode=sys.argv[1]
if mode=='A':   # 024 sec.7: angular integral of lambda_crit(y,u) vs r^4 t^-6
    N=int(sys.argv[2])
    for r in [mp.mpf('0.04'),mp.mpf('0.02'),mp.mpf('0.01')]:
        M,S,obs,tgt=pins(r); Z=Zr(r,4*N)
        u=b-mp.mpf('0.5')*kap*r**3/6
        for tfac in ['2.5r','5r','0.1','0.2','0.4']:
            t=r*mp.mpf(tfac[:-1]) if tfac.endswith('r') else mp.mpf(tfac)
            angs=[mp.pi*(j+0.5)/8 for j in range(8)]   # (0,pi); law symmetric under z->-z
            acc=0
            for th in angs:
                y=(t*mp.cos(th),t*mp.sin(th))
                third=[(y,(0,0)),(y,(1,0)),(y,(0,1))]
                m3,C3,_=condlaw(obs,tgt,third)
                p=gdens(m3,C3,[u,0,0])
                mean,C,_=condlaw(obs+third,tgt+[u,0,0],[(M,a) for a in H]+[(S,a) for a in H]+[(y,a) for a in H])
                E=mc(mean,factor(C),N,lambda v: Wfn(v)*abs(dets(v)[2][0]),2)
                acc+=p*E/Z
            Lam=2*mp.pi*acc/8
            print(f'r={float(r):.3f} t={tfac:>5} ({float(t):.4f})  Lambda={float(Lam):.4e}  Lambda/(r^4 t^-6)={float(Lam/(r**4*t**-6)):.4e}  Z/r^2={Z/float(r)**2:.4f}',flush=True)
if mode=='B':   # 024 sec.5 / P2 Region II: conditional triple determinant at X=M+r(p,q)
    N=int(sys.argv[2]); pp=mp.mpf('0.15')
    for r in [mp.mpf('0.04'),mp.mpf('0.02'),mp.mpf('0.01')]:
        M,S,obs,tgt=pins(r)
        for lab,qq in [('axis q=0',mp.mpf(0)),('RegII q=rp/10',r*pp/10),('RegI q=p',pp)]:
            Xp=(M[0]+r*pp,r*qq)
            obs2=obs+[(Xp,(1,0)),(Xp,(0,1))]
            mean,C,_=condlaw(obs2,tgt+[0,0],[(M,a) for a in H]+[(S,a) for a in H]+[(Xp,a) for a in H])
            L=factor(C)
            E=mc(mean,L,N,lambda v: abs(dets(v)[0][0]*dets(v)[1][0]*dets(v)[2][0]),3)
            rho2=pp**2+qq**2
            print(f'r={float(r):.3f} {lab:>15}: E|detM detS detX|={E:.4e}  /(r^6 rho^2)={E/float(r**6*rho2):.4e}  /(r^3 rho^2)={E/float(r**3*rho2):.4e}  mean A-shift f_xx(M)={float(mean[0]):.3e}',flush=True)
