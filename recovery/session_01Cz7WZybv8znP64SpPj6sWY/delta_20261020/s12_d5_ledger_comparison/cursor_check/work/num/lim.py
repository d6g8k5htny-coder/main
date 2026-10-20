# r->0 limit of LS-DER-024 (4) typed intensity on planar BF kernel, reusing bfcheck.py's cov/condlaw/gdens.
# Six pins -> jet frame at midpoint: f=b, f_x=f_y=0, f_xx=0, f_xy=0, f_xxx=2*kappa (Hermite divided differences).
# Limiting endpoint weight W/r^2 -> kappa^2 S0^2 1{S0<0}, S0=f_yy(0); Z/r^2 -> kappa^2 E[S0^2;S0<0].
import sys; sys.path.insert(0, sys.argv[1])
import mpmath as mp, numpy as np
mp.mp.dps=60
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
    return mean,Sqq-K
def gdens(mean,C,x):
    n=C.rows; d=mp.matrix([x[i]-mean[i] for i in range(n)])
    return mp.e**(-(d.T*mp.lu_solve(C,d))[0]/2)/mp.sqrt((2*mp.pi)**n*mp.det(C))
b=mp.mpf('1.2'); kap=mp.mpf(1)
O=(mp.mpf(0),mp.mpf(0))
jet=[(O,(0,0)),(O,(1,0)),(O,(0,1)),(O,(2,0)),(O,(1,1)),(O,(3,0))]
jtg=[b,0,0,0,0,2*kap]
# check S0 law and Z limit
m0,c0=condlaw(jet,jtg,[(O,(0,2))])
mu0=float(m0[0]); s0=float(mp.sqrt(c0[0,0]))
def m2neg(mu,sig):
    a=-mu/sig
    return (mu**2+sig**2)*mp.ncdf(a)-mu*sig*mp.npdf(a)
Zlim=kap**2*m2neg(m0[0],mp.sqrt(c0[0,0]))
from math import erf, sqrt, pi, exp
def m2neg_np(mu,sig):
    a=-mu/sig
    Ph=0.5*(1+np.vectorize(erf)(a/sqrt(2))) if False else 0.5*(1+erf_np(a/np.sqrt(2)))
    ph=np.exp(-a*a/2)/np.sqrt(2*np.pi)
    return (mu**2+sig**2)*Ph-mu*sig*ph
import math
erf_np=np.frompyfunc(math.erf,1,1)
def lam_point(t,th,N,seed):
    y=(t*mp.cos(th),t*mp.sin(th))
    third=[(y,(0,0)),(y,(1,0)),(y,(0,1))]
    m3,C3=condlaw(jet,jtg,third)
    p=gdens(m3,C3,[b,0,0])
    qry=[(y,(2,0)),(y,(1,1)),(y,(0,2)),(O,(0,2))]
    mean,C=condlaw(jet+third,jtg+[b,0,0],qry)
    m=np.array([float(v) for v in mean]); Cf=np.array([[float(C[i,j]) for j in range(4)] for i in range(4)]); Cf=(Cf+Cf.T)/2
    # Rao-Blackwell: S0 | H_y Gaussian
    CHH=Cf[:3,:3]; cSH=Cf[3,:3]; g=np.linalg.solve(CHH,cSH); sc=np.sqrt(max(Cf[3,3]-cSH@g,0.0))
    L=np.linalg.cholesky(CHH)
    rng=np.random.default_rng(seed)
    acc=[];
    for _ in range(N//200000):
        z=rng.standard_normal((200000,3)); h=m[:3]+z@L.T
        muS=m[3]+(h-m[:3])@g
        a=-muS/sc; Ph=0.5*(1+erf_np(a/np.sqrt(2)).astype(float)); ph=np.exp(-a*a/2)/np.sqrt(2*np.pi)
        e2=(muS**2+sc**2)*Ph-muS*sc*ph
        w=np.abs(h[:,0]*h[:,2]-h[:,1]**2)*kap**0*e2
        acc.append(w)
    w=np.concatenate(acc)
    E=w.mean(); se=w.std(ddof=1)/np.sqrt(len(w))
    pf=float(p); Zf=float(Zlim)
    return pf*E/Zf, pf*se/Zf, pf
if __name__=='__main__':
    print('S0 limit law: mean',mu0,'sd',s0,'var',s0**2,' Zlim/r^2 =',mp.nstr(Zlim,10))
    mode=sys.argv[2]
    if mode=='profile':
        t=mp.mpf(sys.argv[3]); N=int(sys.argv[4])
        for thd in [0.5,1,2,4,6,8,11.25,15,22.5,30,45,60,75,90,105,120,135,150,157.5,165,170,175,178,179.5]:
            th=mp.pi*mp.mpf(thd)/180
            l,se,p=lam_point(t,th,N,1000+int(thd*10))
            print(f't={float(t)} theta={thd:7.2f}deg  lambda={l:.5e} +- {se:.1e}  p*t^6={p*float(t)**6:.4f}',flush=True)
    if mode=='integral':
        t=mp.mpf(sys.argv[3]); N=int(sys.argv[4]); n=int(sys.argv[5])
        vals=[];ses=[]
        for j in range(n):
            th=mp.pi*(j+mp.mpf('0.5'))/n
            l,se,p=lam_point(t,th,N,7919*n+j)
            vals.append(l); ses.append(se)
        Lam=2*np.pi/n*np.sum(vals); SE=2*np.pi/n*np.sqrt(np.sum(np.array(ses)**2))
        print(f't={float(t)} n_angles(0,pi)={n} (full-circle equiv {2*n})  N/angle={N}  Lambda={Lam:.5f} +- {SE:.5f}',flush=True)
