# Independent first variation of P (15.2) at d=3 for the periodized product kernel.
# Written from P.md (15.1)-(15.2) definitions only; no reuse of other lens code.
import itertools, sympy as sp, mpmath as mp
mp.mp.dps=60
# --- reference cone moments, independent symbolic derivation
print('sympy block skipped in fast variant')
# numeric cross-check of Exx by brute 3D quadrature in (s,x,y) polar
fs=lambda S: mp.quad(lambda Z: (S*S-Z)**2*mp.e**(-Z/2)/2,[0,S*S])
phis=lambda S: mp.e**(-S*S/(2*mp.mpf(5)/3))/mp.sqrt(2*mp.pi*mp.mpf(5)/3)
print('D0 numeric - closed', mp.nstr(mp.quad(lambda S: fs(S)*phis(S),[-mp.inf,0])-(mp.mpf(29)/6-mp.sqrt(6)),5))
print('Ess numeric - closed', mp.nstr(mp.quad(lambda S: S*S*fs(S)*phis(S),[-mp.inf,0])-(mp.mpf(445)/18-5*mp.sqrt(6)/8),5))
fx=lambda S: mp.quad(lambda Z: (Z/2)*(S*S-Z)**2*mp.e**(-Z/2)/2,[0,S*S])
ExxN=mp.quad(lambda S: fx(S)*mp.e**(-S*S/(2*mp.mpf(5)/3))/mp.sqrt(2*mp.pi*mp.mpf(5)/3),[-mp.inf,0])
print('Exx numeric - closed', mp.nstr(ExxN-(mp.mpf(19)/2-53*mp.sqrt(6)/16),5))
D0n=mp.mpf(29)/6-mp.sqrt(6); Essn=mp.mpf(445)/18-5*mp.sqrt(6)/8; Exxn=mp.mpf(19)/2-53*mp.sqrt(6)/16
# M=E0[h zz^T] in z=(A22,A33,A23)
M=mp.matrix([[Essn+Exxn,Essn-Exxn,0],[Essn-Exxn,Essn+Exxn,0],[0,0,Exxn]])
# --- jet tensor of product kernel
def T(idx,kap):
    # idx: tuple of axis indices (0..2); returns d^idx K(0) for K=k(x1)k(x2)k(x3)
    c=[0,0,0]
    for i in idx: c[i]+=1
    out=mp.mpf(1)
    for n in c: out*=kap[n]
    return out
def Tc(vecs,kap):
    # full contraction of derivative tensor with list of vectors
    tot=mp.mpf(0)
    for idx in itertools.product(range(3),repeat=len(vecs)):
        p=mp.mpf(1)
        for v_,i in zip(vecs,idx):
            p*=v_[i]
            if p==0: break
        if p!=0: tot+=p*T(idx,kap)
    return tot
def logintegrand_parts(u,e2,e3,kap):
    E=[mp.matrix([1,0,0]),mp.matrix([0,1,0]),mp.matrix([0,0,1])]
    SG=mp.matrix(3,3)
    for i in range(3):
        for j in range(3): SG[i,j]=-Tc([E[i],E[j]],kap)
    c=mp.matrix(3,1)
    for i in range(3): c[i]=-Tc([u,u,u,E[i]],kap)
    vt=-Tc([u]*6,kap)
    tau2=vt-(c.T*mp.inverse(SG)*c)[0]
    Vv=[(u,u),(u,e2),(u,e3)]; Aa=[(e2,e2),(e3,e3),(e2,e3)]
    cov=lambda P,Q: Tc([P[0],P[1],Q[0],Q[1]],kap)
    SV=mp.matrix(3,3); SAV=mp.matrix(3,3); SAA=mp.matrix(3,3)
    for i in range(3):
        for j in range(3):
            SV[i,j]=cov(Vv[i],Vv[j]); SAV[i,j]=cov(Aa[i],Vv[j]); SAA[i,j]=cov(Aa[i],Aa[j])
    G=SAA-SAV*mp.inverse(SV)*SAV.T
    return mp.log(mp.det(SG)), mp.log(mp.det(SV)), mp.log(tau2), G, tau2, SV
kref={0:mp.mpf(1),1:0,2:mp.mpf(-1),3:0,4:mp.mpf(3),5:0,6:mp.mpf(-15)}
G0ref=None
def dlogc_dir(u,e2,e3,n,h=mp.mpf('1e-25')):
    kp=dict(kref); km=dict(kref); kp[n]+=h; km[n]-=h
    a=logintegrand_parts(u,e2,e3,kp); b=logintegrand_parts(u,e2,e3,km); r=logintegrand_parts(u,e2,e3,kref)
    dlG=(a[0]-b[0])/(2*h); dlV=(a[1]-b[1])/(2*h); dlt=(a[2]-b[2])/(2*h)
    dG=(a[3]-b[3])/(2*h); G0=r[3]
    Gi=mp.inverse(G0)
    dD=-(D0n/2)*sum((Gi*dG)[i,i] for i in range(3)) + sum((Gi*dG*Gi*M)[i,i] for i in range(3))/2
    # log integrand = log pG + log pV + (2/3)log tau^2 + log D ; pG ~ det^-1/2
    return -dlG/2 - dlV/2 + (mp.mpf(2)/3)*dlt + dD/D0n, r
def frame(th,ph,rot):
    u=mp.matrix([mp.sin(th)*mp.cos(ph),mp.sin(th)*mp.sin(ph),mp.cos(th)])
    u=rot*u
    # Gram-Schmidt transverse frame with an arbitrary seed
    seed=mp.matrix([mp.mpf('0.3'),mp.mpf('-0.7'),mp.mpf('0.64')])
    e2=seed-(seed.T*u)[0]*u; e2=e2/mp.norm(e2)
    e3=mp.matrix([u[1]*e2[2]-u[2]*e2[1],u[2]*e2[0]-u[0]*e2[2],u[0]*e2[1]-u[1]*e2[0]])
    return u,e2,e3
def rotmat(a,b,c):
    Rz=lambda t: mp.matrix([[mp.cos(t),-mp.sin(t),0],[mp.sin(t),mp.cos(t),0],[0,0,1]])
    Ry=lambda t: mp.matrix([[mp.cos(t),0,mp.sin(t)],[0,1,0],[-mp.sin(t),0,mp.cos(t)]])
    return Rz(a)*Ry(b)*Rz(c)
def sphere_avg(n,nz,nph,rot):
    xs,ws=zip(*[(x,w_) for x,w_ in zip(*mp.gauss_quadrature(nz,'legendre'))]) if hasattr(mp,'gauss_quadrature') else (None,None)
    return None
# Gauss-Legendre nodes
def gl(nz):
    xs=[];ws=[]
    for k in range(1,nz+1):
        x=mp.cos(mp.pi*(k-mp.mpf(1)/4)/(nz+mp.mpf(1)/2))
        for _ in range(100):
            p0,p1=mp.mpf(1),x
            for j in range(2,nz+1): p0,p1=p1,((2*j-1)*x*p1-(j-1)*p0)/j
            dp=nz*(x*p1-p0)/(x*x-1)
            dx=p1/dp; x-=dx
            if abs(dx)<mp.mpf(10)**(-55): break
        p0,p1=mp.mpf(1),x
        for j in range(2,nz+1): p0,p1=p1,((2*j-1)*x*p1-(j-1)*p0)/j
        dp=nz*(x*p1-p0)/(x*x-1)
        xs.append(x); ws.append(2/((1-x*x)*dp*dp))
    return xs,ws
import sys
res={}
for (nz,nph,rot) in [(5,10,rotmat(0,0,0)),(7,14,rotmat(mp.mpf('0.37'),mp.mpf('1.1'),mp.mpf('-2.2')))]:
    xs,ws=gl(nz)
    out={}
    for n in (2,4,6):
        acc=mp.mpf(0)
        for x,wt in zip(xs,ws):
            th=mp.acos(x)
            for j in range(nph):
                ph=2*mp.pi*j/nph
                u,e2,e3=frame(th,ph,rot)
                val,_=dlogc_dir(u,e2,e3,n)
                acc+=wt*val
        out[n]=acc/(2*nph)   # sum wt = 2, nph phi nodes -> divide by 2*nph for average
    res[(nz,nph)]=out
    print('grid',nz,nph,{k:mp.nstr(v_,30) for k,v_ in out.items()})
pred={2:mp.mpf(39)/70,4:mp.mpf(-1)/70,6:mp.mpf(-1)/21}
for g_,o in res.items():
    print(g_,{k:mp.nstr(o[k]-pred[k],3) for k in o})
# reference tau^2, Gamma, det SV at an arbitrary direction
u,e2,e3=frame(mp.mpf('0.9'),mp.mpf('2.0'),rotmat(0,0,0))
_,r=dlogc_dir(u,e2,e3,2)
print('ref tau2',mp.nstr(r[4],15),' detSV',mp.nstr(mp.det(r[5]),15))
print('ref Gamma (A22,A33,A23):'); print(mp.nstr(r[3],12))
# assemble P_3(L)
Ls=sp.symbols('L')
He={2:Ls**2-1,4:Ls**4-6*Ls**2+3,6:Ls**6-15*Ls**4+45*Ls**2-15}
He0={2:-1,4:3,6:-15}
dm={n:sp.expand(2*(He[n]-He0[n])) for n in (2,4,6)}
poly=sp.expand(sum(sp.Rational(int(round(float(pred[n]*210))),210)*dm[n] for n in (2,4,6)))
print('dm',dm,' poly',poly, ' ==P3', sp.expand(poly-(-Ls**2*(10*Ls**4-147*Ls**2+315)/105))==0)
