# CW-DATA-014_run.py v1.2 — Q4-on anisotropic P0.2 runs + Q4-off r=0.0125 companion + wiring control
# Claude (Anthropic) C047R, 2026-07-20. Seed 20260720. Gates: CW-DER-008, CW-DER-013 (compensation).
# AMENDMENT v1.2 (registered pre-execution): dt=r/8 (was r/20) with per-run dt/2 agreement check
# on 3000 samples (must be 0 mismatches); T_max=120.
# Usage: python3 cw_data_014_run.py [off0125|on025|on0125]
# NOTE for replicators (CL-AUD-021 r2 blocker resolution): everything needed is in this file.
# KNOWN COMPUTE LIMIT: on0125 exceeded the authoring sandbox's per-call budget after 4 attempts
# (incl. N=12000, tau=70 variants) — help requested, see CW-DATA-014.
import numpy as np, sys
b=1.2
def sample9(N,r,rng):
    mu=np.array([-b,0,0,0,-3*b,0,0,0,3*b]); mu[2]+=r*r/4  # E[f_xyy]=+r^2/4 finite-r shift (CW-DER-013 §3)
    C=np.diag([2.,2.,2.,6.,24.,6.,6.,6.,96.])
    C[0,6]=C[6,0]=-2.; C[0,8]=C[8,0]=-12.; C[6,8]=C[8,6]=12.
    return mu+rng.standard_normal((N,9))@np.linalg.cholesky(C).T
def hess(J,r):
    q,a,w,z,f4x,f31,f22,f13,f4y=J.T
    fxxS=r+f4x*r*r/12; fxyS=a*r/2+f31*r*r/12; fyyS=q+w*r/2+f22*r*r/8
    fxxM=-r+f4x*r*r/12; fxyM=-a*r/2+f31*r*r/12; fyyM=q-w*r/2+f22*r*r/8
    detS=fxxS*fyyS-fxyS**2; detM=fxxM*fyyM-fxyM**2
    return detM,detS,(fxxM<0)&(detM>0)&(detS<0)
def flow(X,Y,J,r,qu):
    q,a,w,z,f4x,f31,f22,f13,f4y=J.T
    dX=X*X-0.25+a*X*Y+0.5*w*Y*Y
    dY=(q/r)*Y+0.5*a*(X*X-0.25)+w*X*Y+0.5*z*Y*Y
    if qu:
        dX=dX+r*((f4x/6)*X*(X*X-0.25)+(f31/6)*(3*X*X-0.25)*Y+(f22/2)*X*Y*Y+(f13/6)*Y**3)
        dY=dY+r*((f31/6)*(X**3-X/4)+(f22/2)*X*X*Y+(f13/2)*X*Y*Y+(f4y/6)*Y**3)
    return dX,dY
def run(J,r,qu,sgn,dt,Tmax=120.,Rmax=16.):
    q,a,w,z,f4x,f31,f22,f13,f4y=J.T; n=len(q)
    J11=1+(r*f4x/12 if qu else 0); J12=a/2+(r*f31/12 if qu else 0); J22=q/r+w/2+(r*f22/8 if qu else 0)
    tr=J11+J22; det=J11*J22-J12*J12
    lam=0.5*(tr+np.sqrt(np.maximum(tr*tr-4*det,0)))
    vx=np.where(np.abs(J12)>1e-12,J12,1.0); vy=np.where(np.abs(J12)>1e-12,lam-J11,0.0)
    nr=np.hypot(vx,vy); vx/=nr; vy/=nr
    X=0.5+sgn*1e-6*vx; Y=sgn*1e-6*vy
    tM=np.zeros(n,bool); act=np.ones(n,bool)
    for _ in range(int(Tmax/dt)):
        if not act.any(): break
        Xa,Ya,Ja=X[act],Y[act],J[act]
        k1=flow(Xa,Ya,Ja,r,qu); k2=flow(Xa+.5*dt*k1[0],Ya+.5*dt*k1[1],Ja,r,qu)
        k3=flow(Xa+.5*dt*k2[0],Ya+.5*dt*k2[1],Ja,r,qu); k4=flow(Xa+dt*k3[0],Ya+dt*k3[1],Ja,r,qu)
        Xa=Xa+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6; Ya=Ya+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
        idx=np.where(act)[0]; t=np.hypot(Xa+0.5,Ya)<1e-3; e=np.hypot(Xa,Ya)>Rmax
        tM[idx[t]]=True; X[idx]=Xa; Y[idx]=Ya; act[idx[t|e]]=False
    g=flow(X,Y,J,r,qu); st=act&~(np.hypot(*g)<1e-6)
    return tM,st
def analyze(tag,J,r,qu):
    detM,detS,T=hess(J,r); Jt=J[T]; nT=T.sum(); W=np.abs(detM[T]*detS[T]); dt=r/8
    t1,s1=run(Jt,r,qu,-1.0,dt); t2,s2=run(Jt,r,qu,+1.0,dt)
    st=(s1.mean()+s2.mean())/2; A=t1|t2; non=~A; gen=non.mean()
    Wn=W/W.sum(); wg=(Wn*non).sum(); ess=1/np.sum(Wn*Wn); se=np.sqrt(np.sum((Wn*(non-wg))**2))
    zv=1.96;den=1+zv*zv/nT
    lo=(gen+zv*zv/(2*nT)-zv*np.sqrt(gen*(1-gen)/nT+zv*zv/(4*nT*nT)))/den
    hi=(gen+zv*zv/(2*nT)+zv*np.sqrt(gen*(1-gen)/nT+zv*zv/(4*nT*nT)))/den
    print(f"[{tag}] typed={T.mean():.4f} n={nT} stall={st:.1e} {'VALID' if st<1e-3 else 'VOID'}")
    print(f"[{tag}] GENERIC 1-a={gen:.6f} [{lo:.6f},{hi:.6f}]  WEIGHTED 1-a={wg:.2e} (+{1.96*se:.1e}) ESS={ess:.0f}")
    if non.sum()>0:
        for Cc in [0.5,1,2,4]:
            m=np.abs(Jt[:,0])<=Cc*r
            print(f"[{tag}]   |q|<={Cc}r: gen-share={(non&m).sum()/non.sum():.2f}"+(f" wgt-share={(Wn*(non&m)).sum()/wg:.2f}" if wg>0 else ""))
        print(f"[{tag}] fail means(q,a,w,z)=",[f"{Jt[non][:,k].mean():.3f}" for k in range(4)],"n_fail",int(non.sum()))
    sub=Jt[:3000]
    v1=run(sub,r,qu,-1.0,dt/2)[0]|run(sub,r,qu,+1.0,dt/2)[0]
    print(f"[{tag}] dt/2 check: mismatches={int((v1!=A[:3000]).sum())} of 3000 (must be 0)")
cfg=sys.argv[1]
if cfg=="off0125":
    rng0=np.random.default_rng(20260720)
    q0=rng0.normal(-b,np.sqrt(2),30000);a0=rng0.normal(0,np.sqrt(2),30000)
    w0=rng0.normal(0,np.sqrt(2),30000);z0=rng0.normal(0,np.sqrt(6),30000)
    analyze("Q4off r=0.0125",np.column_stack([q0,a0,w0,z0,np.zeros((30000,5))]),0.0125,False)
elif cfg=="on025":
    analyze("Q4on r=0.025",sample9(30000,0.025,np.random.default_rng(20260720+40)),0.025,True)
    Jc=sample9(3000,0.025,np.random.default_rng(99)); Jz=Jc.copy(); Jz[:,4:]=0
    _,_,Tz=hess(Jz,0.025); Jtz=Jz[Tz]
    a1=run(Jtz,0.025,True,-1.0,0.025/8)[0]|run(Jtz,0.025,True,+1.0,0.025/8)[0]
    a2=run(Jtz,0.025,False,-1.0,0.025/8)[0]|run(Jtz,0.025,False,+1.0,0.025/8)[0]
    print(f"[control] quartic-zeroed vs Q4-off mismatches: {int((a1!=a2).sum())} of {len(a1)} (must be 0)")
elif cfg=="on0125":
    analyze("Q4on r=0.0125",sample9(30000,0.0125,np.random.default_rng(20260720+80)),0.0125,True)
