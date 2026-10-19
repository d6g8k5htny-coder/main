import numpy as np, random, sys
sys.setrecursionlimit(10000)
def run(seed, r, ds, dt, smin, smax, tmax, bump):
    rnd=random.Random(seed)
    a=-1-rnd.random(); gam,B,eta=[rnd.uniform(-1,1) for _ in range(3)]
    alpha,beta,q=6.0,22.0,24.0
    F4=-36*r+3*gam**2/a; F5=q+10*eta*gam/a-15*gam**2*B/a**2
    lam,T=alpha*r**3,beta*r**2
    S=np.arange(int(round(smin/ds)),int(round(smax/ds))+1)*ds; Tt=np.arange(-int(round(tmax/dt)),int(round(tmax/dt))+1)*dt
    SS,TT=np.meshgrid(S,Tt,indexing='ij')
    V=(-lam*SS**2/2+a*TT**2/2+T*SS**3/6+gam*SS**2*TT/2+B*SS*TT**2/2+F4*SS**4/24+eta*SS**3*TT/6+F5*SS**5/120)
    bs,bt,bA,bw=bump; V+=bA*np.exp(-((SS-bs)**2+(TT-bt)**2)/(2*bw**2))
    ns,nt=V.shape; Vf=V.ravel(); i0=int(np.argmin(abs(S))); j0=int(np.argmin(abs(Tt))); M=i0*nt+j0; assert abs(Vf[M])<1e-18
    parent=np.full(ns*nt,-1,dtype=np.int64); top=np.empty(ns*nt); death=None
    def find(x):
        root=x
        while parent[root]!=root: root=parent[root]
        while parent[x]!=root: parent[x],x=root,parent[x]
        return root
    for idx in np.argsort(-Vf,kind='stable'):
        idx=int(idx); parent[idx]=idx; top[idx]=Vf[idx]; i,j=divmod(idx,nt)
        for di in (-1,0,1):
            ii=i+di
            if ii<0 or ii>=ns: continue
            for dj in (-1,0,1):
                jj=j+dj
                if (di or dj) and 0<=jj<nt and parent[ii*nt+jj]>=0:
                    ra,rb=find(idx),find(ii*nt+jj)
                    if ra==rb: continue
                    if death is None and parent[M]>=0:
                        rm=find(M)
                        if rm in (ra,rb):
                            other=rb if rm==ra else ra
                            if top[rm]==Vf[M] and top[other]>Vf[M]: death=Vf[idx]
                    if top[ra]<top[rb]: ra,rb=rb,ra
                    parent[rb]=ra
        if death is not None: break
    col=V.max(axis=1); g3=col[(S>2.3*r)&(S<3.7*r)].min(); g1=col[(S>0.5*r)&(S<1.5*r)].min()
    return a,gam,B,eta,death/r**5,g1/r**5,g3/r**5
r=float(sys.argv[1])
for seed in map(int,sys.argv[2:]):
    rnd=random.Random(seed); a=-1-rnd.random(); gam=rnd.uniform(-1,1)
    T=1.6*abs(gam)/(2*abs(a))*(5*r)**2+20*(2*r**5/abs(a))**0.5   # ridge offset at 5r plus transverse margin
    dt=0.25*r**2.5                                                # grid error |a|dt^2/8 << r^5
    amp=abs(a)*(0.8*T)**2/2*1.5+10*r**5
    out=run(seed,r,ds=r/200,dt=dt,smin=-r,smax=5*r,tmax=T,bump=(2.5*r,0.8*T,amp,0.05*T))
    print(f'r={r} seed={seed} a={out[0]:.3f} gam={out[1]:.3f} B={out[2]:.3f} eta={out[3]:.3f} | death/r^5={out[4]:.4f} ridge(s1)/r^5={out[5]:.4f} ridge(s3)/r^5={out[6]:.4f}',flush=True)
