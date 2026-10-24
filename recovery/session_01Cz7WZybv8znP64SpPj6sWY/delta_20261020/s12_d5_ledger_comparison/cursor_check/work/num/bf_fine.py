# Copy of bfcheck.py mode A (finite r), with (i) n midpoint angles on (0,pi) instead of 8, (ii) numpy MC, independent seeds per angle, SE reported.
import sys; sys.path.insert(0, sys.argv[1])
import mpmath as mp, numpy as np
exec(open(sys.argv[2]).read().split("mode=sys.argv[1]")[0])   # defs from the record's bfcheck.py (He, cov, condlaw, factor, gdens, dets, Wfn, b, kap, pins)
def npmc(mean,C,N,seed,third):
    m=np.array([float(v) for v in mean]); n=len(m)
    L=np.array(factor(C)); rng=np.random.default_rng(seed); out=[]
    for _ in range(N//100000):
        v=m+rng.standard_normal((100000,n))@L.T
        dM=v[:,0]*v[:,2]-v[:,1]**2; dS=v[:,3]*v[:,5]-v[:,4]**2
        w=np.where((dM>0)&(v[:,0]<0)&(dS<0),np.abs(dM*dS),0.0)
        if third: w=w*np.abs(v[:,6]*v[:,8]-v[:,7]**2)
        out.append(w)
    w=np.concatenate(out); return w.mean(), w.std(ddof=1)/np.sqrt(len(w))
r=mp.mpf(sys.argv[3]); t=mp.mpf(sys.argv[4]); n=int(sys.argv[5]); N=int(sys.argv[6])
M,S,obs,tgt=pins(r)
mZ,cZ,_=condlaw(obs,tgt,[(M,a) for a in H]+[(S,a) for a in H]); Z,Zse=npmc(mZ,cZ,4*N,1,False)
u=b-mp.mpf('0.5')*kap*r**3/6
vals=[];ses=[]
for j in range(n):
    th=mp.pi*(j+mp.mpf('0.5'))/n; y=(t*mp.cos(th),t*mp.sin(th))
    third=[(y,(0,0)),(y,(1,0)),(y,(0,1))]
    m3,C3,_=condlaw(obs,tgt,third); p=float(gdens(m3,C3,[u,0,0]))
    mean,C,_=condlaw(obs+third,tgt+[u,0,0],[(M,a) for a in H]+[(S,a) for a in H]+[(y,a) for a in H])
    E,se=npmc(mean,C,N,104729*n+j,True)
    vals.append(p*E); ses.append(p*se)
Lam=2*np.pi/n*np.sum(vals)/Z; SE=2*np.pi/n*np.sqrt(np.sum(np.array(ses)**2))/Z
relZ=Zse/Z
print(f'finite r={float(r)} t={float(t)} n_angles(0,pi)={n} N/angle={N}  Z/r^2={Z/float(r)**2:.4f}+-{Zse/float(r)**2:.4f}  Lambda={Lam:.5f} +- {SE:.5f} (angle MC)  [+-{Lam*relZ:.5f} from Z]  Lambda/(r^4 t^-6)={Lam/float(r**4*t**-6):.4e}',flush=True)
