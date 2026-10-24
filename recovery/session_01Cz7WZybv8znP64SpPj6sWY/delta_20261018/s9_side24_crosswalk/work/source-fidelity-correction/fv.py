# Independent first-variation of P (15.2) for product kernel K=k(x1)k(x2)k(x3); written from P (15.1)-(15.2) only.
import itertools, mpmath as mp
mp.mp.dps=60
def kd(n,e):  # 1-D jet k^(n)(0) with perturbation e=(e2,e4,e6)
    base={0:1,2:-1,4:3,6:-15}
    if n%2: return mp.mpf(0)
    return mp.mpf(base[n])+(0 if n==0 else e[n//2-1])
def KD(alpha,e):  # partial^alpha K(0)
    c=[0,0,0]
    for a in alpha: c[a]+=1
    r=mp.mpf(1)
    for n in c: r*=kd(n,e)
    return r
def dirderiv(vecs,e):
    # derivative of K at 0 along list of frame vectors (each a 3-vector)
    tot=mp.mpf(0)
    for idx in itertools.product(range(3),repeat=len(vecs)):
        w=mp.mpf(1)
        for v,i in zip(vecs,idx):
            w*=v[i]
            if w==0: break
        if w!=0: tot+=w*KD(idx,e)
    return tot
def frame(u):
    u=mp.matrix(u); u=u/mp.norm(u)
    a=mp.matrix([1,0,0]) if abs(u[0])<0.9 else mp.matrix([0,1,0])
    v1=a-(a.T*u)[0]*u; v1=v1/mp.norm(v1)
    v2=mp.matrix([u[1]*v1[2]-u[2]*v1[1],u[2]*v1[0]-u[0]*v1[2],u[0]*v1[1]-u[1]*v1[0]])
    return [list(u),list(v1),list(v2)]
def cov(da,db,R,e):  # Cov(partial^da f, partial^db f) in frame; da,db tuples of frame indices
    vecs=[R[i] for i in da]+[R[i] for i in db]
    return (-1)**len(db)*dirderiv(vecs,e)
D0=mp.mpf(29)/6-mp.sqrt(6); Ess=mp.mpf(445)/18-5*mp.sqrt(6)/8; Exx=mp.mpf(19)/2-53*mp.sqrt(6)/16
G0=mp.matrix([[mp.mpf(8)/3,mp.mpf(2)/3,0],[mp.mpf(2)/3,mp.mpf(8)/3,0],[0,0,1]])
M=mp.matrix([[Ess+Exx,Ess-Exx,0],[Ess-Exx,Ess+Exx,0],[0,0,Exx]])  # E0[h w w^T], w=(A22,A33,A23)
def blocks(R,e):
    Gi=[(0,),(1,),(2,)]
    SG=mp.matrix(3,3)
    for i in range(3):
        for j in range(3): SG[i,j]=cov(Gi[i],Gi[j],R,e)
    t=(0,0,0)
    ct=mp.matrix([cov(t,g,R,e) for g in Gi])
    vt=cov(t,t,R,e)
    tau2=vt-(ct.T*mp.inverse(SG)*ct)[0]
    V=[(0,0),(0,1),(0,2)]; A=[(1,1),(2,2),(1,2)]
    def C(X,Y):
        m=mp.matrix(len(X),len(Y))
        for i in range(len(X)):
            for j in range(len(Y)): m[i,j]=cov(X[i],Y[j],R,e)
        return m
    SV=C(V,V); SAV=C(A,V); SAA=C(A,A)
    Gam=SAA-SAV*mp.inverse(SV)*SAV.T
    return SG,tau2,SV,Gam
def logF_parts(R,e):
    SG,tau2,SV,Gam=blocks(R,e)
    lpG=-mp.log(mp.det(SG))/2; lpV=-mp.log(mp.det(SV))/2; lt=(mp.mpf(2)/3)*mp.log(tau2)
    return lpG,lpV,lt,Gam
def dlogF(R,n,h=mp.mpf('1e-25')):
    ep=[0,0,0]; em=[0,0,0]; ep[n]=h; em[n]=-h
    a=logF_parts(R,ep); b=logF_parts(R,em)
    d=sum((a[i]-b[i])/(2*h) for i in range(3))
    dG=(a[3]-b[3])/(2*h)
    Gi=mp.inverse(G0)
    dD=(Gi*dG*Gi*M).apply(lambda x:x)  # matrix
    tr1=sum((Gi*dG*Gi*M)[i,i] for i in range(3))/2-sum((Gi*dG)[i,i] for i in range(3))*D0/2
    return d+tr1/D0
# reference sanity
R0=frame([1,0,0]); SG,tau2,SV,Gam=blocks(R0,[0,0,0])
print('ref tau2',mp.nstr(tau2,12),'detSV',mp.nstr(mp.det(SV),12),'Gam',[mp.nstr(Gam[i,j],8) for i in range(3) for j in range(3)])
# sphere average: Gauss-Legendre in z (nz) x trapezoid phi (nphi)
def avg(n,nz,nphi):
    xs,ws=zip(*[(x,w) for x,w in zip(*gl(nz))])
    tot=mp.mpf(0)
    for x,w in zip(xs,ws):
        r=mp.sqrt(1-x*x)
        for k in range(nphi):
            ph=2*mp.pi*(k+mp.mpf(1)/3)/nphi
            u=[r*mp.cos(ph),r*mp.sin(ph),x]
            tot+=w*dlogF(frame(u),n)/nphi
    return tot/2
def gl(n):
    xs=[mp.cos(mp.pi*(i+1-mp.mpf(1)/4)/(n+mp.mpf(1)/2)) for i in range(n)]
    # refine nodes with mpmath
    X=[];W=[]
    for x0 in xs:
        x=mp.findroot(lambda t: mp.legendre(n,t), mp.mpf(x0))
        dp=mp.diff(lambda t: mp.legendre(n,t),x)
        X.append(x); W.append(2/((1-x*x)*dp*dp))
    return X,W
for nz,nphi in [(4,8),(5,10)]:
    r=[avg(n,nz,nphi) for n in range(3)]
    print(nz,nphi,[mp.nstr(v,25) for v in r], [mp.nstr(v-t,3) for v,t in zip(r,[mp.mpf(39)/70,-mp.mpf(1)/70,-mp.mpf(1)/21])])
