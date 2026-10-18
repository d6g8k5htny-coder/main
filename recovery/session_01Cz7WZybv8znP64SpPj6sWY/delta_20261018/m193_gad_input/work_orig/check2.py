import numpy as np, itertools, math
rng=np.random.default_rng(1)
# finite-L lattice, d=3, direct sums
L=2.7; d=3
ks=np.arange(-40,41); q=2*np.pi*ks/L; w1=np.exp(-q**2/2); w1/=w1.sum()
a=(w1*q**2).sum(); b=(w1*q**4).sum(); c=(w1*q**6).sum()
k4=b-3*a*a; k6=c-15*a*b+30*a**3
# full tensor E[Q_i Q_j Q_k Q_l] by direct product-law summation
def Emon(p):
    out=1.0
    for pw in p: out*=(w1*q**pw).sum()
    return out
M=np.zeros((3,3,3,3))
for idx in itertools.product(range(3),repeat=4):
    p=[0]*3
    for i in idx: p[i]+=1
    M[idx]=Emon(p)
e=rng.normal(size=3); e/=np.linalg.norm(e)
# U orthonormal basis of e-perp
Qm,_=np.linalg.qr(np.column_stack([e,rng.normal(size=(3,2))])); U=Qm[:,1:]
B=np.einsum('ijkl,j,l->ik',M,e,e)
Bf=a*a*(np.eye(3)+2*np.outer(e,e))+k4*np.diag(e**2)
R=np.einsum('jkil,ja,kb,l->abi',M,U,U,e)
Rf=np.array([[[a*a*(1 if A==Bb else 0)*e[i]+k4*U[i,A]*U[i,Bb]*e[i] for i in range(3)] for Bb in range(2)] for A in range(2)])
C=np.einsum('ijkl,ia,jb,kc,ld->abcd',M,U,U,U,U)
dl=lambda x,y: 1.0 if x==y else 0.0
Cf=np.array([[[[a*a*(dl(A,Bb)*dl(Cc,Dd)+dl(A,Cc)*dl(Bb,Dd)+dl(A,Dd)*dl(Bb,Cc))+k4*np.sum(U[:,A]*U[:,Bb]*U[:,Cc]*U[:,Dd]) for Dd in range(2)] for Cc in range(2)] for Bb in range(2)] for A in range(2)])
print("B err",abs(B-Bf).max(),"R err",abs(R-Rf).max(),"C err",abs(C-Cf).max())
# Gamma_e via Schur complement on free entries vs eq(13)
Binv=np.linalg.inv(B)
G=C-np.einsum('abi,ij,cdj->abcd',R,Binv,R)
# brute: joint covariance of (U^T H U free entries, He) as linear functionals of H entries
free=[(0,0),(0,1),(1,1)]
rows=[]
for (A,Bb) in free: rows.append(np.einsum('i,j->ij',U[:,A],U[:,Bb]).ravel())
for i in range(3): rows.append(np.einsum('i,j->ij',np.eye(3)[i],e).ravel())
Lm=np.array(rows)
Mf=M.reshape(9,9)
J=Lm@Mf@Lm.T
Sch=J[:3,:3]-J[:3,3:]@np.linalg.inv(J[3:,3:])@J[3:,:3]
Gf=np.array([[G[A,Bb,Cc,Dd] for (Cc,Dd) in free] for (A,Bb) in free])
print("Gamma_e (13) vs Schur err",abs(Sch-Gf).max())
# V(e) direct regression vs formula
Q6=np.einsum('ijklmn,i,j,k,l,m,n->',np.zeros((1,)*6),*[np.ones(1)]*6) if False else None
# direct E(Q.e)^6 by 3D lattice
QQ=np.array(np.meshgrid(q,q,q,indexing='ij'))
W=np.einsum('i,j,k->ijk',w1,w1,w1)
Qe=np.einsum('xijk,x->ijk',QQ,e)
m6=(W*Qe**6).sum(); v=np.array([(W*QQ[i]*Qe**3).sum() for i in range(3)])
Vd=m6-v@v/a
Vf=6*a**3+9*a*k4*np.sum(e**4)+(k6-k4**2/a)*np.sum(e**6)
print("V err",Vd-Vf, "V",Vf)
# d=2 half: A scalar, E[A^2 1{A<0}] = var/2
print("d=2 half check: E[X^2;X<0]/Var =",0.5)
# units scaling check: scale L -> s L multiplies q by 1/s; a~s^-2. check C_alg scaling ~ s^-d numerically for d=2 integrand pieces
