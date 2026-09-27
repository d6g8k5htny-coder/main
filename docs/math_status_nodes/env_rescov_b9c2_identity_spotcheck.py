# Independent spot-check of RN_WHITENED_JET_THEOREM.md (b9c2, sha256 1324a006...) eqs (1)-(4),(7),(10)-(11),(14)-(15)
# on random polynomial matrix paths. Numerical evidence for the identities; NOT a SIDE24 certificate.
import mpmath as mp, random, json, math
mp.mp.dps=60; random.seed(20260925); d=4; N=4
def rs(sym=True,scale=1):
    M=mp.matrix(d,d)
    for i in range(d):
        for j in range(d): M[i,j]=mp.mpf(random.uniform(-1,1))*scale
    return (M+M.T)/2 if sym else M
def rv(scale=1): return mp.matrix([mp.mpf(random.uniform(-1,1))*scale for _ in range(d)])
fact=math.factorial; binom=math.comb
res={}
# --- path A(t)=sum A_j t^j/j!, m(t) likewise (raw jets = coefficients)
Aj=[rs(scale=0.15)]+[rs(scale=0.1) for _ in range(N)]
mj=[rv(0.2)]+[rv(0.1) for _ in range(N)]
A=lambda t: sum((Aj[j]*t**j/fact(j) for j in range(1,N+1)),Aj[0])
m=lambda t: sum((mj[j]*t**j/fact(j) for j in range(1,N+1)),mj[0])
I=mp.eye(d)
def ell(t):
    a=A(t); mm=m(t)
    return -mp.log(mp.det(I-a*a))/2 + (mm.T*mp.inverse(I+a)*mm)[0]
def inv_jets(X):  # eq (2)
    dd=X[0].rows
    R=[mp.inverse(X[0])]
    for n in range(1,len(X)):
        s=mp.zeros(dd,dd)
        for j in range(1,n+1): s+=binom(n,j)*X[j]*R[n-j]
        R.append(-R[0]*s)
    return R
B=inv_jets([I+Aj[0]]+[Aj[j] for j in range(1,N+1)])
C=inv_jets([I-Aj[0]]+[-Aj[j] for j in range(1,N+1)])
def ell_n(n):  # eq (4)
    s=0
    for j in range(n): s+= -mp.mpf(binom(n-1,j))/2*sum(((B[j]-C[j])*Aj[n-j])[i,i] for i in range(d))
    for i in range(n+1):
        for j in range(n+1-i):
            k=n-i-j; s+=mp.mpf(fact(n))/(fact(i)*fact(j)*fact(k))*(mj[i].T*B[j]*mj[k])[0]
    return s
ref=[mp.diff(ell,0,n) for n in range(N+1)]
res['eq4_rel_err']=[mp.nstr(abs(ell_n(n)-ref[n])/abs(ref[n]),5) for n in range(1,N+1)]
# eq (1) vs direct Gaussian integral in d=1
S=mp.mpf('0.7'); mu=mp.mpf('0.3')
q=mp.quad(lambda x: (mp.npdf(x,mu,mp.sqrt(S))**2)/mp.npdf(x,0,1),[-mp.inf,mp.inf])
a=1-S; res['eq1_d1_rel_err']=mp.nstr(abs(mp.log(q)-(-mp.log(1-a*a)/2+mu**2/(1+a)))/mp.log(q),5)
# eq (7): chi2 jets
X=lambda t: mp.e**ell(t)-1; l=[ell_n(n) for n in range(1,5)]; E=mp.e**ref[0]
X4=E*(l[3]+4*l[0]*l[2]+3*l[1]**2+6*l[0]**2*l[1]+l[0]**4)
res['eq7_X4_rel_err']=mp.nstr(abs(X4-mp.diff(X,0,4))/abs(X4),5)
# eq (10)-(11) bound |ell_n|<=L_n with a_j=||A_j||_2, c_j=||m_j||_2
nrm=lambda M: mp.sqrt(max(mp.eigsy(M.T*M)[0]))
a_=[nrm(M) for M in Aj]; c_=[mp.norm(v) for v in mj]
u=[sum(binom(n,j)*a_[j]*a_[n-j] for j in range(n+1)) for n in range(N+1)]
h=[1/(1-a_[0]**2)]; b=[1/(1-a_[0])]
for n in range(1,N+1):
    h.append(h[0]*sum(binom(n,j)*u[j]*h[n-j] for j in range(1,n+1)))
    b.append(b[0]*sum(binom(n,j)*a_[j]*b[n-j] for j in range(1,n+1)))
L=[mp.mpf(d)/2*sum(binom(n-1,j)*h[j]*u[n-j] for j in range(n))+sum(mp.mpf(fact(n))/(fact(i)*fact(j)*fact(n-i-j))*c_[i]*b[j]*c_[n-i-j] for i in range(n+1) for j in range(n+1-i)) for n in range(1,N+1)]
res['eq11_ratio_abs_ell_n_over_L_n']=[mp.nstr(abs(ref[n])/L[n-1],5) for n in range(1,N+1)]
res['eq11_holds']=all(abs(ref[n])<=L[n-1] for n in range(1,N+1))
# eq (14)-(15): Schur jets A=F D^-1 F^T, m=F D^-1 z on random polynomial F,D,z
k=3
def rm(r,c,s): 
    M=mp.matrix(r,c)
    for i in range(r):
        for j in range(c): M[i,j]=mp.mpf(random.uniform(-1,1))*s
    return M
Fj=[rm(d,k,0.3)]+[rm(d,k,0.1) for _ in range(N)]
D0=rm(k,k,0.2); Dj=[D0*D0.T+mp.eye(k)]+[ (lambda M:(M+M.T)/2)(rm(k,k,0.1)) for _ in range(N)]
zj=[rm(k,1,0.3)]+[rm(k,1,0.1) for _ in range(N)]
P=lambda L_,t: sum((L_[j]*t**j/fact(j) for j in range(1,N+1)),L_[0])
Af=lambda t,i,jj: (P(Fj,t)*mp.inverse(P(Dj,t))*P(Fj,t).T)[i,jj]
J=inv_jets(Dj)
def A_n(n): return sum((mp.mpf(fact(n))/(fact(i)*fact(j)*fact(n-i-j))*Fj[i]*J[j]*Fj[n-i-j].T for i in range(n+1) for j in range(n+1-i)),mp.zeros(d,d))
errs=[]
for n in range(1,N+1):
    An=A_n(n); errs.append(mp.nstr(max(abs(An[i,jj]-mp.diff(lambda t:Af(t,i,jj),0,n)) for i in range(d) for jj in range(d)),5))
res['eq15_A_n_max_abs_err']=errs
print(json.dumps(res,indent=1))
json.dump(res,open('/tmp/env_rescov_b9c2_identity_spotcheck_out.json','w'),indent=1)
