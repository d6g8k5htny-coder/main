# Independent physical-coordinate test of LS-DER-033 (8.1): explicit quartic polynomial field
# with the six pins and the windowed witness imposed exactly; free jets fixed at O(1) values.
import mpmath as mp, itertools
mp.mp.dps=60
kap=mp.mpf(1); b=mp.mpf('1.2'); al=mp.mpf('0.5')
monos=[(i,j) for d in range(5) for i in range(d+1) for j in [d-i]]  # x^i z^j, degree<=4
def val(cf,x,z): return sum(cf[m]*x**m[0]*z**m[1] for m in monos)
def rows(x,z):
    v=[x**i*z**j for (i,j) in monos]
    gx=[(i*x**(i-1)*z**j if i>0 else 0) for (i,j) in monos]
    gz=[(j*x**i*z**(j-1) if j>0 else 0) for (i,j) in monos]
    return v,gx,gz
def hess(cf,x,z):
    fxx=sum(cf[m]*m[0]*(m[0]-1)*x**max(m[0]-2,0)*z**m[1] for m in monos if m[0]>=2)
    fzz=sum(cf[m]*m[1]*(m[1]-1)*x**m[0]*z**max(m[1]-2,0) for m in monos if m[1]>=2)
    fxz=sum(cf[m]*m[0]*m[1]*x**(m[0]-1)*z**(m[1]-1) for m in monos if m[0]>=1 and m[1]>=1)
    return fxx*fzz-fxz**2
free={(1,2):mp.mpf('0.15'),(4,0):mp.mpf('0.7')/24,(3,1):mp.mpf('-1.1')/6,(2,2):mp.mpf('0.9')/4,(1,3):mp.mpf('0.33')/6,(0,4):mp.mpf('-0.5')/24}
unk=[m for m in monos if m not in free]
def solve(r,t,c,s):
    M=(-r/2,0);S=(r/2,0);y=(t*c,t*s)
    A=[];rhs=[]
    def add(row,target):
        A.append([row[monos.index(m)] for m in unk]); rhs.append(target-sum(row[monos.index(m)]*free[m] for m in free))
    for (pt,fv) in ((M,b),(S,b-kap*r**3/6),(y,b-al*kap*r**3/6)):
        v,gx,gz=rows(*pt); add(v,fv); add(gx,0); add(gz,0)
    sol=mp.lu_solve(mp.matrix(A),mp.matrix(rhs))
    cf=dict(free); cf.update({m:sol[i] for i,m in enumerate(unk)})
    return cf,M,S,y
c,s=mp.mpf(3)/5,mp.mpf(4)/5
print('ratio |detH_M detH_S detH_y| / r^6, and endpoint |detH_M|/r^2, third |detH_y|/r^2')
for beta in (mp.mpf(1)/2,mp.mpf(2)/3,mp.mpf(3)/4,mp.mpf(9)/10):
    out=[]
    for r in (mp.mpf('1e-2'),mp.mpf('1e-4'),mp.mpf('1e-6'),mp.mpf('1e-8')):
        t=r**beta
        cf,M,S,y=solve(r,t,c,s)
        dM,dS,dY=hess(cf,*M),hess(cf,*S),hess(cf,*y)
        out.append((mp.nstr(abs(dM*dS*dY)/r**6,3),mp.nstr(abs(dM)/r**2,3),mp.nstr(abs(dY)/r**2,3)))
    print('t=r^%s'%mp.nstr(beta,3),out)
# fixed t, r->0: intensity proxy E-free: |dM dS|/r^2 and |dY|
for t in (mp.mpf('0.05'),):
    for r in (mp.mpf('1e-3'),mp.mpf('1e-5'),mp.mpf('1e-7')):
        cf,M,S,y=solve(r,t,c,s)
        dM,dS,dY=hess(cf,*M),hess(cf,*S),hess(cf,*y)
        fzzM=sum(cf[m]*m[1]*(m[1]-1)*M[0]**m[0] for m in monos if m[1]==2)
        print('t=0.05 r=',mp.nstr(r,2),' |dMdS|/r^2=',mp.nstr(abs(dM*dS)/r**2,6),' |dY|=',mp.nstr(abs(dY),6),' f_zz(M)=',mp.nstr(fzzM,6),' 2t^2P4h/s^2 pred=',mp.nstr(2*t**2*(free[(4,0)]*c**4+free[(3,1)]*c**3*s+free[(2,2)]*c**2*s**2+free[(1,3)]*c*s**3+free[(0,4)]*s**4)/s**2,6), ' (P2 P13: dM/r ~ -6k S0 =',mp.nstr(dM/r,6),'vs',mp.nstr(-kap*fzzM,6),')')
