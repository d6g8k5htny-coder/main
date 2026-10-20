import sympy as sp
t,s=sp.symbols('t s',real=True)
K=sp.exp(-(t**2+s**2)/2)
def cov(a,b):
    return sp.simplify(((-1)**(b[0]+b[1])*sp.diff(K,t,a[0]+b[0],s,a[1]+b[1])).subs({t:0,s:0}))
def condvar(target,cond):
    S=sp.Matrix([[cov(i,j) for j in cond] for i in cond]); c=sp.Matrix([cov(target,j) for j in cond])
    return sp.simplify(cov(target,target)-(c.T*S.inv()*c)[0])
# full leading-order conditioning set implied by 12 pair numbers: 2-jet at S + f_ttt,f_tttt,f_ttttt (line data f,f_t,f_tt at M),
# f_tts,f_ttts (f_s,f_ts at M), f_tss (f_ss at M)
full=[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(4,0),(5,0),(2,1),(3,1),(1,2)]
print('Var(f_sss|full set)=',condvar((0,3),full))
print('Var(f_ttss|full set)=',condvar((2,2),full))
print('Var(f_ttss|f,f_tt,f_ss,f_tttt)=',condvar((2,2),[(0,0),(2,0),(0,2),(4,0)]))
# sd_t = (r^5 z_s^2/4) * sd(f_ttss|...) ; carrier says (z_s^2/2) r^5 -> need sd = 2
v=condvar((2,2),full); print('implied sd_t coefficient (z_s^2 r^5 *)', sp.nsimplify(sp.sqrt(v)/4))
# exact two-point check: numerically compute Var(f_t(y)|12 numbers)/r^10 for small r, z=(0,1)
import mpmath as mp
mp.mp.dps=60
def Kf(dt,ds): return mp.e**(-(dt**2+ds**2)/2)
def dK(a,b,dt,ds):
    return mp.diff(lambda x,y: Kf(x,y),(dt,ds),(a,b))
def covpt(a,pa,b,pb):
    # cov(d^a f(pa), d^b f(pb)) = (-1)^{|b|} d^{a+b} K(pa-pb)
    return (-1)**(b[0]+b[1])*dK(a[0]+b[0],a[1]+b[1],pa[0]-pb[0],pa[1]-pb[1])
def vt(r,zt,zs):
    S=(mp.mpf(0),mp.mpf(0)); M=(-r,mp.mpf(0)); y=(r**2*zt,r**2*zs)
    obs=[(d,p) for p in (S,M) for d in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]]
    A=mp.matrix(len(obs),len(obs))
    for i,(a,pa) in enumerate(obs):
        for j,(b,pb) in enumerate(obs): A[i,j]=covpt(a,pa,b,pb)
    out=[]
    for tgt in [(1,0),(0,1)]:
        c=mp.matrix([covpt(tgt,y,b,pb) for (b,pb) in obs])
        out.append(covpt(tgt,y,tgt,y)-(c.T*mp.inverse(A)*c)[0])
    return out
for r in [mp.mpf('0.02'),mp.mpf('0.01')]:
    vt_,vs_=vt(r,mp.mpf(0),mp.mpf(1))
    print('r',r,'sd_t/r^5',mp.nstr(mp.sqrt(vt_)/r**5,8),'sd_s/r^4',mp.nstr(mp.sqrt(vs_)/r**4,8),' c_s=',mp.nstr(mp.sqrt(1.5),8))
# multiplicative vs additive error in det H_S on tail set
r,kap,nu,eps,m=sp.symbols('r kappa nu epsilon m',real=True)
mu=m*r**2-r*nu**2/kap   # mut_S = m r^2 on tail set
detHS=(kap*r+eps*r**2)*(-mu)-(r*nu)**2
print('det H_S with f_tt(S)=kappa r+eps r^2, tail set:',sp.expand(detHS))
print('carrier form -kappa r mut_S:',sp.expand(-kap*r*m*r**2))
# Schur-complement identity: H z on null line of row 1 = (0, z_s det/a)
a,b,c,zs=sp.symbols('a b c z_s')
H=sp.Matrix([[a,b],[b,c]]); z=sp.Matrix([-b*zs/a,zs])
print('H z on null line:',sp.simplify(H*z).T,' det/a*z_s=',sp.simplify(H.det()/a*zs))
