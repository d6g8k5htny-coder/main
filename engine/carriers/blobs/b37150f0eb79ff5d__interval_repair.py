#!/usr/bin/env python3
"""R05 author-side interval recomputation; see 03_RAYLEIGH_REPAIR_AND_INTERVAL_CERTIFICATE.md for proof.
Not independent of the received derivation. No source artifact is modified.
mpmath.iv bounds every finite arithmetic/transcendental operation. Full 1D
geometric and 2D shell tails are included. Endpoint 31/250 is an imported
analytic theorem, not established by this numeric program. All claims exact.
"""
from mpmath import iv, mp
from fractions import Fraction as F
import json, sys, argparse
iv.dps=65;mp.dps=40
p=argparse.ArgumentParser();p.add_argument('--claim',default='6.238e-44');args=p.parse_args()
C=[]
def ck(name,v):
 if v is not True: raise SystemExit('FAIL:'+name)
 C.append(name)
def q(n,d=1):return iv.mpf(n)/d
def rat_endpoint(x,idx):
 sign,man,e,bc=x._mpi_[idx]
 if bc<0:raise ValueError('nonfinite endpoint')
 return F(-man if sign else man)*F(2)**e
def lo(x):return rat_endpoint(x,0)
def hi(x):return rat_endpoint(x,1)
def U(x):return x.b
def L(x):return x.a
def upsum(xs):return sum(xs,q(0))
def cap(name,x):
 return {'lower_exact':str(lo(x)),'upper_exact':str(hi(x)),
         'display_lower':mp.nstr(mp.mpf(lo(x).numerator)/lo(x).denominator,16),
         'display_upper':mp.nstr(mp.mpf(hi(x).numerator)/hi(x).denominator,16)}
pi=iv.pi;h=pi/12;a=h*h/2;N=70;R=q(1,100000);d=q(1,1024);lam0=q(31,250)
# The real-mode amplitude is Rayleigh, not |xi|+|eta|.
mu1=2*iv.sqrt(2/pi);mu4=12+16/pi
ck('Rayleigh_first_moment_budget',lo(mu1-iv.sqrt(pi/2))>0)
ck('Rayleigh_fourth_moment_budget',lo(mu4-8)>0)
ck('absolute_sum_fourth_moment_identity_is_different',lo((12+32/pi)-mu4)>0)
w=[iv.exp(-a*n*n) for n in range(N+1)];ztr=w[0]+2*upsum(w[1:])
k=[h*n for n in range(N+1)]
tails={}
for power in range(13):
 n=N+1;rho=iv.exp(-a*(2*n+1))*(1+q(1,n))**power
 ck('moment_tail_decrease_'+str(power),lo(2*a*n*n-power)>0)
 ck('moment_tail_ratio_'+str(power),hi(rho)<F(1,2))
 tails[power]=U(2*iv.exp(-a*n*n)*(h*n)**power/(1-rho))
zall=iv.mpf([ztr.a,(ztr+tails[0]).b]);mom=[]
for i in range(7):
 if i==0:mom.append(q(1));continue
 s=2*upsum(w[n]*k[n]**(2*i) for n in range(1,N+1))
 mom.append(iv.mpf([(s/zall).a,((s+tails[2*i])/ztr).b]))
def absmom(n):return U(mom[n//2]) if n%2==0 else U(iv.sqrt(U(mom[(n-1)//2])*U(mom[(n+1)//2])))
A=[(0,0),(1,0),(0,1),(2,0),(1,1),(3,0),(0,2),(2,1),(1,2),(0,3)]
SCALE=[q(1),q(1),q(1),q(1,2),q(1),q(1,6),q(1),q(1,2),q(1,2),q(1,6)]
G=[[q(0) for j in range(10)] for i in range(10)]
for i in range(10):
 for j in range(10):
  xx=A[i][0]+A[j][0];yy=A[i][1]+A[j][1]
  if xx%2 or yy%2:continue
  sign=(-1)**(sum(A[j])+xx//2+yy//2)
  G[i][j]=sign*SCALE[i]*SCALE[j]*mom[xx//2]*mom[yy//2]
# Verified integral profile bounds, as described in the analytic companion.
PB=[None,q(2),q(1),q(1,2),q(1),q(1,4),q(1),q(1,2),q(1,2),q(1,6)]
DP=[None,q(5,4),q(1),q(1,4),q(1,2),q(5,4),q(0),q(0),q(0),q(0)]
def pb(i):return {0:q(1),1:R/4} if i==0 else {0:PB[i]}
def dp(i):return {0:q(1,2),1:R/4} if i==0 else {0:DP[i]}
def pmul(a,b):
 out={}
 for u,x in a.items():
  for v,y in b.items():out[u+v]=out.get(u+v,q(0))+x*y
 return out
def padd(a,b):
 out=dict(a)
 for k,v in b.items():out[k]=out.get(k,q(0))+v
 return out
C1=[[q(0) for j in range(10)] for i in range(10)]
for i in range(10):
 for j in range(10):
  xx=A[i][0]+A[j][0];yy=A[i][1]+A[j][1]
  if (sum(A[i])-sum(A[j]))%2 or yy%2:continue
  pol=padd(pmul(dp(i),pb(j)),pmul(pb(i),dp(j)))
  C1[i][j]=U(U(mom[yy//2])*upsum(coef*absmom(xx+1+power) for power,coef in pol.items())/2)
L1=U(iv.sqrt(upsum(t*t for row in C1 for t in row)))
LGU=U(iv.sqrt(upsum(C1[i][j]**2 for i in range(6) for j in range(6))))
LGX=U(iv.sqrt(upsum(C1[i][j]**2 for i in range(6) for j in range(6,10))))
lam=L(lam0-L1*R);ck('uniform_covariance_floor_positive',lo(lam)>0)
D=[U(abs(G[i][i])+C1[i][i]*R) for i in range(10)]
tr6=U(upsum(D[:6]));tr10=U(upsum(D))
guf2=U(upsum((U(abs(G[i][j]))+C1[i][j]*R)**2 for i in range(6) for j in range(6)))
gx0=U(iv.sqrt(upsum(G[i][j]**2 for i in range(6) for j in range(6,10))))
b=q(6,5);u=U(iv.sqrt(q(5809,3600)));du=U(R**2*iv.sqrt(q(1,16)+R**2/144))
M=U(U(mom[1])*b+(LGX*R/lam+gx0*LGU*R/(lam0*lam))*u+(gx0/lam0)*du)
RJ2=U(((10+2*d)*R)**2+(2+d)**2+2*d*d);RJ=U(iv.sqrt(RJ2))
# Full J covariance; A-D3 covariance is nonzero.
grow=[U(upsum(U(abs(G[i][j])) for j in range(6,10))) for i in range(6,10)]
La=max(grow,key=hi);ex=U((RJ+M)**2/(2*lam));dens=L(iv.exp(-ex)/(4*pi*pi*La*La))
ck('uniform_density_ge_1e_minus21',lo(dens)>=F(1,10**21))
# Real full-lattice representation, amplitude rho_k=sqrt(xi_k^2+eta_k^2).
Sp={}
for power in (3,4):
 terms=[]
 for n in range(N+1):
  for m in range(N+1):
   mult=(1 if n==0 else 2)*(1 if m==0 else 2)
   terms.append(mult*iv.exp(-a*(n*n+m*m)/2)*(1+iv.sqrt(k[n]**2+k[m]**2))**power)
 s=upsum(terms);n=N+1
 def shell(j):return 8*j*iv.exp(-a*j*j/2)*(1+h*iv.sqrt(q(2))*j)**power
 rho=U(shell(n+1)/shell(n));ck('shell_ratio_'+str(power),hi(rho)<1)
 # Consecutive shell ratios decrease: each factor does, for j>=N+1.
 tail=U(shell(n)/(1-rho));Sp[power]=U((s+tail)/L(ztr))
def maxvar(order):return max([U(mom[i])*U(mom[j]) for i in range(order+1) for j in range(order+1-i)],key=hi)
A3=U(iv.sqrt(maxvar(3)*tr6)/lam);A4=U(iv.sqrt(maxvar(4)*tr10)/lam)
EU4=U(tr6**2+2*guf2);EV=U(iv.sqrt(tr10));v=U(iv.sqrt(u*u+RJ2))
field3=U(mu4*Sp[3]**4);field4=U(mu1*Sp[4])
B3=U((1+iv.sqrt(iv.sqrt(field3))+A3*(iv.sqrt(iv.sqrt(EU4))+u))**4)
B4=U(field4+A4*(EV+v))
def ceilF(x):return -(-x.numerator//x.denominator)
B3rat=ceilF(hi(B3));Krat=ceilF(2*hi(B4));r0=F(1,256*Krat);clo=F(260,B3rat*2**40*10**21)
ck('hard_geometry',16*F(1,1024)+8*Krat*r0==F(3,64)<F(1,16))
ck('radius_within_covariance_domain',r0<=F(1,100000))
ck('claimed_decimal_is_downward_safe',F(args.claim)<=clo)
ck('path_clearance',F(1,6)-F(99,1280)-F(1,16)==F(103,3840)>0)
ck('weight_base',F(81,16)*F(1673,128)>=65)
ck('power',1+4-2==3)
vals={'L1':L1,'lam':lam,'M':M,'R_J':RJ,'Lambda':La,'exponent':ex,'density_lower':dens,'S3_upper':Sp[3],'S4_upper':Sp[4],'A3':A3,'A4':A4,'B3_upper':B3,'B4_upper':B4,'J_A_D3_covariance':G[7][9]}
out={'status':'PASS_AT_STATED_ANALYTIC_REPAIR_AND_INTERVAL_SCOPE','checks':C,'check_count':len(C),'precision_decimal_digits':iv.dps,'B3_rational_ceiling':B3rat,'K_rational_ceiling':Krat,'radius_exact':str(r0),'c_floor_exact':str(clo),'c_floor_display':mp.nstr(mp.mpf(clo.numerator)/clo.denominator,32),'claimed_decimal':args.claim,'intervals':{k:cap(k,v) for k,v in vals.items()},'limitations':['Author-side new repair, not independent of received mathematical design','Imports exact LPW deterministic proof and proved endpoint eigenfloor','Finite arithmetic enclosed with mpmath.iv; not a Lean proof','Infinite-tail and global analytic profile/conditional-regression lemmas are written in companion, not proved by finite tests','No promotion or Drive changes performed by this script']}
print(json.dumps(out,indent=2,sort_keys=True))
