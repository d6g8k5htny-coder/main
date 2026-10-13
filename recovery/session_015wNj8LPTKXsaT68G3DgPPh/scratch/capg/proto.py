import math, random
from math import erf, exp, sqrt, pi
def Phi(x): return 0.5*(1+erf(x/sqrt(2)))
def phi(x): return exp(-x*x/2)/sqrt(2*pi)
def R(b):
    be=b/sqrt(2); return (phi(be)/sqrt(2))/((2+b*b)*Phi(be)+sqrt(2)*b*phi(be))
def I_of(k,v,om,Y):
    a=v*v; c=6*k*om-v*v; M=max(12*k,2*abs(v),abs(om),abs(Y)); top=8*M*M; lo=max(a,c)
    if top<=lo: return 0.0
    F=lambda s: s**3/3-(a+c)*s*s/2+a*c*s
    return (F(top)-F(lo))/(6*k)
def J_mc(k,n=400000,seed=1):
    rng=random.Random(seed); s=0.0; s2=0.0
    for _ in range(n):
        v=rng.gauss(0,sqrt(0.5)); om=rng.gauss(0,sqrt(2)); Y=rng.gauss(0,sqrt(6))
        x=I_of(k,v,om,Y)/(36*k*k); s+=x; s2+=x*x
    m=s/n; return m, sqrt((s2/n-m*m)/n)
# Y-part in closed form: E_Y[g(max(m0,|Y|))] for g(M)=poly in M^2 ; here I as function of M given (a,c): (F(8M^2)-F(lo))/(6k)
def EY_poly(m0, sig, coeffs):
    # E[ sum_j coeffs[j] * max(m0,|Y|)^(2j) ], Y~N(0,sig^2)
    # = sum_j c_j [ m0^(2j) P(|Y|<=m0) + 2*int_{m0}^inf y^(2j) phi_sig(y) dy ]
    t=m0/sig; P_in=2*Phi(t)-1
    # int_t^inf u^(2j) phi(u) du via recursion: A_j = t^(2j-1) phi(t) + (2j-1) A_{j-1}, A_0 = 1-Phi(t)
    A=[1-Phi(t)]
    for j in range(1,len(coeffs)):
        A.append(t**(2*j-1)*phi(t)+(2*j-1)*A[j-1])
    tot=0.0
    for j,cj in enumerate(coeffs):
        tot+=cj*(m0**(2*j)*P_in + 2*sig**(2*j)*A[j])
    return tot
def J_quad(k, nv=400, no=400, V=7.0, O=11.0):
    # integrand in (v,om): E_Y[I]; F(8M^2) = (8M^2)^3/3 - (a+c)(8M^2)^2/2 + a c (8M^2)
    tot=0.0
    sv=sqrt(0.5); so=sqrt(2)
    for i in range(nv):
        v=-V+(i+0.5)*(2*V/nv); wv=phi(v/sv)/sv*(2*V/nv)
        for j in range(no):
            om=-O+(j+0.5)*(2*O/no); wo=phi(om/so)/so*(2*O/no)
            a=v*v; c=6*k*om-a; lo=max(a,c); m0=max(12*k,2*abs(v),abs(om))
            # assume 8 m0^2 > lo (check) 
            assert 8*m0*m0>lo
            coeffs=[-(lo**3/3-(a+c)*lo*lo/2+a*c*lo), 8*a*c, -(a+c)*64/2, 512/3]
            tot+=wv*wo*EY_poly(m0,sqrt(6),coeffs)/(6*k)
    return tot/(36*k*k)
for k in [1/6,1/2,3/4,1,3/2,2]:
    jq=J_quad(k,200,200); lead=2359296*k**3
    print('k=%g  J_quad=%.6g  lead=%.6g  ratio=%.5f' % (k,jq,lead,jq/lead))
for k in [1/6,2]:
    m,se=J_mc(k,200000); print('k=%g  J_mc=%.6g +- %.3g'%(k,m,se))
for b in [0,0.25,0.5,1,1.2]:
    print('b=%g R=%.6g'%(b,R(b)))
print('c_G(0,2)=%.4g  c_G(6/5,1/6)=%.4g  c_G(0,1/2)=%.4g'%(R(0)*J_quad(2,200,200), R(1.2)*J_quad(1/6,200,200), R(0)*J_quad(0.5,200,200)))
