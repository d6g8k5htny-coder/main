from math import gamma, pi, sqrt, exp
import random
# numeric: 144*int_0^inf k^{4/3} phi_tau(12k) dk vs Gamma(7/6) tau^{4/3}/(24^{1/3} sqrt(pi))
def phi(z,t): return exp(-z*z/(2*t*t))/sqrt(2*pi*t*t)
def integ(f,a,b,n=400000):
    h=(b-a)/n; s=0.5*(f(a)+f(b))
    for i in range(1,n): s+=f(a+i*h)
    return s*h
for tau in (0.7,1.0,2.3):
    lhs=144*integ(lambda k:k**(4/3)*phi(12*k,tau),0,2*tau)
    rhs=gamma(7/6)*tau**(4/3)/(24**(1/3)*sqrt(pi))
    print(tau,lhs,rhs,abs(lhs-rhs)/rhs)
# half-Gaussian: int_0^inf z^{4/3} phi_tau(z) dz = 2^{-1/3} Gamma(7/6) tau^{4/3}/sqrt(pi)
tau=1.3
print(integ(lambda z:z**(4/3)*phi(z,tau),0,12*tau), 2**(-1/3)*gamma(7/6)*tau**(4/3)/sqrt(pi))
# Jacobian of U_r rows on (F(M),F(S),Fu(M),Fu(S)) block, d=2 transverse pair -> 12 r^{-(d+3)}
import itertools
def det(m):
    n=len(m)
    if n==1: return m[0][0]
    return sum((-1)**j*m[0][j]*det([row[:j]+row[j+1:] for row in m[1:]]) for j in range(n))
r=0.37; d=3
# order: F(M),F(S),Fu(M),Fu(S),Fv1(M),Fv1(S),Fv2(M),Fv2(S)
rows=[[0.5,0.5,0,0,0,0,0,0],[-1/r,1/r,0,0,0,0,0,0],[0,0,-1/r,1/r,0,0,0,0],
      [12/r**3,-12/r**3,6/r**2,6/r**2,0,0,0,0],
      [0,0,0,0,0.5,0.5,0,0],[0,0,0,0,-1/r,1/r,0,0],[0,0,0,0,0,0,0.5,0.5],[0,0,0,0,0,0,-1/r,1/r]]
print(abs(det(rows)), 12*r**(-(d+3)))
# cubic pins: phi(s)=2k s^3-(3/2)k r^2 s + C ; phi(-r/2)-phi(r/2)=k r^3 ; phi''(-r/2)/r=-6k
k=1.7
ph=lambda s:2*k*s**3-1.5*k*r*r*s
print(ph(-r/2)-ph(r/2), k*r**3, 12*k*(-r/2)/r, -6*k)
