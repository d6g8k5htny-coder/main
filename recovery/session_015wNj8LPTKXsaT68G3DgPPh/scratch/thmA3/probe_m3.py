import math, random
from math import erf, sqrt, exp, pi
def Phi(x): return 0.5*(1+erf(x/sqrt(2)))
def phi(x): return exp(-x*x/2)/sqrt(2*pi)
def Mj(x, J):
    M=[1-Phi(x), phi(x)]
    for j in range(2,J+1): M.append(phi(x)*x**(j-1)+(j-1)*M[j-2])
    return M
def Es_pos(b, n):  # E[s^n 1{s>0}], s~N(b,1)
    M=Mj(-b, n)
    return sum(math.comb(n,j)*b**(n-j)*M[j] for j in range(n+1))
def m3(b):
    return Es_pos(b,4)-4*Es_pos(b,2)+8*Es_pos(b,0)-4*sqrt(2)*exp(-b*b/4)*Phi(b/sqrt(2))
def m3_mc(b, n=400000, seed=1):
    rng=random.Random(seed); tot=0
    for _ in range(n):
        g11=rng.gauss(0,sqrt(2)); g22=rng.gauss(0,sqrt(2)); g12=rng.gauss(0,1)
        a=b+g11; c=b+g22; det=a*c-g12*g12
        if a>0 and det>0: tot+=det*det
    return tot/n
def m3_quad(b, N=2000):
    # direct 2D quadrature of the eigenvalue density
    h=12.0/N; tot=0
    for i in range(N):
        l1=(i+0.5)*h
        for j in range(i,N):
            l2=(j+0.5)*h
            tot+=l1*l1*l2*l2*(l2-l1)*exp(-((l1-b)**2+(l2-b)**2)/4)
    return tot*h*h/(4*sqrt(2*pi))
print('m3(0) closed', m3(0), 'exact', (7-4*sqrt(2))/2)
for b in (0, 0.5, 1.0):
    print(b, m3(b), m3_mc(b), m3_quad(b, 600))
def I(j, m, s2=2.0, N=20000, top=30.0):
    h=top/N
    return sum(((i+0.5)*h)**j*exp(-((i+0.5)*h-m)**2/(2*s2)) for i in range(N))*h
def R3R(b):
    num=exp(-b*b/4)*I(3,b)/(4*sqrt(2*pi))
    return num/m3(b)
r0=R3R(0)
print('R3R(0)', r0, ' expected R3(0)R(0)=', (32+28*sqrt(2))/17/(2*sqrt(pi)))
for b in (0.001,0.01,0.05,0.1,0.25,0.5,1.0):
    print(b, R3R(b)/r0)
