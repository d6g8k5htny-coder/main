import math, random
from math import erf, exp, sqrt, pi
def phi(x): return exp(-x*x/2)/sqrt(2*pi)
def Phi(x): return 0.5*(1+erf(x/sqrt(2)))
def gauss_legendre(n):
    xs, ws = [], []
    for i in range(1, n+1):
        x = math.cos(math.pi*(i-0.25)/(n+0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for m in range(2, n+1): p0, p1 = p1, ((2*m-1)*x*p1-(m-1)*p0)/m
            dp = n*(x*p1-p0)/(x*x-1.0); dx = p1/dp; x -= dx
            if abs(dx) < 1e-15: break
        xs.append(x); ws.append(2/((1-x*x)*dp*dp))
    return xs, ws
R = gauss_legendre(200)
def gl(f, lo, hi, rule=R):
    mid, half = 0.5*(lo+hi), 0.5*(hi-lo)
    return half*sum(w*f(mid+half*x) for x, w in zip(*rule))
def dens(x, b): return phi((x+b)/sqrt(2))/sqrt(2)     # N(-b,2)
def m2(b): m, s = -b, sqrt(2); t = m/s; return (m*m+s*s)*Phi(-t) - m*s*phi(t)   # E[A^2 1{A<0}]
def m3odd(b): return gl(lambda x: (-x)**3*dens(x, b), -40, 0)                   # E[|A|^3 1{A<0}]
def inner(pq):  # E[(pq-g^2)^2 1{g^2<pq}], g~N(0,1), pq>0
    c = sqrt(pq)
    # ∫_{-c}^{c} (pq - g^2)^2 phi = pq^2 P + -2pq E[g^2 1] + E[g^4 1]; E[g^2 1{|g|<c}] = P - 2c phi(c); E[g^4 1] = 3P - 2c(c^2+3)phi(c)
    P = 2*Phi(c)-1; e2 = P-2*c*phi(c); e4 = 3*P-2*c*(c*c+3)*phi(c)
    return pq*pq*P - 2*pq*e2 + e4
def m3(b):  # E[(det A)^2 1{A neg def}], A 2x2 shifted GOE
    return gl(lambda p: dens(p, b)*gl(lambda q: dens(q, b)*inner(p*q), -40, 0), -40, 0)
def m3_mc(b, n=400000, seed=1):
    random.seed(seed); acc = 0.0; acc2 = 0.0
    for _ in range(n):
        p = -b+sqrt(2)*random.gauss(0,1); q = -b+sqrt(2)*random.gauss(0,1); g = random.gauss(0,1)
        d = p*q-g*g
        v = d*d if (p < 0 and q < 0 and d > 0) else 0.0
        acc += v; acc2 += v*v
    m = acc/n; se = sqrt((acc2/n-m*m)/n); return m, se
for b in (0.0, 1.0):
    M2, M3o, M3 = m2(b), m3odd(b), m3(b)
    R3 = sqrt(pi/2)*M2*M3o/M3
    mc = m3_mc(b)
    print(f'b={b}: m2={M2:.10f} E|A|^3 1(A<0)={M3o:.10f} m3(det^2,negdef)={M3:.10f} (MC {mc[0]:.5f}±{mc[1]:.5f})  R_3(b)={R3:.10f}')
print('check b=0 closed forms: m2=1 ->', m2(0), ' E|A|^3/2 = 4/sqrt(pi) =', 4/sqrt(pi), m3odd(0))
