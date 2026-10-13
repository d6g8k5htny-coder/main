from mpmath import mp, mpf, exp, sqrt, cos, sin, pi, polyroots
mp.dps = 40
r1 = mpf(1)/32; R = mpf(1)/8; w = R**2/2 - r1**2; b = mpf(4)/9
u0 = mpf(3)/2**32; v0 = mpf(1)/2**47; a_star = r1/4; delta = mpf(1)/2**26
def q(t): return exp(-1/t) if t > 0 else mpf(0)
def qp(t): return t**-2*exp(-1/t) if t > 0 else mpf(0)
def psi(t):
    if t <= 0: return mpf(1)
    if t >= 1: return mpf(0)
    return q(1-t)/(q(t)+q(1-t))
def psip(t):
    if t <= 0 or t >= 1: return mpf(0)
    D = q(t)+q(1-t); return -(qp(1-t)*q(t) + q(1-t)*qp(t))/D**2
def c(rr): return 1 - (1 - rr/4)*psi((rr - r1**2)/w)
def cp(rr):
    a = (rr - r1**2)/w; return psi(a)/4 - (1 - rr/4)*psip(a)/w
def chi_arg(rho): return (rho**2 - r1**2/4)/(3*r1**2/4)
def gradF(s, z, u, v):
    rho = sqrt(s**2+z**2); rr = rho**2
    a = chi_arg(rho); ch = psi(a); dps = psip(a)*(8/(3*r1**2))
    P = u*s**2/2 + v*s; Ps = u*s + v
    Fs = -2*s*(c(rr) + s**2*cp(rr)) + dps*s*P + ch*Ps
    Fz = -2*z*(1 + s**2*cp(rr)) + dps*z*P
    return sqrt(Fs**2+Fz**2)
def F(s, z, u, v):
    rho = sqrt(s**2+z**2); rr = rho**2
    return b - s**2*c(rr) - z**2 + psi(chi_arg(rho))*(u*s**2/2 + v*s)
corners = ((u0, v0), (u0, -v0), (-u0, v0), (-u0, -v0), (u0, mpf(0)), (mpf(0), v0), (mpf(0), mpf(0)))
# 1. min ||grad F|| over rho in [r1/2, R] (outside core) at corners, log-spaced radial grid + angles
NR, NT = 600, 121
mn = (mpf(10), None)
import math
for i in range(NR+1):
    rho = r1/2 * (R/(r1/2))**(mpf(i)/NR)
    for j in range(NT):
        th = 2*pi*mpf(j)/NT
        s = rho*cos(th); z = rho*sin(th)
        for (u, v) in corners:
            g = gradF(s, z, u, v)
            if g < mn[0]: mn = (g, (rho, th, u, v))
print('min ||grad F|| outside core (rho in [r1/2,R]):', mn[0], 'at rho,th,u,v=', mn[1])
# 2. max F over rho in [r1/2, R] at corners vs b - 3 delta/4
mx = (mpf(-10), None)
for i in range(NR+1):
    rho = r1/2 * (R/(r1/2))**(mpf(i)/NR)
    for j in range(NT):
        th = 2*pi*mpf(j)/NT
        s = rho*cos(th); z = rho*sin(th)
        for (u, v) in corners:
            f = F(s, z, u, v)
            if f > mx[0]: mx = (f, (rho, th, u, v))
print('max F outside core:', mx[0], ' b-3delta/4=', b - 3*delta/4, ' barrier ok:', mx[0] <= b - 3*delta/4, ' at', mx[1])
# 3. core critical points at corners: roots of -s^3+u s+v, check |s|<a*, heights in (b-delta/4, b+delta/4)
for (u, v) in corners:
    roots = polyroots([-1, 0, u, v])
    real = [r.real for r in roots if abs(r.imag) < mpf(10)**-30]
    hs = [b + (-s**4/4 + u*s**2/2 + v*s) for s in real]
    print(f'u={u} v={v}: real roots', [str(x) for x in real], ' all |s|<a*:', all(abs(x) < a_star for x in real), ' heights-b in (-delta/4,delta/4):', all(abs(h-b) < delta/4 for h in hs), ' rho<r1/2:', all(abs(x) < r1/2 for x in real))
# 4. endpoint behaviour of psi' at high precision (doc G3 bound 4, denominator >= e^-2)
mp.dps = 60
for t in [mpf(10)**-k for k in (1,2,3,6,12,30)] + [1-mpf(10)**-k for k in (1,2,3,6,12,30)]:
    D = q(t)+q(1-t)
    print(f't={t}: |psi\'|={abs(psip(t))}  M/D={4*exp(-2)/D}  D>=e^-2: {D >= exp(-2)}  |psi\'|<=4: {abs(psip(t))<=4}')
# 5. CF.md line 228 formula vs (G7)
import sympy as sp
r_, p_ = sp.symbols('r p')
print('CF c-formula == (G7):', sp.expand((r_/4 + (1-r_/4)*(1-p_)) - (1 - (1-r_/4)*p_)) == 0)
