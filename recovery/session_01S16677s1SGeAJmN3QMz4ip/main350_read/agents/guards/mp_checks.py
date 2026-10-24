"""mpmath sup/inf checks of the transcendental bounds in ANALYTIC_GUARDS.md.
Grid + local refinement at 40 digits; margins are reported against the stated bounds."""
from mpmath import mp, mpf, exp, sqrt, findroot, diff
mp.dps = 40
r1 = mpf(1)/32; R = mpf(1)/8
w = R**2/2 - r1**2
def q(t): return exp(-1/t) if t > 0 else mpf(0)
def qp(t): return t**-2*exp(-1/t) if t > 0 else mpf(0)
def psi(t):
    if t <= 0: return mpf(1)
    if t >= 1: return mpf(0)
    return q(1-t)/(q(t)+q(1-t))
def psip(t):
    if t <= 0 or t >= 1: return mpf(0)
    D = q(t)+q(1-t)
    return -(qp(1-t)*q(t) + q(1-t)*qp(t))/D**2
N = 20001
# q' max
m = max((qp(mpf(i)/N), mpf(i)/N) for i in range(1, N))
print('sup q\' grid:', m[0], 'at', m[1], ' 4e^-2=', 4*exp(-2), ' ratio', m[0]/(4*exp(-2)))
# psi' sup
m = max((abs(psip(mpf(i)/N)), mpf(i)/N) for i in range(1, N))
print('sup |psi\'| grid:', m[0], 'at t=', m[1])
# refine: stationary point of psi' near argmax
tstar = findroot(lambda t: diff(psip, t), m[1])
print('refined argmax t*=', tstar, '|psi\'(t*)|=', abs(psip(tstar)), '  bound 4: margin', 4-abs(psip(tstar)))
# Also check the doc's intermediate M/(q(t)+q(1-t)) <= 4 and the denominator bound
md = min(q(mpf(i)/N)+q(1-mpf(i)/N) for i in range(1, N))
print('min denominator grid:', md, '>= e^-2 =', exp(-2), md >= exp(-2))
# ||grad chi|| on transition rho in [r1/2, r1]
def chi_arg(rho): return (rho**2 - r1**2/4)/(3*r1**2/4)
def gradchi(rho): return abs(psip(chi_arg(rho)))*8*rho/(3*r1**2)
m = max((gradchi(r1/2 + (r1/2)*mpf(i)/N), r1/2 + (r1/2)*mpf(i)/N) for i in range(N+1))
print('sup ||grad chi|| grid:', m[0], 'at rho=', m[1], ' (G5) bound 32/(3r1)=', 32/(3*r1), ' 64/r1=', 64/r1, ' ok:', m[0] <= 32/(3*r1))
# B1, D1 actual sup over support (polar: s = rho cos th, z = rho sin th)
from mpmath import cos, sin, pi
def grads_B(rho, th):
    s = rho*cos(th); z = rho*sin(th)
    a = chi_arg(rho); c = psi(a); dps = psip(a)*(8/(3*r1**2))
    gx = dps*s*(s**2/2) + c*s; gz = dps*z*(s**2/2)
    return sqrt(gx**2+gz**2)
def grads_D(rho, th):
    s = rho*cos(th); z = rho*sin(th)
    a = chi_arg(rho); c = psi(a); dps = psip(a)*(8/(3*r1**2))
    gx = dps*s*s + c; gz = dps*z*s
    return sqrt(gx**2+gz**2)
best_B = best_D = (mpf(0), None)
NR, NT = 801, 181
for i in range(NR+1):
    rho = r1*mpf(i)/NR
    for j in range(NT):
        th = pi*mpf(j)/(NT-1)
        vB = grads_B(rho, th); vD = grads_D(rho, th)
        if vB > best_B[0]: best_B = (vB, (rho, th))
        if vD > best_D[0]: best_D = (vD, (rho, th))
print('sup B1 grid:', best_B[0], 'at', best_B[1], ' bound 33 r1=', 33*r1, ' ok:', best_B[0] <= 33*r1, ' tighter 19r1/3=', 19*r1/3)
print('sup D1 grid:', best_D[0], 'at', best_D[1], ' bound 65 ok:', best_D[0] <= 65, ' tighter 35/3=', mpf(35)/3)
print('sup B0 = r1^2/2 =', r1**2/2, ' sup D0 = r1 =', r1)
# c(r), c'(r) over transition, monotone and in [0,1]
def c(rr): return 1 - (1 - rr/4)*psi((rr - r1**2)/w)
def cp(rr):
    a = (rr - r1**2)/w
    return psi(a)/4 - (1 - rr/4)*psip(a)/w
mn = min(cp(r1**2 + w*mpf(i)/N) for i in range(N+1))
print("min c' over transition:", mn, ' >=0:', mn >= 0)
cvals = [c(mpf(i)/N*(R**2/2)) for i in range(N+1)]
print('c range over [0,R^2/2]:', min(cvals), max(cvals), ' in [0,1]:', min(cvals) >= 0 and max(cvals) <= 1)
print('c(r1^2)=r1^2/4:', c(r1**2), r1**2/4, ' c(R^2/2)=1:', c(R**2/2))
print('c(r1^2/4)=', c(r1**2/4), ' r1^2/16=', r1**2/16)
# annular ||grad F0|| with GENERAL c (not just core polynomial) over rho in [r1/2, r1]
def gradF0(rho, th):
    s = rho*cos(th); z = rho*sin(th); rr = rho**2
    Fs = -2*s*(c(rr) + s**2*cp(rr)); Fz = -2*z*(1 + s**2*cp(rr))
    return sqrt(Fs**2+Fz**2)
best = (mpf(10), None)
for i in range(NR+1):
    rho = r1/2 + (r1/2)*mpf(i)/NR
    for j in range(NT):
        th = pi*mpf(j)/(NT-1)
        g = gradF0(rho, th)
        if g < best[0]: best = (g, (rho, th))
print('min ||grad F0|| annulus grid:', best[0], 'at', best[1], ' rho^3/2 there:', best[1][0]**3/2, ' r1^3/16=', r1**3/16, ' ok:', best[0] >= r1**3/16)
# is ||grad F0|| >= rho^3/2 everywhere on the annulus?
viol = 0
for i in range(NR+1):
    rho = r1/2 + (r1/2)*mpf(i)/NR
    for j in range(NT):
        th = pi*mpf(j)/(NT-1)
        if gradF0(rho, th) < rho**3/2: viol += 1
print('violations of ||grad F0|| >= rho^3/2 on annulus grid:', viol)
# barrier: min of b - F0 = s^2 c + z^2 over r1/2 <= rho <= R (whole chart ball)
best = (mpf(10), None)
for i in range(NR+1):
    rho = r1/2 + (R - r1/2)*mpf(i)/NR
    for j in range(NT):
        th = pi*mpf(j)/(NT-1)
        s = rho*cos(th); z = rho*sin(th)
        loss = s**2*c(rho**2) + z**2
        if loss < best[0]: best = (loss, (rho, th))
print('min (b-F0) over rho in [r1/2,R]:', best[0], 'at', best[1], ' delta=', mpf(1)/2**26, ' ok:', best[0] >= mpf(1)/2**26)
# perturbation gradient actual sup at (u0,v0) corners and full F gradient on annulus at corners
u0 = mpf(3)/2**32; v0 = mpf(1)/2**47
def gradpert(rho, th, u, v):
    s = rho*cos(th); z = rho*sin(th)
    a = chi_arg(rho); ch = psi(a); dps = psip(a)*(8/(3*r1**2))
    P = u*s**2/2 + v*s; Ps = u*s + v
    gx = dps*s*P + ch*Ps; gz = dps*z*P
    return sqrt(gx**2+gz**2)
mx = mpf(0)
for i in range(NR+1):
    rho = r1*mpf(i)/NR
    for j in range(NT):
        th = pi*mpf(j)/(NT-1)
        for (uu, vv) in ((u0, v0), (u0, -v0), (-u0, v0), (-u0, -v0)):
            mx = max(mx, gradpert(rho, th, uu, vv))
print('sup ||grad pert|| grid at corners:', mx, ' (G9) bound 101441/2^47=', mpf(101441)/2**47, ' ok:', mx <= mpf(101441)/2**47, ' g*/2=2^-22=', mpf(1)/2**22)
# min ||grad F|| on annulus at corners (should be > 0 by margin)
mn = mpf(10)
for i in range(NR+1):
    rho = r1/2 + (r1/2)*mpf(i)/NR
    for j in range(NT):
        th = pi*mpf(j)/(NT-1)
        for (uu, vv) in ((u0, v0), (u0, -v0), (-u0, v0), (-u0, -v0)):
            s = rho*cos(th); z = rho*sin(th); rr = rho**2
            a = chi_arg(rho); ch = psi(a); dps = psip(a)*(8/(3*r1**2))
            P = uu*s**2/2 + vv*s; Ps = uu*s + vv
            Fs = -2*s*(c(rr) + s**2*cp(rr)) + dps*s*P + ch*Ps
            Fz = -2*z*(1 + s**2*cp(rr)) + dps*z*P
            mn = min(mn, sqrt(Fs**2+Fz**2))
print('min ||grad F|| annulus at corners:', mn, ' > 0:', mn > 0, '  vs r1^3/16 - 2^-22 =', r1**3/16 - mpf(1)/2**22)
