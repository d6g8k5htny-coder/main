"""P15-N2 verifier: does the coordinate convention for B_1, D_1 matter?

Concrete member (ANALYTIC_GUARDS.md Sec.1): d=2, L=1, R=1/8, r1=1/32,
u0=3/2^32, v0=2^-47.  chi(s,z)=psi((s^2+z^2-r1^2/4)/(3 r1^2/4)),
psi(t)=q(1-t)/(q(t)+q(1-t)), q(t)=exp(-1/t) (t>0).
Chart: s=sqrt(2/3) sin(pi y1), z=sqrt(2/9) sin(pi y2).
"""
import math, sys
assert sys.version_info[:2] == (3, 12), sys.version

r1 = 1/32
R = 1/8
u0 = 3/2**32
v0 = 1/2**47
gstar = r1**3/64

def q(t):
    return math.exp(-1/t) if t > 0 else 0.0
def dq(t):
    return math.exp(-1/t)/t**2 if t > 0 else 0.0
def psi(t):
    if t <= 0: return 1.0
    if t >= 1: return 0.0
    return q(1-t)/(q(t)+q(1-t))
def dpsi(t):
    if t <= 0 or t >= 1: return 0.0
    a, b = q(t), q(1-t)
    da, db = dq(t), dq(1-t)
    # d/dt [b/(a+b)] = (-db*(a+b) - b*(da - db))/(a+b)^2 = -(db*a + b*da)/(a+b)^2
    return -(db*a + b*da)/(a+b)**2

W = 3*r1**2/4
def chi(s, z):
    return psi((s*s+z*z - r1**2/4)/W)
def grad_chi(s, z):
    d = dpsi((s*s+z*z - r1**2/4)/W)/W
    return (2*s*d, 2*z*d)

# --- 1. chart-coordinate gradient sup norms of chi*s and chi*s^2/2 ---
N = 1201
best_D1 = best_B1 = 0.0
best_gchi = 0.0
for i in range(N):
    for j in range(N):
        s = -r1 + 2*r1*i/(N-1)
        z = -r1 + 2*r1*j/(N-1)
        c = chi(s, z)
        gs, gz = grad_chi(s, z)
        best_gchi = max(best_gchi, math.hypot(gs, gz))
        # grad(chi s) = (chi + s*gs, s*gz)
        best_D1 = max(best_D1, math.hypot(c + s*gs, s*gz))
        # grad(chi s^2/2) = (chi s + s^2/2 gs, s^2/2 gz)
        best_B1 = max(best_B1, math.hypot(c*s + s*s/2*gs, s*s/2*gz))
print("[chart (s,z) coordinates, concrete member]")
print(f"  sup|grad chi|        ~ {best_gchi:.6g}   (G5) bound 32/(3 r1) = {32/(3*r1):.6g}, frozen 64/r1 = {64/r1:.6g}")
print(f"  sup|grad(chi s)|     ~ {best_D1:.6g}   (G6) D1 = 65")
print(f"  sup|grad(chi s^2/2)| ~ {best_B1:.6g}   (G6) B1 = 33 r1 = {33*r1:.6g}")
assert best_gchi <= 32/(3*r1)*1.0001 and best_D1 <= 65 and best_B1 <= 33*r1
print(f"  guard: B1 u0 + D1 v0 = {33*r1*u0 + 65*v0:.6e} < g*/2 = {gstar/2:.6e}  ->", 33*r1*u0 + 65*v0 < gstar/2)

# --- 2. chart Jacobian dz/dy on the annulus, concrete member d=2, L=1 ---
# z_i = sqrt(2 w_i) sin(omega y_i/2), w_i = 3^-i, omega = 2 pi / L
def sigma_min_on_annulus(d, L):
    omega = 2*math.pi/L
    w = [3.0**-(i+1) for i in range(d)]
    # on the annulus |z|<=r1, |z_i|<=r1, so |sin(omega y_i/2)| <= r1/sqrt(2 w_i)
    sig = []
    for wi in w:
        smax = min(1.0, r1/math.sqrt(2*wi))
        cmin = math.sqrt(1 - smax**2)
        sig.append(math.sqrt(2*wi)*omega/2*cmin)
    return min(sig), max(math.sqrt(2*wi)*omega/2 for wi in w)

print("\n[physical reading of B_1,D_1: exclusion needs sigma_min(dz/dy) >= 1/2 on the annulus]")
for d, L in [(2, 1), (3, 1), (4, 1), (5, 1), (2, 2), (2, 3), (2, 4)]:
    smin, smax = sigma_min_on_annulus(d, L)
    ok = smin >= 0.5
    print(f"  d={d}, L={L}: sigma_min ~ {smin:.4f}, sigma_max ~ {smax:.4f}  -> mixed-reading guard {'suffices' if ok else 'FAILS'}")
