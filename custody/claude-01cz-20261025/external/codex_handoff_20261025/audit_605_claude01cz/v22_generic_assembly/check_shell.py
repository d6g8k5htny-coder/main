# Diagnostic only (not proof): numerically evaluate the per-shell quantity of V22-BAND-GENERIC-ASSEMBLY-01
#   Shell(r,s) = r^2 * s * I(r,s),  I = int_0^{eps s} e * min(1, X exp(-c (c1 k s (r+e)|e-e*| - L)_+^2 / s^8)) de
# with X = k r^3/s^4, L = k r^3 + s^4, e* = r, c = c1 = k = 1, eps = 1/2, and report Shell / r^5 over dyadic s in [4r, 0.1].
import math
def I(r, s, k=1.0, c=1.0, c1=1.0, eps=0.5, es=None):
    es = r if es is None else es
    X = k*r**3/s**4; L = k*r**3 + s**4; s8 = s**8
    E = eps*s
    # adaptive grid: dense near e*
    pts = sorted(set([0.0, E] + [min(E, max(0.0, es + d)) for d in [x*1e-3*s for x in range(-1000, 1001)]]))
    tot = 0.0
    def f(e):
        g = c1*k*s*(r+e)*abs(e-es)
        ex = -c*max(g-L, 0.0)**2/s8
        return e*min(1.0, X*math.exp(ex)) if ex > -700 else 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        if b <= a: continue
        n = 40; h = (b-a)/n
        tot += sum(f(a + (j+0.5)*h) for j in range(n))*h
    return tot
worst = 0.0
for r in (1e-3, 1e-4, 1e-5, 1e-6):
    row = []; s = 4*r; total = 0.0
    while s <= 0.1:
        sh = r**2*s*I(r, s)/r**5; worst = max(worst, sh); total += sh; row.append(f"{sh:.2f}"); s *= 2
    print(f"r={r:g}: shells/r^5 (s=4r,8r,..,<=0.1): {' '.join(row)} | sum/r^5={total:.1f}  log2(0.1/4r)={math.log2(0.1/(4*r)):.1f}")
print("max per-shell / r^5 =", round(worst, 3))
