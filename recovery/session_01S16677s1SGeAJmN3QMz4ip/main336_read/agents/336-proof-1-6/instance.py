# Concrete d=2 instance of the PROOF.md construction (sections 2-6), with explicit c and chi.
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize

d = 2
L = 2*np.pi; om = 2*np.pi/L
w = np.array([3.0**-1, 3.0**-2]); b = w.sum(); Delta = 2*w[-1]
R = 0.3; assert R**2 < min(w[-1], 1.0)
r1 = R/4; delta = r1**4/64
eta = 0.2; assert 0 < eta < 2/(3*np.sqrt(3))

# ---- smooth step S(x): 0 for x<=0, 1 for x>=1 (C-infinity) ----
def _g(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = x > 0
    out[m] = np.exp(-1.0/x[m])
    return out
def S(x):
    x = np.asarray(x, dtype=float)
    return _g(x)/(_g(x)+_g(1-x))

# ---- c'(v): 1/4 on [0,r1^2], tapers to 0 by v=2 r1^2; plus bump on (3 r1^2, R^2/2) ----
va, vb = r1**2, 2*r1**2            # taper interval
vc, vd = 3*r1**2, R**2/2           # bump support (strictly inside (r1^2, R^2/2))
assert vb < vc < vd
def d0(v):
    v = np.asarray(v, dtype=float)
    return 0.25*(1 - S((v - va)/(vb - va)))
def bump_raw(v):
    v = np.asarray(v, dtype=float)
    return S((v - vc)/(vd - vc)) * S((vd - v)/(vd - vc))   # smooth, >0 on (vc,vd), 0 outside
I0 = quad(lambda v: float(d0(v)), 0, vb, limit=200)[0]
Ib = quad(lambda v: float(bump_raw(v)), vc, vd, limit=200)[0]
K = (1 - I0)/Ib
assert I0 < 1 and K > 0
def cprime(v):
    return d0(v) + K*bump_raw(v)
# tabulate c on a fine grid by high-accuracy quadrature
_vgrid = np.linspace(0, R**2, 20001)
_cvals = np.zeros_like(_vgrid)
acc = 0.0
for i in range(1, len(_vgrid)):
    acc += quad(lambda v: float(cprime(v)), _vgrid[i-1], _vgrid[i])[0]
    _cvals[i] = acc
def c(v):
    v = np.asarray(v, dtype=float)
    out = np.interp(v, _vgrid, _cvals)
    out = np.where(v <= va, v/4, out)          # exact on the linear regime
    out = np.where(v >= vd, 1.0, out)
    return out

# ---- chi(z) = psi(|z|^2): 1 for |z|<=r1/2, 0 for |z|>=r1 ----
pa, pb = (r1/2)**2, r1**2
def psi(v):
    v = np.asarray(v, dtype=float)
    return 1 - S((v - pa)/(pb - pa))
def psiprime(v, hh=1e-9):
    return (psi(v+hh) - psi(v-hh))/(2*hh)

# ---- fields in the z-chart (s = z[0], t = z[1:]) ----
def F0(s, t):
    v = s**2 + t**2
    return b - s**2*c(v) - t**2
def gradF0(s, t):
    v = s**2 + t**2
    cv = c(v); cpv = cprime(v)
    return np.array([-2*s*(cv + s**2*cpv), -2*t*(1 + s**2*cpv)])
def pert(s, t, lam):
    v = s**2 + t**2
    return psi(v)*(lam*s**2/2 + eta*lam**1.5*s)
def gradpert(s, t, lam):
    v = s**2 + t**2
    ps = psi(v); dps = psiprime(v)
    P = lam*s**2/2 + eta*lam**1.5*s
    dP = lam*s + eta*lam**1.5
    return np.array([dps*2*s*P + ps*dP, dps*2*t*P])
def Flam(s, t, lam): return F0(s, t) + pert(s, t, lam)
def gradFlam(s, t, lam): return gradF0(s, t) + gradpert(s, t, lam)

# ---- Section 4 roots ----
f = lambda z: -z**3 + z + eta
zm = brentq(f, -1, -1/np.sqrt(3)); z0 = brentq(f, -1/np.sqrt(3), 0); zp = brentq(f, 1, 2)
Phi = lambda z: -z**4/4 + z**2/2 + eta*z
Zstar = max(abs(zm), abs(z0), abs(zp))
kappa = Phi(zm) - Phi(z0)
CP = 1 + max(abs(Phi(zm)), abs(Phi(z0)), abs(Phi(zp)))

# ---- Section 5 constants ----
# g_* = min over annulus A of |grad F0|; on A (|z|<=r1) F0 is the exact quartic form
ph = np.linspace(0, 2*np.pi, 4001); rr = np.linspace(r1/2, r1, 801)
RR, PH = np.meshgrid(rr, ph, indexing='ij')
Sg, Tg = RR*np.cos(PH), RR*np.sin(PH)
G = gradF0(Sg, Tg); gstar_grid = np.sqrt((G**2).sum(axis=0)).min()
# analytic: |grad|^2 = s^2(s^2+t^2/2)^2 + 4t^2(1+s^2/4)^2 ; minimize on annulus numerically with polish
def gn(x):
    r, p = x; s, t = r*np.cos(p), r*np.sin(p)
    return np.sqrt((gradF0(s, t)**2).sum())
res = min((minimize(gn, [r, p], bounds=[(r1/2, r1), (0, 2*np.pi)]) for r in [r1/2, 0.75*r1, r1] for p in np.linspace(0, 2*np.pi, 9)), key=lambda r: r.fun)
gstar = min(gstar_grid, res.fun)
# C_0, C_1 over support |z|<=r1
rr2 = np.linspace(0, r1, 1201); RR2, PH2 = np.meshgrid(rr2, ph, indexing='ij')
S2, T2 = RR2*np.cos(PH2), RR2*np.sin(PH2); V2 = S2**2 + T2**2
C0 = np.abs(psi(V2)*S2**2/2).max() + eta*np.abs(psi(V2)*S2).max()
dps2 = psiprime(V2); ps2 = psi(V2)
g_a = np.stack([dps2*2*S2*(S2**2/2) + ps2*S2, dps2*2*T2*(S2**2/2)])
g_b = np.stack([dps2*2*S2*S2 + ps2, dps2*2*T2*S2])
C1 = np.sqrt((g_a**2).sum(0)).max() + eta*np.sqrt((g_b**2).sum(0)).max()
bounds16 = dict(one=1.0, grad=gstar/(2*C1), core=(r1/(4*Zstar))**2, sup=delta/(4*C0), val=np.sqrt(min(delta/4, Delta/4)/CP))
lam0 = 0.9*min(bounds16.values())

if __name__ == "__main__":
    print(f"d={d} L={L:.6g} w={w} b={b:.9g} Delta={Delta:.6g} R={R} r1={r1} delta={delta:.6g} eta={eta}")
    print(f"c: I0={I0:.6g} K={K:.6g}  c(r1^2)={float(c(r1**2)):.6g} (=r1^2/4={r1**2/4:.6g})  c(R^2/2)={float(c(R**2/2)):.6g}  c(r1^2/4)={float(c(r1**2/4)):.6g} (=r1^2/16={r1**2/16:.6g})")
    # monotone and in [0,1]
    cv = c(_vgrid); print("c nondecreasing:", bool(np.all(np.diff(cv) >= -1e-15)), " c in [0,1]:", bool(cv.min() >= 0 and cv.max() <= 1+1e-12), " c>0 for v>0:", bool(cv[1:].min() > 0))
    print(f"roots z-={zm:.9g} z0={z0:.9g} z+={zp:.9g}  Phi: {Phi(zm):.9g} {Phi(z0):.9g} {Phi(zp):.9g}  kappa={kappa:.9g} Zstar={Zstar:.6g} CP={CP:.6g}")
    print(f"(14) Phi(z+)>Phi(z-)>Phi(z0): {Phi(zp)>Phi(zm)>Phi(z0)}   Phi(z+)>Phi(1)={Phi(1):.6g}: {Phi(zp)>Phi(1)}")
    print(f"g*={gstar:.6g}  C0={C0:.6g}  C1={C1:.6g}")
    print("(16) individual lambda_0 caps:", {k: f"{v:.3e}" for k, v in bounds16.items()}, f" -> lambda_0={lam0:.3e}")
