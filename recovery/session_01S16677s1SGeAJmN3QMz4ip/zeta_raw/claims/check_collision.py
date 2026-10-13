"""Track real zeros of H_t backward in t from t=0, locate the first collision
(real double zero) and test the local normal form H ~ A (x^2 - 2 tau),
r^2 ~ 8 tau, bar length |A| r^3 / 6, and complex pair for tau < 0."""
import time
from mpmath import mp
import dbn_common
from dbn_common import HFlow

mp.dps = 40
out = []
def rep(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

F = HFlow(U=1.3, pieces=26, degree=4)
rep(f"[nodes {len(F.u)}]")

gam = [mp.mpf(g) for g in ["14.134725141734693790", "21.022039638771554993", "25.010857580145688763",
                          "30.424876125859513210", "32.935061587739189691", "37.586178158825671257",
                          "40.918719012147495187", "43.327073280914999519", "48.005150881167159728",
                          "49.773832477672302182", "52.970321477714460644", "56.446247697063394804"]]
zs = [2 * g for g in gam]
zs = [mp.findroot(lambda z: F.H(0, z), z0) for z0 in zs]
rep("zeros of H_0:", [mp.nstr(z, 10) for z in zs])

def newton(t, z, it=30):
    for _ in range(it):
        h = F.H(t, z); d = F.Hz(t, z)
        dz = h / d
        z -= dz
        if abs(dz) < mp.mpf(10) ** (-25):
            return z, True
    return z, False

t = mp.mpf(0); dt = mp.mpf("-0.02")
hist = []
collided = None
while t > -6 and collided is None:
    t += dt
    new = []
    for z in zs:
        zn, ok = newton(t, z)
        new.append(zn)
    gaps = [new[i + 1] - new[i] for i in range(len(new) - 1)]
    hist.append((t, new, gaps))
    imin = min(range(len(gaps)), key=lambda i: gaps[i])
    if gaps[imin] < mp.mpf("0.6"):
        collided = imin
    zs = new
rep(f"stopped tracking at t={mp.nstr(t,6)}; closest pair index {collided}: "
    f"z={mp.nstr(zs[collided],10)}, {mp.nstr(zs[collided+1],10)}, gap {mp.nstr(zs[collided+1]-zs[collided],6)}")

# report gap^2 vs t along the approach (two-body prediction: d(gap^2)/dt ~ 8)
for (tt, zz, gg) in hist[-8:]:
    rep(f"  t={mp.nstr(tt,6)} gap={mp.nstr(gg[collided],8)} gap^2={mp.nstr(gg[collided]**2,8)}")
g2a = hist[-2][2][collided] ** 2; g2b = hist[-1][2][collided] ** 2
rep(f"  slope d(gap^2)/dt over last step = {mp.nstr((g2b-g2a)/dt,8)}  (normal form predicts 8 at collision)")

# locate the double zero: H = H_z = 0 in (z, t)
zg = (zs[collided] + zs[collided + 1]) / 2
tg = t + (hist[-1][2][collided] ** 2) / 8 * (-1)
sol = mp.findroot([lambda z, tt: F.H(tt, z) / F.Hzz(t, zg), lambda z, tt: F.Hz(tt, z) / F.Hzz(t, zg)], (zg, tg))
z0, t0 = sol[0], sol[1]
A = F.Hzz(t0, z0) / 2
Hzzz = F.Hzzz(t0, z0)
rep(f"double zero at z0={mp.nstr(z0,20)}, t0={mp.nstr(t0,20)}; H={mp.nstr(F.H(t0,z0),3)}, Hz={mp.nstr(F.Hz(t0,z0),3)}, A=Hzz/2={mp.nstr(A,12)}")
dtt = mp.mpf("1e-15")
Ht = (F.H(t0 + dtt, z0) - F.H(t0 - dtt, z0)) / (2 * dtt)
rep(f"dH/dt at (z0,t0) = {mp.nstr(Ht,12)}; -2A = {mp.nstr(-2*A,12)}  => H ~ A x^2 + (-2A) tau = A (x^2 - 2 tau)")

for tau in ["1e-2", "1e-3", "1e-4", "1e-6"]:
    tau = mp.mpf(tau)
    tt = t0 + tau
    s = mp.sqrt(2 * tau)
    zl, _ = newton(tt, z0 - s)
    zr, _ = newton(tt, z0 + s)
    r = zr - zl
    area = mp.quad(lambda x: F.H(tt, x), [zl, zr])
    rep(f"tau=+{mp.nstr(tau,3)}: real zeros {mp.nstr(zl,15)}, {mp.nstr(zr,15)}; r={mp.nstr(r,12)}; "
        f"r^2/(8 tau)={mp.nstr(r*r/(8*tau),12)}; |area|={mp.nstr(abs(area),12)}; |A| r^3/6={mp.nstr(abs(A)*r**3/6,12)}; ratio={mp.nstr(abs(area)/(abs(A)*r**3/6),12)}")
for tau in ["-1e-2", "-1e-4"]:
    tau = mp.mpf(tau)
    tt = t0 + tau
    zc = mp.findroot(lambda z: F.H(tt, z), z0 + 1j * mp.sqrt(-2 * tau))
    rep(f"tau={mp.nstr(tau,3)}: complex zero {mp.nstr(zc,15)}; Im/sqrt(2|tau|)={mp.nstr(mp.im(zc)/mp.sqrt(-2*tau),12)}; "
        f"real zeros near z0? H(z0)={mp.nstr(F.H(tt,z0),5)} (sign of A={'+' if A>0 else '-'})")

open("check_collision.out", "w").write("\n".join(out) + "\n")
mp_state = dict(z0=str(z0), t0=str(t0), A=str(A))
open("collision_point.txt", "w").write(repr(mp_state) + "\n")
