"""Checks: H_0(z) = Xi(z/2)/8; Phi even and equal to a theta-derivative expression;
backward heat equation sign; zero-dynamics ODE (Csordas--Smith--Varga form) at t=0.

The zero ODE dz_j/dt = 2 sum_{k != j} 1/(z_j - z_k) does not converge absolutely; it is
summed symmetrically here, pairing each zero with its mirror -z_k:
    dz_j/dt = 2 [ 1/(2 z_j) + sum_{k != j} 2 z_j / (z_j^2 - z_k^2) ]  (positive zeros only),
truncated at 300 zeros plus a tail estimate (truncation-limited at about 1.5e-4).
The exact velocity is H_zz/H_z (see zero_velocity.py).
Output: results/dbn/check_dbn_basic.out
"""
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "dbn"

from mpmath import mp
from dbn_common import Phi, Xi, HFlow

mp.dps = 60
out = []
def rep(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

# 1. Phi evenness and theta form. psi(x) = sum_{n>=1} exp(-pi n^2 x) = (Theta(x)-1)/2
def psi_d(x, k):
    return mp.fsum((-mp.pi * n * n) ** k * mp.exp(-mp.pi * n * n * x) for n in range(1, 200))
for u in ["0.05", "0.2", "0.4"]:
    u = mp.mpf(u)
    x = mp.exp(4 * u)
    theta_form = mp.exp(u) * (2 * x * x * psi_d(x, 2) + 3 * x * psi_d(x, 1))
    rep(f"Phi({u}) = {mp.nstr(Phi(u), 20)};  Phi(-u) = {mp.nstr(Phi(-u, nmax=400), 20)};  e^u(2x^2 psi''+3x psi')|x=e^4u = {mp.nstr(theta_form, 20)}")

# 2. H_0(z) vs Xi(z/2)/8
t0 = time.time()
F = HFlow()
rep(f"[quadrature nodes: {len(F.u)}, setup {time.time()-t0:.1f}s]")
for z in [0, 10, 28.2694, 40, 75, 150]:
    z = mp.mpf(z)
    h = F.H(0, z)
    x = Xi(z / 2) / 8
    rep(f"z={mp.nstr(z,8)}: H_0(z)={mp.nstr(h,15)}  Xi(z/2)/8={mp.nstr(mp.re(x),15)}  rel.diff={mp.nstr(abs(h-x)/abs(x),3)}")

# 3. PDE sign by finite differences on the computed function (no use of the derivation)
for (t, z) in [(0, 30), (0.3, 55), (-1, 20)]:
    t = mp.mpf(t); z = mp.mpf(z)
    dt = mp.mpf("1e-12"); dz = mp.mpf("1e-10")
    Ht = (F.H(t + dt, z) - F.H(t - dt, z)) / (2 * dt)
    Hzz = (F.H(t, z + dz) - 2 * F.H(t, z) + F.H(t, z - dz)) / dz**2
    rep(f"t={t}, z={z}: dH/dt={mp.nstr(Ht,12)}  d2H/dz2={mp.nstr(Hzz,12)}  sum={mp.nstr(Ht+Hzz,3)}  (backward heat eq. predicts dH/dt = -d2H/dz2)")

# 4. Zero dynamics at t=0 for the first zero z_1 = 2*gamma_1
K = 300
t1 = time.time()
gam = [mp.im(mp.zetazero(k)) for k in range(1, K + 1)]
rep(f"[{K} zeta zeros in {time.time()-t1:.1f}s; gamma_K={mp.nstr(gam[-1],10)}]")
zs = [2 * g for g in gam]
for j in [0, 1, 4]:
    zj = zs[j]
    # zero of H_t near zj for t = +-h
    hstep = mp.mpf("1e-6")
    zp = mp.findroot(lambda z: F.H(hstep, z), zj)
    zm = mp.findroot(lambda z: F.H(-hstep, z), zj)
    z0 = mp.findroot(lambda z: F.H(0, z), zj)
    vel = (zp - zm) / (2 * hstep)
    # predicted: 2 * [ 1/(2 z_j) + sum_{k != j} 2 z_j / (z_j^2 - z_k^2) ]  (mirror zeros -z_k included)
    S = 1 / (2 * zj) + mp.fsum(2 * zj / (zj**2 - zk**2) for k, zk in enumerate(zs) if k != j)
    # tail k > K: sum 2 z_j/(z_j^2 - z_k^2) ~ -2 z_j sum 1/z_k^2, sum_{gamma>G} 1/gamma^2 ~ (log(G/2pi)+1)/(2 pi G)
    G = gam[-1]
    tail = -2 * zj * (mp.log(G / (2 * mp.pi)) + 1) / (2 * mp.pi * G) / 4
    pred = 2 * (S + tail)
    alt = mp.mpf(-1) * pred
    rep(f"zero #{j+1}: z_j(0)={mp.nstr(z0,15)} (2*gamma={mp.nstr(zj,15)}); dz_j/dt (finite diff)={mp.nstr(vel,10)}; "
        f"ODE 2*sum 1/(z_j-z_k) (incl. mirrors, tail est {mp.nstr(2*tail,3)})={mp.nstr(pred,10)}; opposite sign would be {mp.nstr(alt,6)}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "check_dbn_basic.out").write_text("\n".join(out) + "\n")
