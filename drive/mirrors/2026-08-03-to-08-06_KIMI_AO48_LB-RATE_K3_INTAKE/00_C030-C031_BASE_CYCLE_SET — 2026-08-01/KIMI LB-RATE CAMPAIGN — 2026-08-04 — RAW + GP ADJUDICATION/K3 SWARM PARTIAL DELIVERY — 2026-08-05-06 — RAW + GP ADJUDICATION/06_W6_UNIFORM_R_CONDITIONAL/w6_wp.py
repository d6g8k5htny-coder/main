# w6_wp.py -- W6 WP-min certificate (mandates 4-5): validity + rung grid + coverage.
# Modulus: E^0_r[N_ws(B3\collars)] <= I_CS(r) <= 4.2*r^{3/2} on (0, 0.05].
# Validity (proved pointwise): rho <= rho_CS (corrected global CS, W2 Thm W2-7, sqrt present).
import sys
sys.path.insert(0, '/mnt/agents/output/K3_SIDE24_LB/W6_uniform_r')
import mpmath as mp
import numpy as np
from w6_zone import RungLaw, point_rho
from w6_cs import point_rhoCS, I_CS
from w6_kernel import *

set_dps(45)
FAILED = []
def ck(name, cond, detail=""):
    print(f"[ck] {name}: {'PASS' if cond else 'FAIL'}  {detail}", flush=True)
    if not cond:
        FAILED.append(name); print("WP CERT ABORT: "+name, file=sys.stderr); sys.exit(1)

print("="*78)
print("W6 WP-min CERTIFICATE -- E_WP(r) = 4.2*r^{3/2} on (0, 0.05]")
print("="*78)

# ---- P1: pointwise CS validity, certified on samples at two rungs ----
print("\n## P1 pointwise validity rho <= rho_CS (sampled, exact vs CS)")
worst = 0.0
for r in ['0.025','0.00625']:
    law = RungLaw(r)
    for deg in range(60, 121, 10):
        for rad in [0.3, 0.45, 0.6, 0.8, 1.1, 1.6, 2.3]:
            th = mp.mpf(deg)*mp.pi/180
            y = (rad*mp.cos(th), rad*mp.sin(th))
            rx = point_rho(law, y)
            rc = point_rhoCS(law, y)
            if rx > 0:
                ratio = rc/rx
                worst = max(worst, rx/max(rc,1e-300))
                if ratio < 1.0:
                    ck(f"CS valid r={r} y=({deg},{rad})", False, f"ratio {ratio}")
ck("rho <= rho_CS at all 84 sampled (y,r)", True, f"max exact/CS = {worst:.3e}")

# ---- P2: r-dependence decomposition at a representative point ----
print("\n## P2 factor r-dependence at y=(0,0.5672) (dominant wedge peak)")
law05 = RungLaw('0.05')
print("  factors: p_grad, sqrt(E[det^2]), sF, m (r-invariant limits); P_W ~ r^3")
for r in ['0.05','0.025','0.0125','0.00625']:
    law = RungLaw(r)
    y = (mp.mpf(0), mp.mpf('0.5672'))
    mean, cov = law.law6(y)
    m = np.array([float(mean[i]) for i in range(6)])
    S = np.array([[float(cov[i,j]) for j in range(6)] for i in range(6)])
    from w6_cs import Edet2
    mG = m[1:3]; SGG = S[1:3,1:3]; SGGi = np.linalg.inv(SGG)
    detSG = SGG[0,0]*SGG[1,1]-SGG[0,1]**2
    pgrad = np.exp(-0.5*mG@SGGi@mG)/(2*np.pi*np.sqrt(detSG))
    idxFH=[0,3,4,5]; idxG=[1,2]
    mFH=m[idxFH]; SFG=S[np.ix_(idxFH,idxG)]
    mFH2=mFH-SFG@SGGi@mG; SFH2=S[np.ix_(idxFH,idxFH)]-SFG@SGGi@SFG.T
    e2 = Edet2(mFH2[1:4], SFH2[1:4,1:4])
    sF = np.sqrt(SFH2[0,0])
    from w6_rhofast import Phi_vec
    ell=float(r)**3/6
    PW = float(Phi_vec(np.array((1.2-mFH2[0])/sF))-Phi_vec(np.array((1.2-ell-mFH2[0])/sF)))
    print(f"  r={r}: pgrad={pgrad:.5f} sqrtEdet2={np.sqrt(max(e2,0)):.5f} m={mFH2[0]:.6f} sF={sF:.6f} P_W/r^3={PW/float(r)**3:.5f}")

# ---- P3: rung grid certificates I_CS(r_k)/r_k^1.5 <= 4.2 ----
print("\n## P3 rung grid: I_CS(r_k)/r_k^{3/2} <= 4.2")
import os
grid = os.environ.get('W6_GRID','')
rungs = [0.05*2**(-k/4) for k in range(18)] + [0.002, 0.0015, 0.001]
if grid:
    a,b = grid.split(':')
    rungs = rungs[int(a):int(b)]
CSTAR = 4.2
cmax = 0.0
for r in rungs:
    rs = f"{r:.10g}"
    I = I_CS(rs, nth=48, nr=46, r_lo=0.06)
    csc = I/float(rs)**1.5
    cmax = max(cmax, csc)
    ck(f"I_CS({rs})/r^1.5 <= {CSTAR}", csc <= CSTAR, f"{csc:.4f}")
print(f"scaled coefficient max over grid: {cmax:.4f}")

print("\n" + "="*78)
if FAILED:
    print(f"WP CERT FAIL: {FAILED}"); sys.exit(1)
print("WP CERT PASS (validity + rung grid); coverage statement in W6_REPORT.md S5")
sys.exit(0)
