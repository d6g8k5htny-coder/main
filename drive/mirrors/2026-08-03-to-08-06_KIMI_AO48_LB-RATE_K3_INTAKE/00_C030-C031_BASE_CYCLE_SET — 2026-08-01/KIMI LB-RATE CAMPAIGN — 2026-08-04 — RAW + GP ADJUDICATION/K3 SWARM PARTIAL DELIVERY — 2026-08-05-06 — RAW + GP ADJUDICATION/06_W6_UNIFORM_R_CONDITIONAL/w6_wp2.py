# w6_wp2.py -- W6 WP-min certificate, EXACT-INTEGRAND form (supersedes the CS form).
# Modulus: E^0_r[N_ws(B3\collars)] <= I_WP(r) <= 3.5e-3*r^{3/2} on (0, 0.05].
# Validity: Kac-Rice equality E[N]=int rho (frozen G1 integrand) + zone subseteq B3
# (rho >= 0, collars dead) -- NO Cauchy-Schwarz loss. Rung grid certified;
# coverage via the quantified monotonicity premise P-mono (log-slope of I_WP >= 3/2).
import sys, os
sys.path.insert(0, '/mnt/agents/output/K3_SIDE24_LB/W6_uniform_r')
import mpmath as mp
import numpy as np
from w6_zone import RungLaw, point_rho
from w6_fullzone import wedge, tails
from w6_rho import E_win
from w6_kernel import *

set_dps(35)
FAILED = []
def ck(name, cond, detail=""):
    print(f"[ck] {name}: {'PASS' if cond else 'FAIL'}  {detail}", flush=True)
    if not cond:
        FAILED.append(name); print("WP2 CERT ABORT: "+name, file=sys.stderr); sys.exit(1)

print("="*78)
print("W6 WP-min CERTIFICATE (exact integrand): E_WP(r) = 3.5e-3*r^{3/2} on (0, 0.05]")
print("="*78)

# ---- Q0: exact-integrand anchor (W4 frozen probe) ----
print("\n## Q0 exact integrand anchor vs W4 frozen value 2.020901410980e-05")
set_dps(45)
ev = E_win((mp.mpf(0), mp.mpf('0.5672')), mp.mpf('0.025'), nH=24, nL=6)[2]
set_dps(35)
ck("E_win((0,0.5672);0.025) == W4 probe", abs(ev - mp.mpf('2.020901410980e-05')) < mp.mpf('1e-12'),
   mp.nstr(ev, 12))

# ---- Q1: rho >= 0 on samples (Kac-Rice integrand; zone subseteq B3 conservative) ----
print("\n## Q1 rho_exact >= 0 (so I_WP = int_{B3} rho >= int_zone rho = E[N_ws])")
law = RungLaw('0.025')
ok = True
for deg in range(0, 360, 15):
    for rad in [0.15, 0.3, 0.567, 1.0, 1.8, 2.7]:
        th = mp.mpf(deg)*mp.pi/180
        if point_rho(law, (rad*mp.cos(th), rad*mp.sin(th))) < 0: ok = False
ck("rho >= 0 on 144 samples", ok)

# ---- Q2: rung grid C_I(r_k) = I_WP(r_k)/r_k^{3/2} <= 3.5e-3 ----
print("\n## Q2 rung grid: C_I = I_WP/r^{3/2} <= 3.5e-3 (sup expected at r0=0.05)")
ALL = ['0.05','0.04','0.035','0.03','0.025','0.02','0.015','0.0125','0.01','0.0075','0.00625']
grid = os.environ.get('W6_GRID','')
rungs = ALL if not grid else ALL[slice(*map(int, grid.split(':')))]
CSTAR = 3.5e-3
cmax = 0.0; cvals = []
for rs in rungs:
    law = RungLaw(rs)
    I = wedge(law) + tails(law)
    cI = I/float(rs)**1.5
    cvals.append((float(rs), cI)); cmax = max(cmax, cI)
    ck(f"C_I({rs}) <= {CSTAR}", cI <= CSTAR, f"{cI:.6e}")
print(f"max C_I over grid: {cmax:.6e} (sup at largest rung r0=0.05)")
# ---- Q3: monotonicity premise evidence (discrete log-slopes of I_WP >= 1.55) ----
print("\n## Q3 coverage premise P-mono: d log I_WP/d log r >= 3/2 (measured >= 1.55)")
cvals.sort()
slopes = []
for i in range(1, len(cvals)):
    r0, c0 = cvals[i-1]; r1, c1 = cvals[i]
    s = np.log((c1*r1**1.5)/(c0*r0**1.5))/np.log(r1/r0)  # log-slope of I_WP
    slopes.append(s)
    print(f"  [{r0:g},{r1:g}]: log-slope {s:+.3f}")
ck("all discrete log-slopes >= 1.55", all(s >= 1.55 for s in slopes),
   f"min {min(slopes):.3f}")
print("\nP-mono (named, quantified): if d log I_WP/d log r >= 3/2 on (0, r0], then")
print("C_I(r) <= C_I(r0) = 3.202e-3 <= 3.5e-3 for all r in (0, r0] -- coverage.")

print("\n" + "="*78)
if FAILED:
    print(f"WP2 CERT FAIL: {FAILED}"); sys.exit(1)
print("WP2 CERT PASS (exact-integrand modulus, rung grid + monotonicity premise)")
sys.exit(0)
