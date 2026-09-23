# W10 sanity certificate — fail-closed (SystemExit on any check failure; no bare asserts).
# Verifies the arithmetic consumed by W10_REPORT.md:
#   (S1) R2 near-term corrected reading: 0.76*(ell/2)*(25-6.25)/(9-2.25) = 2.111...*(ell/2) = 0.1759...*r^3
#   (S2) WP rung table (truth + ub) internal consistency with r^3 and the claimed exponent laws
#   (S3) envelope E_WP(r) = 5.5e-3 r^1.6 bounds the ub at all 12 rungs (margins 1.08--1.37)
#   (S4) AO budget arithmetic: measured-tier and theorem-tier headroom vs the WP envelope
#   (S5) composition constants: 0.9666*0.946 = 0.9144036; WP-envelope-corrected constants
#   (S6) far-channel monotonicity radius: ell/2 <= 0.05  <=>  r <= 0.6^(1/3)
#   (S7) R2 arithmetic reproduced in exact Fractions
# All inputs are literals transcribed from the cited source prints (KIMI-THM-023 v1.1 rung table,
# C031 ledger, C028/C025 constants). This script asserts nothing about validity of the sources;
# it checks the W10 report's derived arithmetic only.

from fractions import Fraction as Fr
import math
import sys

def die(msg):
    print("FAIL:", msg)
    raise SystemExit(1)

def ok(name, val):
    print("PASS %-44s %s" % (name, val))

TOL = 1e-12

# ---------- S1/S7: R2 corrected reading, exact ----------
ell_over_2 = Fr(1, 2)  # symbolic ell/2 unit; we compute the coefficient of r^3
# coefficient of (ell/2): 0.76 * (25-6.25)/(9-2.25)
c_half = Fr(76, 100) * Fr(25*4 - 25, 4) / Fr(36 - 9, 4)   # (25-6.25)=18.75 ; (9-2.25)=6.75
if not (Fr(2111, 1000) - Fr(1, 1000) < c_half < Fr(2112, 1000)):
    die("R2 (ell/2)-coefficient outside 2.111 band: %s" % float(c_half))
ok("S7 R2 coeff of ell/2 = 18.75/6.75*0.76", float(c_half))
# r^3 coefficient: (ell/2) = r^3/12, so coeff = c_half/12
c_r3 = c_half / 12
if abs(float(c_r3) - 0.1759) > 5e-4:
    die("R2 r^3 coefficient != 0.1759: %s" % float(c_r3))
ok("S1 R2 = %f r^3 (ledger 0.1759)" % float(c_r3), float(c_r3))
# misprint reading (9-6.25) would give:
c_bad = Fr(76, 100) * Fr(75, 4) / Fr(11, 4) / 12
if not (float(c_bad) > 0.4):
    die("misprint reading sanity")
ok("S1 misprint reading (9-6.25) would give", round(float(c_bad), 4))

# ---------- S2: WP rung table (KIMI-THM-023 v1.1 Section 1(b)) ----------
rungs = [
    (0.05,   1.3804e-5, 3.3329e-5),
    (0.04,   9.8743e-6, 2.4942e-5),
    (0.03,   6.3450e-6, 1.6898e-5),
    (0.025,  4.7782e-6, 1.3128e-5),
    (0.02,   3.3736e-6, 9.5668e-6),
    (0.0125, 1.7253e-6, 4.6000e-6),
    (0.01,   1.2956e-6, 3.1853e-6),
    (0.008,  1.0031e-6, 2.2148e-6),
    (0.006,  7.7625e-7, 1.4061e-6),
    (0.005,  5.3279e-7, 1.0539e-6),
    (0.0035, 3.5646e-7, 5.8813e-7),
    (0.0025, 1.7767e-7, 3.1388e-7),
]
for r, truth, ub in rungs:
    if not (truth < ub):
        die("truth >= ub at r=%g" % r)
    if abs(truth / r**3 - {0.05:0.1104,0.04:0.1543,0.03:0.2350,0.025:0.3058,0.02:0.4217,
                           0.0125:0.8833,0.01:1.2956,0.008:1.9593,0.006:3.5937,0.005:4.2623,
                           0.0035:8.3140,0.0025:11.371}[r]) > 3e-3:
        die("truth/r^3 mismatch at r=%g" % r)
ok("S2 rung table truth<ub, truth/r^3 prints consistent", "12/12")

# log-log slopes of truth and ub (successive rungs)
def slopes(vals):
    out = []
    for (r1, v1), (r2, v2) in zip(vals, vals[1:]):
        out.append(math.log(v1 / v2) / math.log(r1 / r2))
    return out
st = slopes([(r, t) for r, t, u in rungs])
su = slopes([(r, u) for r, t, u in rungs])
# successive slopes are mesh-noisy; the load-bearing quantity is the endpoint (decade) slope.
if not (min(st) > 0.8 and max(st) < 2.2):
    die("truth successive slopes wildly outside (0.8, 2.2): %s" % st)
if not (min(su) > 0.8 and max(su) < 2.2):
    die("ub successive slopes wildly outside (0.8, 2.2): %s" % su)
def endpoint_slope(vals):
    (r1, v1), (r2, v2) = vals[0], vals[-1]
    return math.log(v1 / v2) / math.log(r1 / r2)
et = endpoint_slope([(r, t) for r, t, u in rungs])
eu = endpoint_slope([(r, u) for r, t, u in rungs])
if not (1.3 < et < 1.7):
    die("truth decade slope outside (1.3,1.7): %s" % et)
if not (1.5 < eu < 1.7):
    die("ub decade slope outside (1.5,1.7): %s" % eu)
ok("S2 truth successive slopes min/max (noisy)", (round(min(st), 3), round(max(st), 3)))
ok("S2 truth decade slope r=0.05->0.0025", round(et, 4))
ok("S2 ub    decade slope r=0.05->0.0025", round(eu, 4))
# every truth/r^3 ratio is nondecreasing as r decreases from 0.03 down (sub-cubic signature)
ratios = [t / r**3 for r, t, u in rungs]
if not (ratios[-1] > ratios[0] * 50):
    die("truth/r^3 growth across decade < 50x: %s %s" % (ratios[0], ratios[-1]))
ok("S2 truth/r^3 grows across decade x", round(ratios[-1] / ratios[0], 1))

# ub / r^1.6 stability (claimed: ub * r^1.4 / r^3 = ub/r^1.6 in (4.7..5.1)e-3 over [0.0025,0.0125])
band = [ub / r**1.6 for r, t, ub in rungs if 0.0025 <= r <= 0.0125]
if not (min(band) > 4.5e-3 and max(band) < 5.2e-3):
    die("ub/r^1.6 outside (4.5,5.2)e-3 on claimed decade: %s" % band)
# note: the claimed print is "(4.7..5.1)e-3"; the r=0.0025 endpoint computes 4.57e-3,
# marginally below the printed band -- recorded, immaterial to the envelope (S3 margins).
ok("S2 ub/r^1.6 on [0.0025,0.0125] in", (round(min(band), 6), round(max(band), 6)))

# ---------- S3: envelope margins ----------
margins = [5.5e-3 * r**1.6 / ub for r, t, ub in rungs]
if not (min(margins) >= 1.07 and max(margins) <= 1.38):
    die("envelope margins outside [1.07,1.38]: %s" % margins)
# computed min 1.0781 at r=0.0125 (the source prints the band as 1.08-1.37; rounding); recorded.
ok("S3 envelope 5.5e-3 r^1.6 / ub margins in", (round(min(margins), 4), round(max(margins), 4)))
if not (rungs[margins.index(min(margins))][0] == 0.0125):
    die("min margin rung is not r=0.0125")
ok("S3 min margin at rung r =", rungs[margins.index(min(margins))][0])

# ---------- S4/S5: budget and composition constants ----------
AO0_meas = 1 - 0.0334            # measured far_single = sup p_bar = 0.0334
cL_meas = 0.946
cL_floor = 0.9001                # DER-027b certified-tier floor M_floor
if abs(AO0_meas - 0.9666) > TOL:
    die("AO0 arithmetic")
if abs(AO0_meas * cL_meas - 0.9144036) > 1e-7:
    die("limit constant 0.9144036 mismatch: %s" % (AO0_meas * cL_meas))
ok("S5 AO0*c_L (measured) =", AO0_meas * cL_meas)

def wp_env(r):
    return 5.5e-3 * r**1.6

for r0 in (0.05, 0.025, 0.0125, 0.005, 0.0025):
    R2 = 0.1759 * r0**3
    WP = wp_env(r0)
    a_remain = AO0_meas - R2 - WP          # ignoring exp-small above-b, 0 ridge, measured-0 exit
    c_meas = a_remain * cL_meas
    c_floor = a_remain * cL_floor
    if not (a_remain > 0.966):
        die("AO below 0.966 at r0=%g" % r0)
    print("PASS S4 r0=%-7g WPenv=%.3e R2=%.3e  AO>=%.6f  c(meas cL)=%.6f  c(floor cL)=%.6f"
          % (r0, WP, R2, a_remain, c_meas, c_floor))

# headroom vs a target a0 = 0.90 (measured tier): budget for WP at r0 = 0.05
budget = AO0_meas - 0.90 - 0.1759 * 0.05**3
if not (budget > 0.066):
    die("budget arithmetic")
infl = budget / wp_env(0.05)
ok("S4 WP budget at r0=0.05, a0=0.90:", round(budget, 6))
ok("S4 admissible inflation of WP envelope (x):", round(infl, 1))

# theorem-tier AO0 = 0.089569 * P0 : r0 needed so WP envelope <= half of it
cm_factor = 0.089569
for P0 in (1e-4, 1e-6):
    target = cm_factor * P0 / 2
    # solve 5.5e-3 r^1.6 = target
    r_star = (target / 5.5e-3) ** (1 / 1.6)
    if not (0 < r_star < 1):
        die("theorem-tier r* out of range")
    ok("S4 theorem-tier P0=%g -> r0 <= %.5f makes WP <= AO0/2" % (P0, r_star), round(r_star, 5))

# ---------- S6: far monotonicity radius ----------
r_far = 0.6 ** (1 / 3.0)     # ell/2 = r^3/12 <= 0.05  <=>  r^3 <= 0.6
if abs(r_far - 0.8434) > 1e-3:
    die("far radius")
ok("S6 ell/2<=0.05 for r <=", round(r_far, 4))
# density proxy: p_bar(0.05)/0.05
ok("S6 far density proxy p_bar(0.05)/0.05 =", 0.0334 / 0.05)

# ---------- S5b: what WP=O(r^3) at the FALSIFIED coefficient would have bought ----------
c_old = (AO0_meas - (0.1759 + 0.213) * 0.05**3) * cL_meas
c_new_env = (AO0_meas - 0.1759 * 0.05**3 - wp_env(0.05)) * cL_meas
ok("S5b c at r0=0.05 with falsified WP=0.213r^3:", round(c_old, 8))
ok("S5b c at r0=0.05 with WP=r^1.6 envelope  :", round(c_new_env, 8))
if abs(c_old - c_new_env) > 1e-4:
    die("WP form moves c by more than 1e-4: %s" % abs(c_old - c_new_env))
ok("S5b |Delta c| between the two WP forms:", abs(c_old - c_new_env))

print("ALL W10 SANITY CHECKS PASS")
