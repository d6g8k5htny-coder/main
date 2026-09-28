#!/usr/bin/env python3
"""
verify_gamma_loc_iic_v1.py -- fail-closed certificate for KIMI-DER-026
(AO48-WO-063 Task 3: gamma-LOC condition (ii-c) discharge, verdict
CONDITIONAL modulo the single named premise P-NMZ-gamma).

Every computable link of the discharge chain is verified here:
  S1  F.4 machine-verified margins (exact Fraction arithmetic, R=1,2,5,50,
      tau at its floor): cone/strip/tube budgets, terminal-ledger inputs,
      pin-preserving corrected tube cost 12(eta_R/R)(R^2/tau)h^2.
  S2  Terminal ledger and escape-level-vs-b: p_0(-1/2)=0=P(M),
      p_0(1/2)=-1/6 (S pin = b - ell), p_0(5/4)=49/192,
      49/192 - 1/768 = 65/256 > 1/4 > 0; physical escape level
      b + kappa r^3 * 65/256 > b = 6/5 (kappa=1 testbed); margin ratio
      (65/256)/(1/6) = 195/128 (escape clears b by 195/128 window
      heights); units-mismatch guard 65/256 < 6/5 EXACT (the ledger is a
      margin above the M-pin level, not an absolute height).
  S3  Tail order o(r^p) for all p: e^{-c/r^2} <= p! c^{-p} r^{2p}
      (from e^x >= x^p/p!), checked numerically at the rungs for
      p up to 32; the Lambda-tail threshold 2 e^{-Lambda^2/2} <= 2 r^2
      iff Lambda >= 2 sqrt(ln(1/r)); diagnostic measured clearance
      Lambda in [4.6, 9.7] (UB0 section 4, diagnostic only).
  S4  R3' station table of record (c031 verify.json, sha256
      83313c96...a271): hash binding, Decimal-exact recomputation
      total = proj_c1 + mean_c1 (tol 1e-15), total <= gate 0.1307,
      margin factor 36.0076 (decimal-dps-16), min station v =
      0.9999999577222840 at (5.0, 0.0), closes == true.
  S5  C028 barrier arithmetic: exp(-4.82550294/2) = 0.089569 (dps-6
      ledger print, |err| < 5e-7 at dps-50), and the structural fact
      1 - c >= 1 - 0.089569 = 0.910431 > 0: the far-barrier-only route
      cannot yield 1 - O(exp) (c is O(1), P0 <= 1).
  S6  Hash bindings of the cited corpus (fail-closed).

No bare assert anywhere; ck() raises SystemExit on any failure and
survives `python3 -O`. IIC_CERTIFICATE_PASS is printed only after all
checks pass. Transcripts in normal and -O modes are byte-identical.
"""
from fractions import Fraction as F
from decimal import Decimal, getcontext
import hashlib, json, math
import mpmath as mp

getcontext().prec = 60
mp.mp.dps = 60


def ck(cond, msg="check"):
    if not cond:
        raise SystemExit("CHECK FAILED: " + str(msg))


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def mpf_fr(x):
    """mpf from a Fraction (exact quotient of integer mpfs)."""
    return mp.mpf(x.numerator) / mp.mpf(x.denominator)


print("=== verify_gamma_loc_iic_v1.py -- KIMI-DER-026 certificate ===")
print("scope: gamma-LOC (ii-c) discharge chain links; verdict text lives in KIMI_DER_026.md")

# ---------------------------------------------------------------- S1
print("\n--- S1: F.4 exact-rational margins (R = 1,2,5,50; tau at floor) ---")
K = F(4096)
eps0 = F(1, 1024) / K


def at_R(R):
    R = F(R)
    tau = 2 * K * R**5 + 2 * R
    m = 2 * R / tau
    delta0 = F(1, 512) / R**3
    eta_R = F(1, 8192) / R**3
    out = {}
    out["cone_entry"] = (m <= F(1) / (K * R**4))
    out["pin_grad"] = (R * (1 + m) <= R * F(3, 2)) and (2 * eps0 / R**4 <= F(1, 16) / R**4)
    delta = delta0
    rR = eps0 / R**4  # maximal rR given r R^5 <= eps0
    margin3 = 1 - (F(1, 2) + delta) * R * m - F(1, 2) * R * m**2 * delta - rR * (1 + m)
    out["cone_FX"] = (margin3 >= F(3, 4))
    s4 = F(257, 1024) + F(257, 1024) / K + F(1, 2048) / K**2 + eps0
    out["cone_flux"] = (s4 < F(1, 3))
    out["strip_FX"] = (margin3 >= F(3, 4))
    s5 = F(7, 8) + F(5, 8) / K + F(3, 32) / K**2 + F(4, 3) * eps0
    out["strip_energy"] = (s5 < F(9, 10))
    out["terminal"] = (F(49, 192) - F(1, 768) == F(65, 256)) and (F(65, 256) > F(1, 4))
    out["bracket"] = (F(9869, 16777216) < F(1, 768))
    out["tube_grad"] = (R * (2 + m) <= 3 * R)
    tube_long = 1 - R * m / F(2) - R * m**2 / F(8) - 3 * rR
    out["tube_FX"] = (tube_long >= F(3, 4))
    adverse7 = 1 + F(1) / K + F(1, 4) / K**2 + 6 * eps0 + 4 * R**2 / (K * tau)
    out["tube_flux"] = (adverse7 < 2)
    cone_cost = 3 * eta_R / R
    out["cost_cone"] = (cone_cost < (F(1, 3) - s4))
    strip_cost = 4 * eta_R / (3 * R)
    out["cost_strip"] = (strip_cost < (F(9, 10) - s5))
    tube_cost_ratio = 12 * eta_R / R  # corrected V4 bound = 12/(8192 R^4)
    out["cost_tube"] = (tube_cost_ratio < (2 - adverse7))
    out["cost_tube_small"] = (tube_cost_ratio <= F(12, 8192))
    out["eta_delta"] = (eta_R == delta0 / 16)
    d = delta0
    out["withdrawn_false"] = ((F(1, 4) - (-F(1, 2) + d)**2) == d - d**2) and (d - d**2 < d)
    return out, s4, s5, adverse7, tube_cost_ratio


for R in (1, 2, 5, 50):
    res, s4, s5, adv7, tcr = at_R(R)
    bad = [k for k, v in res.items() if not v]
    ck(not bad, "S1 margins fail at R=%s: %s" % (R, bad))
    print("R=%-2d ok (%d checks)  cone_flux s4=%s  strip s5=%s  adverse7=%s  12eta_R/R=%s"
          % (R, len(res), float(s4), float(s5), float(adv7), float(tcr)))
ck(512 * eps0 == F(1, 8192) and 2 * K * eps0 == F(1, 512), "S1 eps0 identities")
print("S1 ok: 512 eps0 = 1/8192; 2 K eps0 = 1/512; eta_R = 1/(8192 R^3) = delta0/16 [EXACT]")

# ---------------------------------------------------------------- S2
print("\n--- S2: terminal ledger and escape-level-vs-b [EXACT] ---")
b = F(6, 5)


def p0(X):
    return X**3 / F(3) - X / F(4) - F(1, 12)


ck(p0(F(-1, 2)) == 0, "S2 p0(M)")
ck(p0(F(1, 2)) == F(-1, 6), "S2 p0(S)")
ck(p0(F(5, 4)) == F(49, 192), "S2 p0(5/4)")
print("p0(-1/2) = 0 = P(M);  p0(1/2) = -1/6 = P(S);  p0(5/4) = 49/192 [EXACT]")
led = F(49, 192) - F(1, 768)
ck(led == F(65, 256), "S2 ledger value")
ck(led > F(1, 4) and F(1, 4) > 0, "S2 ledger chain")
print("terminal ledger: 49/192 - 1/768 = 65/256 > 1/4 > 0 = P(M) [EXACT]")
# pin-consistent height scale: physical f = b + kappa r^3 P, kappa = 1 (testbed);
# P(S) = -1/6 maps to f(x_S) = b - r^3/6 = b - ell  (ell = r^3/6, Lemma G.2.3 K2)
for r in (F(1, 20), F(1, 40)):
    ell = r**3 / F(6)
    ck(b + r**3 * p0(F(1, 2)) == b - ell, "S2 pin consistency r=%s" % r)
    esc = b + r**3 * led
    ck(esc > b, "S2 escape > b r=%s" % r)
    print("r=%-5s: ell = r^3/6; b + r^3*P(S) = b - ell [EXACT]; escape = b + (65/256) r^3 > b [EXACT] (margin %s, decimal-dps-6)" % (r, float(r**3 * led)))
# margin ratio vs the window depth: (65/256) / (1/6) = 195/128
ck(F(65, 256) * 6 == F(195, 128), "S2 ratio 195/128")
ck(F(65, 256) > F(1, 6), "S2 65/256 > 1/6")
print("margin ratio (65/256)/(1/6) = 195/128 = 1.5234375: escape clears b by 195/128 window heights [EXACT]")
# units-mismatch guard: the ledger is a margin above the M-pin level (= b), NOT an absolute height
ck(F(65, 256) < b, "S2 units guard")
print("units guard: 65/256 = 0.25390625 < 6/5 = 1.2 [EXACT]: the naive '65/256 vs b' comparison is a units error and is NOT the ledger's content")

# ---------------------------------------------------------------- S3
print("\n--- S3: tail order o(r^p) for every fixed p ---")
# mechanism: e^x >= x^p/p! (x>=0) => e^{-c/r^2} <= p! c^{-p} r^{2p} = C_p r^{2p} = o(r^p)
print("derivation: e^x >= x^p/p! (x>=0, p integer>=0) => e^{-c/r^2} <= p! c^{-p} r^{2p}; for each FIXED p this is C_p r^{2p} = o(r^p) [EXACT statement]")
c1 = F(1)
for p in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32):
    for r in (F(1, 20), F(1, 40)):
        rhs = math.factorial(p) * c1**-p * r ** (2 * p)  # exact Fraction
        lhs = mp.exp(-1 / mpf_fr(r) ** 2)
        ck(mpf_fr(rhs) > lhs, "S3 instance p=%d r=%s" % (p, r))
    print("p=%-2d: e^{-1/r^2} <= %d! r^{%d} at r=0.05,0.025 verified (decimal-dps-60 vs EXACT rhs)" % (p, p, 2 * p))
# Lambda-tail threshold: 2 e^{-Lambda^2/2} <= 2 r^2 iff Lambda >= 2 sqrt(ln(1/r))
for r in (F(1, 20), F(1, 40)):
    thr = 2 * mp.sqrt(mp.log(1 / mpf_fr(r)))
    lhs = 2 * mp.e ** (-(thr ** 2) / 2)
    rhs = 2 * mpf_fr(r) ** 2
    ck(abs(lhs - rhs) < mp.mpf("1e-40"), "S3 Lambda threshold r=%s" % r)
    print("r=%-5s: Lambda threshold 2 sqrt(ln(1/r)) = %s (decimal-dps-16); 2 e^{-Lambda^2/2} = 2 r^2 at equality"
          % (r, mp.nstr(thr, 16)))
# growth form: Lambda(r)/sqrt(2 ln(1/r)) -> inf implies o(r^p) for all p
print("growth form: Lambda(r)/sqrt(2 ln(1/r)) -> inf  =>  for each p, eventually Lambda^2/2 >= (p+1) ln(1/r), so e^{-Lambda^2/2} <= r^{p+1} = o(r^p) [EXACT statement]")
# diagnostic only (never theorem-grade): measured m* profile 4.6--9.7 sigma above b (UB0 sec.4)
for lam in ("4.6", "9.7"):
    val = mp.exp(-mp.mpf(lam) ** 2 / 2)
    print("diagnostic (UB0 sec.4, measured): Lambda=%s sigma -> e^{-Lambda^2/2} = %s (decimal-dps-6); threshold at r=0.05 is %s" % (lam, mp.nstr(val, 6), mp.nstr(2 * mp.sqrt(mp.log(20)), 16)))
    ck(mp.mpf(lam) > 2 * mp.sqrt(mp.log(20)), "S3 diagnostic above r=0.05 threshold")

# ---------------------------------------------------------------- S4
print("\n--- S4: R3' station table of record (c031 verify.json) [decimal-dps-16 unless noted] ---")
path = "/mnt/agents/upload/c031 verify.json"
h = sha256(path)
ck(h == "83313c96d1610130451a7e936245a06e80d8cc84df00a20ae9e46e7aeef2a271", "S4 station-table hash")
print("sha256(c031 verify.json) = %s [bound]" % h)
doc = json.loads(open(path, "rb").read().decode(), parse_float=Decimal)
tot = doc["proj_c1"] + doc["mean_c1"]
ck(abs(tot - doc["total"]) <= Decimal("1e-15"), "S4 total = proj_c1 + mean_c1")
print("proj_c1 + mean_c1 = %s vs printed total %s (|diff| <= 1e-15, print rounding)" % (tot, doc["total"]))
ck(doc["total"] <= doc["gate"], "S4 total <= gate")
slack = doc["gate"] - doc["total"]
mf = doc["gate"] / doc["total"]
ck(Decimal("36.0") < mf < Decimal("36.01"), "S4 margin factor range")
print("total = %s <= gate = %s; slack = %s; margin factor = %s (~36x)" % (doc["total"], doc["gate"], slack, mf))
ck(len(doc["stations"]) == 7, "S4 station count")
vmin = min(s["v"] for s in doc["stations"])
smin = [s for s in doc["stations"] if s["v"] == vmin][0]
ck(smin["x"] == Decimal("5.0") and smin["y"] == Decimal("0.0"), "S4 argmin v station")
ck(vmin == Decimal("0.999999957722284"), "S4 min v print")
print("min station v = %s at (x,y) = (%s, %s); 1 - v_min = %s" % (vmin, smin["x"], smin["y"], 1 - vmin))
mmax = max(s["m"] for s in doc["stations"])
gmax = max(s["gradm"] for s in doc["stations"])
ck(gmax == doc["mean_c1"], "S4 mean_c1 = max station gradm")
ck(doc["closes"] is True, "S4 closes flag")
ck(doc["rice"] == Decimal("2.434"), "S4 rice print")
ck(abs(doc["normH"] - Decimal("3.636")) < Decimal("1e-3"), "S4 normH ~ C031 ledger 3.636")
print("max station m = %s; max station gradm = mean_c1 = %s; rice = %s; normH = %s; closes = true" % (mmax, gmax, doc["rice"], doc["normH"]))

# ---------------------------------------------------------------- S5
print("\n--- S5: C028 barrier arithmetic and the far-route structural shortfall ---")
val = mp.exp(-mp.mpf("4.82550294") / 2)
ck(abs(val - mp.mpf("0.089569")) < mp.mpf("5e-7"), "S5 barrier print")
print("exp(-||u0||^2_H/2) with ||u0||^2_H = 4.82550294 (decimal-dps-8, C028 exact-mp) = %s (decimal-dps-20); ledger print 0.089569 (decimal-dps-6), |err| < 5e-7" % mp.nstr(val, 20))
cmax = mp.mpf("0.089569")  # P0 <= 1
ck(1 - cmax > mp.mpf("0.9"), "S5 O(1) shortfall")
print("c = 0.089569 * P0 <= 0.089569 (P0 <= 1), so 1 - c >= %s (decimal-dps-6) > 0: the far-barrier-only route yields 1 - O(1), NOT 1 - O(exp) [structural fact]" % mp.nstr(1 - cmax, 6))
print("projection fact: ||u0'||_H <= ||u0||_H (pin annihilation is an orthogonal projection in H) [EXACT statement]")

# ---------------------------------------------------------------- S6
print("\n--- S6: hash bindings of the cited corpus (fail-closed) ---")
BIND = [
    ("/mnt/agents/upload/AO48-WO-063 - LB-RATE HARDENING CAMPAIGN work order for KIMI - WP resolution + AUD-023 v1.1 + gamma-LOC ii-c + rigorous far-grid-ridge + THM-023 v1.1 all-small-r + transport settlements - agents authorized - 2026-08-04.md",
     "e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2"),
    ("/mnt/agents/upload/C031_LBRATE_Integration.md",
     "e7998ef0d17d951bc89978f9fe32e510019059dd0650c8f0e0ae33273f40f32e"),
    ("/mnt/agents/upload/Lemma UB0.md",
     "c6a90469b6e43c592b661a63c85a6fa6590505f3d22fd888a328ebf7926d41c0"),
    ("/mnt/agents/upload/Lemma UB0 Addendum.md",
     "224438cb740565ff8b2f0acab83a75c8dc80cc15265a55451f43ff930894c94c"),
    ("/mnt/agents/upload/C028 FarAscent Package.md",
     "64b94b6e5342b3f8d2f3a6a8dfd5e1aa38e769ba319eeee87d657074b568e606"),
    ("/mnt/agents/upload/C029 BasinPersistence Package.md",
     "b23c3d42ccc2182e9a59a4ae1b7eda8c73939f543884268a0da3e2cc6880a785"),
    ("/mnt/agents/upload/C030 CountingLemmas Package.md",
     "aed6683edaed6d483704a63cd321fcb300e329a26bf6c143042ead58a60bfc7b"),
    ("/mnt/agents/upload/C022 Observed Update.json",
     "9bc0647b885931ba4dc862bedfb879f9bd12c4cd6d39f51c6e98e54c7f75d5cb"),
    ("/mnt/agents/upload/c031 verify.json",
     "83313c96d1610130451a7e936245a06e80d8cc84df00a20ae9e46e7aeef2a271"),
    ("/mnt/agents/output/SIDE24_gap_fill/SIDE24_GAP_FILL_SUPPLEMENT.md",
     "33609b70d8c4d970d3594ebdc2cb59200508bfb2e53002ae7c6b10b3bc19d41f"),
    ("/mnt/agents/output/SIDE24_gap_fill/scripts/verify_capture_escape_budgets.py",
     "56206651a85d128b3b2d3395f12773b74a55a5efff7539eb86392a2ab416fb38"),
    ("/mnt/agents/output/lb3/LB3_DISCHARGE.md",
     "08153fa42a2d4c01fe9932bd5430ecfc62cfda3f43523420f253a4c0c8526f0c"),
    ("/mnt/agents/output/lb3/lb3_certificate.py",
     "6daa149cb5a7580a4c4e16e973da2cc08486db09219fbe83963319bbd17a6f75"),
]
for pth, want in BIND:
    got = sha256(pth)
    ck(got == want, "S6 hash mismatch: %s" % pth)
    print("bound %s... %s" % (want[:16], pth.split("/")[-1][:60]))

print("\nSCOPE LIMIT: this certificate verifies arithmetic/numeric links only; the single named premise P-NMZ-gamma (near/moderate-zone growing tube clearance, third-saddle arch/rim escape) is stated in KIMI_DER_026.md and is NOT asserted here.")
print("IIC_CERTIFICATE_PASS")
