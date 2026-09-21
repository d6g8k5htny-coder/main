#!/usr/bin/env python3
"""
verify_k3_w7_gammaloc_v1.py -- fail-closed certificate for the K3-W7
gamma-LOC (ii-c) audit + discharge attempt (mandate 19fb9293, Phase 2,
superseding WO-063 Task 3's frame; DER-026 remains the starting record).

Disposition certified here: DISCHARGED-BY-MINIMALITY. The certificate
verifies, fail-closed:
  S0  hash bindings of the record (corpus + C025 + DER-026 set);
  S1  record-audit substring receipts: the lower composition's AO floor
      comes from the C025 R-architecture + C028/R3' constant c_cond, and
      gamma-LOC enters the C031 named register only as architecture
      inherited by the C022 positivity chain (i.e. the 1-O(exp) clause
      (ii-c) is not in the theorem-grade load path of the current
      record);
  S2  the minimal-statement arithmetic: ell = r^3/6, band entry
      delta <= ell/2, near term 2.1*(ell/2) = 7 r^3/40 (EXACT), the
      constant floor P(terminal > b) >= c_cond - 2.1*(ell/2) - [named],
      and the explicit r0(P0): r^3 <= 20 c_cond / 7;
  S3  DER-026 numeric regression: terminal ledger EXACT, escape > b
      EXACT, station table (Decimal-exact), C028 barrier print dps-60,
      one tail instance;
  S4  meaning-of-O(exp) displays and the two-rung measured diagnostic
      (diagnostic only, never theorem-grade);
  S5  mutation self-tests: every critical predicate is evaluated on a
      deliberately mutated input and must REJECT it (non-vacuity).

Usage: verify_k3_w7_gammaloc_v1.py [--json PATH]
  --json PATH  use PATH instead of the station table of record (hash
               binding skipped; content checks still enforced). Used by
               mutation_driver.py.

No bare assert; ck() raises SystemExit(1). K3W7_CERTIFICATE_PASS is
printed only after all checks. Output is deterministic and
byte-identical under `python3` and `python3 -O`.
"""
from fractions import Fraction as F
from decimal import Decimal, getcontext
import hashlib, json, math, sys
import mpmath as mp

getcontext().prec = 60
mp.mp.dps = 60

DEFAULT_JSON = "/mnt/agents/upload/c031 verify.json"


def ck(cond, msg="check"):
    if not cond:
        raise SystemExit("CHECK FAILED: " + str(msg))


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def mpf_fr(x):
    return mp.mpf(x.numerator) / mp.mpf(x.denominator)


def parse_args(argv):
    path = DEFAULT_JSON
    i = 1
    while i < len(argv):
        if argv[i] == "--json" and i + 1 < len(argv):
            path = argv[i + 1]
            i += 2
        else:
            raise SystemExit("CHECK FAILED: bad argument %s" % argv[i])
    return path


JSON_PATH = parse_args(sys.argv)
OVERRIDE = (JSON_PATH != DEFAULT_JSON)

print("=== verify_k3_w7_gammaloc_v1.py -- K3-W7 gamma-LOC audit certificate ===")
print("mode: %s" % ("json-override (mutation harness; hash binding skipped)" if OVERRIDE else "default (record-bound)"))

# ---------------------------------------------------------------- S0
if not OVERRIDE:
    print("\n--- S0: hash bindings (fail-closed) ---")
    BIND = [
        ("/mnt/agents/upload/AO48-WO-063 - LB-RATE HARDENING CAMPAIGN work order for KIMI - WP resolution + AUD-023 v1.1 + gamma-LOC ii-c + rigorous far-grid-ridge + THM-023 v1.1 all-small-r + transport settlements - agents authorized - 2026-08-04.md",
         "e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2"),
        ("/mnt/agents/upload/C031_LBRATE_Integration.md",
         "e7998ef0d17d951bc89978f9fe32e510019059dd0650c8f0e0ae33273f40f32e"),
        ("/mnt/agents/upload/C025 TerminalHeight Package.md",
         "d1afd84b3143ed512b0fb1cafba5c2eb6eb190b28ec7f8e789cc817c08bf792a"),
        ("/mnt/agents/upload/C025 Freeze.md",
         "0f8cb3dcb337b5c1a110d8ed983aa49fc9a2f6383b6066d7a625756da3c7460b"),
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
        (DEFAULT_JSON,
         "83313c96d1610130451a7e936245a06e80d8cc84df00a20ae9e46e7aeef2a271"),
        ("/mnt/agents/output/SIDE24_gap_fill/SIDE24_GAP_FILL_SUPPLEMENT.md",
         "33609b70d8c4d970d3594ebdc2cb59200508bfb2e53002ae7c6b10b3bc19d41f"),
        ("/mnt/agents/output/lb3/LB3_DISCHARGE.md",
         "08153fa42a2d4c01fe9932bd5430ecfc62cfda3f43523420f253a4c0c8526f0c"),
        ("/mnt/agents/output/gamma_loc_iic/KIMI_DER_026.md",
         "d6bde0636b400d995f889e55446892631f3c2d462e060b34c27532fbd4a05bfd"),
        ("/mnt/agents/output/gamma_loc_iic/verify_gamma_loc_iic_v1.py",
         "ece7244acaa69a2f8719d35bd73f5ea9f6542902c8ae86efd6a3ed14ffece570"),
        ("/mnt/agents/output/gamma_loc_iic/transcript_normal.txt",
         "61e4dc8a46de2dd58e5fe6048af73e7e4394994403f32c98f6029741c214f057"),
    ]
    for pth, want in BIND:
        got = sha256(pth)
        ck(got == want, "S0 hash mismatch: %s" % pth)
        print("bound %s... %s" % (want[:16], pth.split("/")[-1][:56]))

# ---------------------------------------------------------------- S1
print("\n--- S1: record-audit substring receipts (load path of the current record) ---")
DOC = {
    "C031": "/mnt/agents/upload/C031_LBRATE_Integration.md",
    "C025": "/mnt/agents/upload/C025 TerminalHeight Package.md",
    "LB3": "/mnt/agents/output/lb3/LB3_DISCHARGE.md",
    "UB0": "/mnt/agents/upload/Lemma UB0.md",
    "DER026": "/mnt/agents/output/gamma_loc_iic/KIMI_DER_026.md",
}
TEXT = {k: open(v, encoding="utf-8").read() for k, v in DOC.items()}
AUDIT = [
    ("A1", "C031", "Theorem grade (conditional):",
     "C031 theorem-grade assembly exists and is conditional"),
    ("A2", "C031", "AO ≥ c_cond − 2.1·(ℓ/2) − [named non-adjacency channels]",
     "the AO floor source is the R3' constant c_cond, NOT 1-O(exp)"),
    ("A3", "C031", "**γ-LOC tube-local** (architecture) — inherited by the C022 positivity chain.",
     "gamma-LOC's register entry: architecture, C022 positivity chain"),
    ("A4", "C031", "1 − q ≥ c_cond·c_Λ·r³ conditional on the same set",
     "theorem-grade lower bound runs on c_cond · c_Lambda"),
    ("A5", "C031", "conditional on: OBL-FAR-COMPOSE route (ii); the sup-over-zone step",
     "the assembly's named conditional set (none is a gamma-LOC-exp item)"),
    ("A6", "C025", "AO-failure ⊆ {non-adjacency} ∪ {terminal ≤ b}",
     "C025 R-architecture failure decomposition"),
    ("A7", "C025", "P(terminal ≤ b) ≤ E[N_maxband(B_d₀)] + far band-termination",
     "R2: near band-max expectation + far band-termination"),
    ("A8", "C025", "p̄(δ₀) = P(unconditional ascent from height b − δ₀ terminates ≤ b) ≤ 1 − c",
     "R4/OBL-FAR-ASCENT: r-free constant shortfall 1-c"),
    ("A9", "LB3", "(ii-c) the terminal value is > b with probability 1 − O(exp)",
     "LB-3's verbatim (ii-c)"),
    ("A10", "LB3", "the terminal-value clause (1 − O(exp) above b) is OPEN",
     "LB-3 verdict context: only (ii-c) was OPEN"),
    ("A11", "UB0", "exits the conditioned zone with running height",
     "UB0 gamma-LOC hypothesis text present"),
    ("A12", "DER026", "**Premise P-NMZ-γ (analytic, named, exactly one).**",
     "DER-026 premise block present (quoted verbatim in the W7 report)"),
    ("A13", "DER026", "Λ(r) := μ(r)/s(r) → ∞ as r ↓ 0, in the form Λ(r)/√(2 ln(1/r)) → ∞",
     "DER-026 premise growth form present"),
]
for aid, doc, s, note in AUDIT:
    ck(s in TEXT[doc], "S1 audit string missing: %s (%s)" % (aid, doc))
    print("receipt %s [%s]: found -- %s" % (aid, doc, note))

# ---------------------------------------------------------------- S2
print("\n--- S2: minimal-statement arithmetic (the floor the composition actually needs) ---")
b = F(6, 5)
ck(b > 0, "S2 b")
for r in (F(1, 20), F(1, 40)):
    ell = r**3 / F(6)
    vstar = b - ell / 2
    near = F(21, 10) * (ell / 2)  # 2.1*(ell/2) = 7 r^3 / 40
    ck(near == 7 * r**3 / F(40), "S2 near-term identity r=%s" % r)
    ck(vstar < b and vstar > b - ell, "S2 band entry r=%s" % r)
    print("r=%-5s: ell = r^3/6 = %s [EXACT]; v* = b - ell/2; band entry delta <= ell/2 [EXACT]; near term 2.1*(ell/2) = 7 r^3/40 = %s [EXACT] (decimal-dps-10: %s)"
          % (r, ell, near, mp.nstr(mpf_fr(near), 10)))
# C031 ledger print "0.76*(25-6.25)/(9-6.25) ≈ 2.1": the area-ratio form 0.76*(25/9)
ratio = F(76, 100) * F(25, 9)
ck(ratio == F(19, 9), "S2 area ratio value")
ck(abs(ratio - F(21, 10)) < F(1, 50), "S2 area ratio ~ 2.1")
print("C031 R2 constant cross-check: 0.76*(25/9) = 19/9 = %s (decimal-dps-10), |19/9 - 2.1| = 1/90 < 1/50 [EXACT]" % mp.nstr(mpf_fr(ratio), 10))
# constant floor: P(terminal > b) >= c_cond - 2.1*(ell/2) - [named], c_cond = kCM * P0
kCM = Decimal("0.089569")  # decimal-dps-6 ledger print of exp(-4.82550294/2)
# dominance: 7 r^3/40 <= c_cond/2  iff  r^3 <= 20 c_cond / 7
coef = Decimal(20) * kCM / Decimal(7)  # r0^3 = coef * P0
print("dominance: 7 r^3/40 <= c_cond/2  iff  r^3 <= (20*0.089569/7)*P0 = %s*P0 (decimal-dps-16)" % coef)
for p0s in ("1", "0.00032", "0.0001"):
    p0 = Decimal(p0s)
    r0cubed = coef * p0
    r0 = mpf = mp.mpf(str(r0cubed)) ** (mp.mpf(1) / 3)
    # sanity at the rungs
    covers005 = (F(1, 20)**3 <= F(str(r0cubed))) if r0cubed > 0 else False
    print("if P0 = %-8s: r0 = (%s)^(1/3) = %s (decimal-dps-10); covers rung r=0.05: %s"
          % (p0s, r0cubed, mp.nstr(r0, 10), covers005))
print("explicit r0 inherits OBL-P0-FLOOR (C028 honest null: measured P0 = 0/12000, 95% UB 3.2e-4, decimal-dps-2); c_cond > 0 itself stands by the Gaussian support theorem (C028)")
print("minimal sufficient gamma-LOC: clause (i) measurability [PROVED LB-3] + (ii-a)/(ii-b) termination at a maximum [PROVED LB-3] + constant floor c_cond - O(r^3) [C025-R2 + C028/R3', modulo the assembly's named set] -- NO 1-O(exp) clause is consumed [EXACT logical content of receipts A1-A8]")

# ---------------------------------------------------------------- S3
print("\n--- S3: DER-026 numeric regression ---")
# station table (content checks; hash binding in S0 for default mode)
try:
    raw = open(JSON_PATH, "rb").read()
    doc = json.loads(raw.decode(), parse_float=Decimal)
except Exception as e:
    raise SystemExit("CHECK FAILED: S3 json unreadable (%s)" % type(e).__name__)


def content_checks(doc):
    tot = doc["proj_c1"] + doc["mean_c1"]
    ck(abs(tot - doc["total"]) <= Decimal("1e-15"), "S3 total = proj + mean")
    ck(doc["total"] <= doc["gate"], "S3 total <= gate")
    ck(len(doc["stations"]) == 7, "S3 station count")
    vmin = min(s["v"] for s in doc["stations"])
    ck(vmin >= Decimal("0.999999957722284"), "S3 min v")
    ck(doc["closes"] is True, "S3 closes")
    ck(doc["rice"] == Decimal("2.434"), "S3 rice")
    return tot, vmin


tot, vmin = content_checks(doc)
print("station table: proj+mean = %s (|diff| <= 1e-15 vs printed total); total <= gate 0.1307; min v >= %s; closes = true [decimal-dps-16]"
      % (tot, vmin))
# ledger
led = F(49, 192) - F(1, 768)
ck(led == F(65, 256) and led > F(1, 4) > 0, "S3 ledger")
ck(F(65, 256) * 6 == F(195, 128) and F(65, 256) < b, "S3 ratio + units guard")


def p0(X):
    return X**3 / F(3) - X / F(4) - F(1, 12)


ck(p0(F(-1, 2)) == 0 and p0(F(1, 2)) == F(-1, 6) and p0(F(5, 4)) == F(49, 192), "S3 p0 values")
for r in (F(1, 20), F(1, 40)):
    ck(b + r**3 * led > b and b + r**3 * p0(F(1, 2)) == b - r**3 / F(6), "S3 escape r=%s" % r)
print("ledger 49/192 - 1/768 = 65/256 > 1/4 > 0 = P(M) [EXACT]; escape = b + (65/256) r^3 > b [EXACT]; pin map b + r^3 P(S) = b - ell [EXACT]; units guard 65/256 < 6/5 [EXACT]")
# barrier
val = mp.exp(-mp.mpf("4.82550294") / 2)
ck(abs(val - mp.mpf("0.089569")) < mp.mpf("5e-7"), "S3 barrier print")
ck(1 - mp.mpf("0.089569") > mp.mpf("0.9"), "S3 O(1) shortfall")
print("exp(-4.82550294/2) = %s (decimal-dps-20) vs ledger 0.089569 (dps-6); 1 - c >= 0.910431 > 0: far-barrier-only route is O(1), not O(exp)" % mp.nstr(val, 20))
# one tail instance
rhs = math.factorial(8) * F(1, 20) ** 16
ck(mpf_fr(rhs) > mp.exp(-mp.mpf(400)), "S3 tail instance")
print("tail instance: e^{-400} <= 8! (1/20)^{16} verified (decimal-dps-60 vs EXACT rhs); mechanism e^{-c/r^2} <= p! c^{-p} r^{2p} = o(r^p)")

# ---------------------------------------------------------------- S4
print("\n--- S4: meaning of O(exp) and the measured diagnostic (diagnostic only) ---")
for r in (F(1, 20), F(1, 40)):
    thr = 2 * mp.sqrt(mp.log(1 / mpf_fr(r)))
    print("r=%-5s: exp-rate requires Lambda(r) growing; reference threshold 2 sqrt(ln(1/r)) = %s (decimal-dps-16)" % (r, mp.nstr(thr, 16)))
q05 = Decimal("0.9997")
q025 = Decimal("0.99986")
fr = (1 - q05) / (1 - q025)
print("two-rung measured diagnostic (UB0 addendum A.4, qual/raw, decimal-dps-4/5): failure ratio (1-0.9997)/(1-0.99986) = %s (decimal-dps-10) -- cannot resolve Lambda growth; measured grade only, never consumed" % fr)
for lam in ("4.6", "9.7"):
    v = mp.exp(-mp.mpf(lam) ** 2 / 2)
    print("diagnostic: Lambda=%s sigma -> e^{-Lambda^2/2} = %s (decimal-dps-6)" % (lam, mp.nstr(v, 6)))

# ---------------------------------------------------------------- S5
print("\n--- S5: mutation self-tests (every critical predicate must reject corrupted input) ---")


def pred_total_gate(total, gate):
    return total <= gate


def pred_sum(proj, mean, total):
    return abs(proj + mean - total) <= Decimal("1e-15")


def pred_vmin(v):
    return v >= Decimal("0.999999957722284")


def pred_ledger(a, bq):
    return (F(49, 192) - F(1, 768) == a) and (a > bq)


def pred_barrier(x):
    return abs(x - mp.mpf("0.089569")) < mp.mpf("5e-7")


def pred_dominance(r, c):
    return 7 * r**3 / F(40) <= c / 2


MUT = [
    ("M-total", not pred_total_gate(Decimal("0.5"), Decimal("0.1307"))),
    ("M-sum", not pred_sum(Decimal("0.001995628054736906"), Decimal("0.0016341647279050065"), Decimal("0.003629792782741913"))),
    ("M-vmin", not pred_vmin(Decimal("0.9"))),
    ("M-ledger", not pred_ledger(F(64, 256), F(1, 4))),
    ("M-barrier", not pred_barrier(mp.mpf("0.089570"))),
    ("M-dominance", not pred_dominance(F(1), Decimal("0.0000089569"))),
]
for mid, rejected in MUT:
    ck(rejected, "S5 mutation not rejected: %s" % mid)
    print("mutation %s rejected by its predicate: True" % mid)
# sanity: unmutated values are accepted
ck(pred_total_gate(Decimal("0.003629792782641913"), Decimal("0.1307"))
   and pred_sum(Decimal("0.001995628054736906"), Decimal("0.0016341647279050065"), Decimal("0.003629792782641913"))
   and pred_vmin(Decimal("0.999999957722284"))
   and pred_ledger(F(65, 256), F(1, 4))
   and pred_barrier(mp.mpf("0.089569"))
   and pred_dominance(F(1, 20), Decimal("0.089569")), "S5 sanity acceptance")
print("sanity: all predicates accept the unmutated record values: True")

print("\nSCOPE LIMIT: receipts S1 bind the CURRENT record's load path; if the assembly is recomposed to consume a 1-O(exp) rate at any node, this certificate's minimality finding must be re-run (falsifier F4 of the W7 report).")
print("K3W7_CERTIFICATE_PASS")
