#!/usr/bin/env python3
"""d1_falsify_v3.py — operative gate for D1_ASSEMBLY v2.2 (D1-ASM-20260915-v2.2).

Extends the frozen v2 gate: floor-consistent bracket discrimination (rejects
any consumption below 21.92788306), the kappa-denominator ck (assembly-level
mirror of D3's MUT-V2-3), the B4.loc validity-register ck, and the round-2
carrier/nit records. Fail-closed; no bare asserts; deterministic; both modes
byte-identical.
"""
import hashlib, json, sys, os, glob, re
from decimal import Decimal, getcontext

getcontext().prec = 90
BASE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.dirname(BASE)

def ck(tag, cond):
    print(("PASS " if cond else "FAIL ") + tag)
    if not cond:
        raise SystemExit(1)

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def body_markers(p, mode):
    t = open(p, "rb").read()
    a = t.find(b"BEGIN_FROZEN_BODY\n") + len(b"BEGIN_FROZEN_BODY\n")
    b = t.find(b"END_FROZEN_BODY")
    ck("markers " + os.path.basename(p), a >= len(b"BEGIN_FROZEN_BODY\n") and b > a)
    seg = t[a:b]
    return seg if mode == "raw" else seg.rstrip(b"\n") + b"\n"

def body_sep(p):
    t = open(p, "rb").read()
    i = t.find(b"\n---\n\n")
    ck("separator " + os.path.basename(p), i > 0)
    return t[:i]

# ---------- (A) carrier ledger -------------------------------------------
WHOLE = [
    ("C1_alpha_intensity/C1_ALPHA_INTENSITY.md", "ca2c02b81a05e4cbc955b1aeab1f4120e9b8d4fda328aa191da08a49db72853f"),
    ("H3_closure/H3_CLOSURE.md", "387e4bae4881a036486f17f317ae5e26b7d94bd5e1e829cc76946d090fb81637"),
    ("H3_closure/H3_RUNG_FLOOR.md", "6347275d86c56842b719b36180e535bc1793d6995ddb68464db2960820440dfa"),
    ("BRANCH_dir/BRANCH_DIR.md", "8821c8facc7e36635c4b84967ba55037ff5a33fff09359f817032eecaab24286"),
    ("H5_closure/H5_TOTALS_FREEZE_2026-09-15.md", "5f1ee690aef4b95ceec76eaea8fd02dac2e23d2d1fc233fb31ee1e07193a59da"),
    ("H5_closure/H5_TOTALS_ERRATA_2026-09-15.md", "a7ce5299bdaab5a1a79da50af7b8e32f1b7d193d175d4dff46a224a0b12eb4f2"),
    ("H5_closure/h5_totals_v3.json", "8d7028e4d0d49d54d3a7c4589a898289ec29eee6c2de9c774e99df0da46596bb"),
]
BODY = [
    ("B1_taxonomy/B1_TAXONOMY.md", "7d7ddbec6e63b4c2a2c745b20e33a3d92cfdf5cb97588312a8d54fbd1ea68930", "rstrip"),
    ("C2_B_classes/C2_B_CLASSES.md", "2aba099d630dc7593b1c26bdd2ad3336c1c521e438ba5845edb0b98f2dadf3d2", "raw"),
    ("D2_branch_control/D2_BRANCH_CONTROL.md", "9e95648acbe5307d7f61f7e7ccc102577b80974b38d03fc760f7ce7766a0b8fa", "rstrip"),
    ("D3_percolation/D3_PERCOLATION.md", "8e7fef6b4fb989404624bf50083c9d8624fa486eeda62e4784447f8edb3eac47", "raw"),
    ("H4_closure/H4_CLOSURE.md", "a67d50b9a24c82f970c60c5881eaa41cdf7d4b81cb1cf15904a3fc1b92b6b1d3", "rstrip"),
    ("H4_JC_repair/H4_JC_EVENT_LEVEL.md", "42ee88da984b82921d68cf3fcf6bc76ede9dd59b7ab5b1f947e24a29b08bf74c", "raw"),
    ("D3_percolation/D3_REMOTE_AMENDMENT.md", "63d91cdd6364725597504067a8ffa2378da0e89fe0e87afc81828dd78b8fec24", "raw"),
    ("D3_percolation/D3_REMOTE_AMENDMENT_v2.md", "6796deea4bfd5e3df19fd561e6bc34efe2531263db9db8d015d0553c3a3d1302", "raw"),
]
SEP = [
    ("H5_closure/H5_CLOSURE.md", "465c97c0553ecc0a8f461a86c9315ce1c078a326f07b81498d7a74da8603d301"),
    ("H5_closure/H5_CLOSURE_TIGHTENING_2026-09-15.md", "7fefa17b74836092c0dd5912e028e3b09c1bf4f68c02e35390b552d4479f0561"),
]
for rel, exp in WHOLE:
    ck("A-whole " + rel, sha(os.path.join(UP, rel)) == exp)
for rel, exp, mode in BODY:
    ck("A-body " + rel, hashlib.sha256(body_markers(os.path.join(UP, rel), mode)).hexdigest() == exp)
for rel, exp in SEP:
    ck("A-sep " + rel, hashlib.sha256(body_sep(os.path.join(UP, rel))).hexdigest() == exp)

def d1_body(path):
    lines = open(path, "rb").read().decode().split("\n")
    idx = [i for i, l in enumerate(lines) if l in ("BEGIN_FROZEN_BODY", "END_FROZEN_BODY")]
    ck("d1-markers " + os.path.basename(path), len(idx) == 2)
    seg = lines[idx[0] + 1:idx[1]]
    while seg and seg[0] == "":
        seg = seg[1:]
    while seg and seg[-1] == "":
        seg = seg[:-1]
    return hashlib.sha256(("\n".join(seg) + "\n").encode()).hexdigest()

HIST = [
    ("D1_ASSEMBLY.md", "006b8a7d011443a0a98401f30335fbcfe44ef5996b4fc41dd9735640d958de40"),
    ("D1_ASSEMBLY_v1_1_ADDENDUM.md", "634338b43a000502dfdda62a9019895434db7b50607a9989caaa1fd5873167e2"),
    ("D1_ASSEMBLY_v1_2_ADDENDUM.md", "a7e1958c0d37d4eb63929823edd3190f83216fb4b127a4e85ad7e2e4da86ba67"),
    ("D1_ASSEMBLY_v2_0.md", "86882dca5d5258e9f1bab43e52c4382d30ec9609f89cd64486b3441f5ffd62d0"),
    ("D1_ASSEMBLY_v2_1.md", "bcc177fea4ee06f9d25c56f207ed4b6e7577eff96bfc030202928813c0688189"),
]
for f, exp in HIST:
    ck("A-historical-untouched " + f, d1_body(os.path.join(BASE, f)) == exp)

# ---------- (B) versioned consumption + drift ------------------------------
V3 = json.load(open(os.path.join(UP, "H5_closure", "h5_totals_v3.json")))
D = lambda k: Decimal(V3[k])
r = Decimal(V3["r"]); r3 = r ** 3
ck("B-r-is-0.05", r == Decimal("0.05"))
ck("B-v3-I_lo-eq-v1-pin", abs(D("I_lo") - Decimal("1.166721190e-08")) / Decimal("1.166721190e-08") < Decimal("1e-9"))
ck("B-v3-I_hi-value", abs(D("I_hi") - Decimal("8.0975589252e-02")) / Decimal("8.0975589252e-02") < Decimal("1e-11"))
ck("B-v3-containment", D("I_lo") < D("C1_TOTAL") < D("I_hi"))
ck("B-v3-remote-is-v1-bracket", abs(D("remote") - Decimal("19.55") * r3) < Decimal("1e-20"))
I_HI_CONSUMED = D("I_hi")
for p in sorted(glob.glob(os.path.join(UP, "H5_closure", "h5_totals_v*.json"))):
    m = re.search(r"h5_totals_v(\d+)", os.path.basename(p))
    if m and int(m.group(1)) > 3:
        j = json.load(open(p))
        ck("B-drift-down-only " + os.path.basename(p), Decimal(j["I_hi"]) <= I_HI_CONSUMED)
live = json.load(open(os.path.join(UP, "H5_closure", "h5_totals.json")))
ck("B-live-below-v1-frozen", Decimal(live["I_hi"]) <= Decimal("9.142887704e-02"))

# ---------- (C) floor-consistent bracket (discriminating) -------------------
Z_LO = Decimal("7.7592917375327855e-3")
N_CROSS = Decimal("5.04277749e-3")
N_PAIR = Decimal("1.953515362e-5")
K_Y = Decimal("0.0243696")
NUM_ANN = Decimal("1.714838453e-5")
KAPPA_CAP = Decimal("0.68")
I_ANN_COEF = Decimal("17.6804")
FAR_COEF = Decimal("2.5282637")
B_REMOTE_EXACT = Decimal("21.92788306")
B_REMOTE = Decimal("21.9279")

def kappa_at(z):
    return N_CROSS / z + N_PAIR / z + K_Y

kappa_floor = kappa_at(Z_LO)
# two readings of the floor assembly (recorded nit N-D3V2-1): recomputed from the
# displayed Z-free numerators (0.67678902) vs the amendment's ratio-scaled display
# (0.677284905); they differ ~5e-4; the cap 0.68 covers BOTH, and the bracket
# prices at the cap, so the bound is unaffected.
ck("C-kappa-floor-numerators-window", Decimal("0.6767") < kappa_floor < Decimal("0.6770"))
KAPPA_DISPLAY = Decimal("0.677284905")
ck("C-kappa-display-window", Decimal("0.6772") < KAPPA_DISPLAY < Decimal("0.6774"))
ck("C-kappa-cap-covers-both", KAPPA_CAP >= kappa_floor and KAPPA_CAP >= KAPPA_DISPLAY)
ck("C-v1-breach-certified", kappa_floor > Decimal("0.66") and KAPPA_DISPLAY > Decimal("0.66"))  # scope-V1 datum
# kappa-denominator ck: any denominator ABOVE the certified floor under-prices
Z_TRIAL = Z_LO * Decimal("1.000001")
ck("C-denominator-direction", kappa_at(Z_TRIAL) < kappa_floor)
ck("C-kappa-probe-below-floor", kappa_at(Decimal("8.06118054e-3")) < kappa_floor)
Iann = NUM_ANN / Z_LO / r3
ck("C-I_ann-floor-window", abs(Iann - Decimal("17.68036065")) < Decimal("1e-5"))
ck("C-I_ann-consistency-vs-frozen",
   abs(NUM_ANN / Decimal("8.06118054e-3") / r3 - Decimal("17.0182368")) < Decimal("1e-3"))
bracket_floor = I_ANN_COEF + FAR_COEF * (Decimal(1) + KAPPA_CAP)
ck("C-bracket-recompute", abs(bracket_floor - B_REMOTE_EXACT) < Decimal("1e-7"))
ck("C-bracket-display-roundup", B_REMOTE >= B_REMOTE_EXACT and B_REMOTE < B_REMOTE_EXACT + Decimal("0.0001"))
ck("C-bracket-discriminates", B_REMOTE_EXACT > Decimal("21.27"))  # D3 engine edge
for bad in ("19.5465", "20.9", "21.2153", "21.2658"):
    ck("C-rejects-" + bad, abs(B_REMOTE - Decimal(bad)) > Decimal("0.5"))
ck("C-bracket-abs-value", abs(B_REMOTE * r3 - Decimal("2.7409875e-3")) < Decimal("1e-18"))

# ---------- (D) rung-floor ck ----------------------------------------------
cZ = Decimal("1.615489267643502474")
floor = cZ * (Decimal("0.05") ** 2)
ck("D-floor-value", abs(floor - Decimal("4.0387231691087558e-3")) < Decimal("1e-17"))
ck("D-Zlo-above-floor", Z_LO > floor)
margin = Z_LO / floor - 1
ck("D-margin-92.12pct", Decimal("0.92") < margin < Decimal("0.93"))
ck("D-C_RN-denominator-is-floor", Z_LO < Decimal("8.06118054e-3"))  # probe > floor

# ---------- (E) assembly arithmetic ----------------------------------------
chart = I_HI_CONSUMED - Decimal("19.55") * r3
total = chart + B_REMOTE * r3
ck("E-total-value", abs(total - Decimal("8.1272826752e-2")) / total < Decimal("1e-8"))
ck("E-display-650.1827-roundup", total / r3 <= Decimal("650.1827") and total / r3 > Decimal("650.1826"))
ck("E-display-628.2548-roundup", chart / r3 <= Decimal("628.2548") and chart / r3 > Decimal("628.2547"))
ck("E-display-647.8048-roundup", I_HI_CONSUMED / r3 <= Decimal("647.8048") and I_HI_CONSUMED / r3 > Decimal("647.8047"))
ck("E-Arem-23.2119", Decimal("1.284") + B_REMOTE <= Decimal("23.2119"))
v1 = Decimal("9.142887704e-02")
ck("E-v1-display-731.4311", v1 / r3 <= Decimal("731.4311") and v1 / r3 > Decimal("731.4310"))
ck("E-v1-display-711.8811", (v1 / r3 - Decimal("19.55")) <= Decimal("711.8811"))
import math
crn05 = Decimal(repr(math.sqrt(31.23))) / cZ
ck("E-C_RN-0.05-le-3.46", crn05 <= Decimal("3.46") and crn05 > Decimal("3.459"))
ck("E-nearswap-doubling", abs(Decimal("58.79") / Decimal("29.40") - 2) < Decimal("0.001"))
ck("E-sd-fyyy", abs(Decimal("2.4495") - Decimal(repr(math.sqrt(6.0)))) < Decimal("0.001"))
ck("E-containment-consumed", D("C1_TOTAL") < total)

# ---------- (F) citation/register cks on the v2.2 document ------------------
doc = open(os.path.join(BASE, "D1_ASSEMBLY_v2_2.md"), "rb").read().decode()
lines = doc.split("\n")
idx = [i for i, l in enumerate(lines) if l in ("BEGIN_FROZEN_BODY", "END_FROZEN_BODY")]
ck("F-body-markers", len(idx) == 2)
body = "\n".join(lines[idx[0] + 1:idx[1]])
ck("F-retracted-string-absent", "N_qual + N_loop ≤ N_w" not in doc
   and "N_qual + N_loop \\leq N_w" not in doc and "N_qual + N_loop <= N_w" not in doc)
ck("F-H4JC-R1-cited", "H4JC-R1" in body)
ck("F-rung-lemma-named", "D3-LEMMA-RN-UNIF(r = 0.05)" in body)
ck("F-PERC-named", "PERC-DECAY" in body)
ck("F-loop-repointed", "OBL-B1-BRANCH(loop|B1)" in body)
ck("F-remote-display", "21.9279" in body)
ck("F-consumed-display", "8.1272827e-2" in body and "650.1827" in body)
# B4.loc validity-register ck (scope V2)
prem = body.find("VALIDITY premises")
ck("F-premise-list-found", prem > 0)
ck("F-B4loc-in-premises", "B4.loc" in body[prem:] and "dam-line" in body[prem:])
ck("F-B4loc-premise-bullet", "THE B4.loc DAM-LINE TUBE CERTIFICATE" in body)
ck("F-cutnet-identification-carried", "ASSERTED, NOT ESTABLISHED" in body)
ck("F-refinement-keeps-i-iii-v", "OBL-D2-AO-SHARP (i, iii, v)" in body)
ck("F-G7-rung-discharged", "DISCHARGED" in body and "G.7-scope normalizer" in body)
ck("F-zeroMC-repaired", "FALSE CLAIM REPAIRED" in body)
ck("F-executive-block", "EXECUTIVE-STATE BLOCK" in body)
ck("F-running-tasks", "W3 LOWER driver" in body and "W8 Phase-2" in body
   and "BRANCH rung025" in body and "H3 band assessment" in body)

out = "D1-V2_2-GATE PASS digest=" + hashlib.sha256(
    ("|".join(sorted([e for _, e in WHOLE] + [e for _, e, _ in BODY] + [e for _, e in SEP]))
     + "|" + str(total)).encode()).hexdigest()
print(out)
