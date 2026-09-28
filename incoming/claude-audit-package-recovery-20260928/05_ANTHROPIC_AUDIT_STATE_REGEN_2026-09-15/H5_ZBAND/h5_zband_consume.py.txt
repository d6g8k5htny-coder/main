#!/usr/bin/env python3
"""h5_zband_consume.py — OBL-H5-ZBAND consumption certificate (Anthropic family, 2026-09-15).

Discharges OBL-H5-ZBAND ("Z_r window bounds over bands; lo rides H3's c_Z r^2; hi side needs the band
version of the LPW bracket") by CONSUMING the two frozen H3 band carriers:
    lo : H3_BAND_FLOOR  (h3_band_floor.py -> band_normal.txt)   E[G_r] = Z_r/r^2 >= floor(cell)
    hi : H3_BAND_CEIL   (h3_band_ceil.py  -> ceil_normal.txt)   E[G_r] = Z_r/r^2 <= ceil(cell)
and mapping H3's seven certified cells onto the H5 promotion bands [r_{k+1}, r_k].

Fail-closed. Deterministic. No MC. Every numeric input is parsed from the frozen transcripts whose sha256 is
pinned below; nothing is typed in from prose. Both modes (python3 / python3 -O) must be byte-identical.
Mutation suite: each mutation must FAIL (exit 1) or the certificate is void.

Direction discipline (verified in h5_run.py / h5_kernel.py):
    ctx.Z_lo feeds every UPPER bound (rho_hi, coarse_fine_hi, I_hi) via 1/Z_lo  -> needs a FLOOR on Z_r
    ctx.Z_hi feeds only the LOWER side (I_lo, containment display)               -> needs a CEILING on Z_r
"""
import hashlib, os, re, sys
from decimal import Decimal as D, getcontext
getcontext().prec = 50

ROOT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/mnt/agents/output/K3_SIDE24_LB"
MUT = [a for a in sys.argv[1:] if a.startswith("--mut=")]
MUT = MUT[0][6:] if MUT else None
H3 = os.path.join(ROOT, "UPPER2D/H3_closure")
LINES = []

def out(s):
    LINES.append(s); print(s)

def ck(tag, cond):
    out(("PASS " if cond else "FAIL ") + tag)
    if not cond:
        print("H5-ZBAND-CONSUME FAIL"); raise SystemExit(1)

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

# ---------------------------------------------------------------- pins (frozen carriers, bytes)
PINS = {
    "h3_band_floor.py": "a907eeedd767b0b97461df8e237cdca130a607b5dede50ad5ba4b05c957d22a5",
    "band_normal.txt":  "dcb3c8e6152876021c8b116903c9a7c1ac6be7417a9b4042b89c00b911fe565d",
    "h3_band_ceil.py":  "b97c5428a20e5dd33f3e53653d20447f1e27515dcc7dc7a7c2004ebe30029452",
    "ceil_normal.txt":  "26d08534979083275fe14ce789d9cfc4f59ab9250a6cfc830e9acca80a14ae00",
}
out("H5 ZBAND CONSUME (OBL-H5-ZBAND discharge via H3 band floor + ceiling)")
for f, h in PINS.items():
    p = os.path.join(H3, f)
    ck("PIN-%s" % f, os.path.exists(p) and sha(p) == h)
# band_O / ceil_O byte-identical to normal (both-mode discipline of the consumed carriers)
ck("PIN-band_O-identical", sha(os.path.join(H3, "band_O.txt")) == PINS["band_normal.txt"])
ck("PIN-ceil_O-identical", sha(os.path.join(H3, "ceil_O.txt")) == PINS["ceil_normal.txt"])

# ---------------------------------------------------------------- parse certified cell rows
CELL_RE = re.compile(r"^(CB6|CC6) \((0(?:\.\d+)?),(0\.\d+)\] CERTIFIED: E\[G_r\] (>=|<=) ([0-9.]+)")
def parse(path, want_tag, want_sign):
    rows = {}
    for ln in open(os.path.join(H3, path), encoding="utf8"):
        m = CELL_RE.match(ln.strip())
        if m and m.group(1) == want_tag:
            assert m.group(4) == want_sign, ln
            rows[(D(m.group(2)), D(m.group(3)))] = D(m.group(5))
    return rows
FLOOR = parse("band_normal.txt", "CB6", ">=")
CEIL  = parse("ceil_normal.txt",  "CC6", "<=")
ck("PARSE-7-floor-cells", len(FLOOR) == 7)
ck("PARSE-7-ceil-cells",  len(CEIL) == 7)
ck("PARSE-same-cells",    set(FLOOR) == set(CEIL))
CELLS = sorted(FLOOR)
ck("CELLS-tile-(0,0.05]", CELLS[0][0] == 0 and CELLS[-1][1] == D("0.05") and
   all(CELLS[i][1] == CELLS[i + 1][0] for i in range(6)))
ck("ALL_CHECKS_PASS-floor", any(l.strip() == "ALL_CHECKS_PASS" for l in open(os.path.join(H3, "band_normal.txt"))))
ck("ALL_CHECKS_PASS-ceil",  any(l.strip() == "ALL_CHECKS_PASS" for l in open(os.path.join(H3, "ceil_normal.txt"))))

if MUT == "swap":          # consume the floor as a ceiling and vice versa -> must fail
    FLOOR, CEIL = CEIL, FLOOR
if MUT == "inflate_floor": # a floor above the true floor -> must fail
    k = CELLS[-1]; FLOOR[k] = FLOOR[k] + D("0.1")

C_Z = D("1.615489267643502474048123284509081575382")   # H3 c_Z (H3_BAND_FLOOR.md)
U_CAP = D(4)                                          # LPW v4 consumption cap
for c in CELLS:
    ck("CELL-%s-floor<ceil" % (c[1],), FLOOR[c] < CEIL[c])
    ck("CELL-%s-floor>c_Z"  % (c[1],), FLOOR[c] > C_Z)
    ck("CELL-%s-ceil<=U4"   % (c[1],), CEIL[c] <= U_CAP)
uni_lo = min(FLOOR.values()); uni_hi = max(CEIL.values())
ck("UNIFORM-floor=2.30659559567154", uni_lo == D("2.30659559567154"))
ck("UNIFORM-ceil=3.74767948915996",  uni_hi == D("3.74767948915996"))
if MUT == "swap":
    # unreachable in a valid swap (floor<ceil already failed); guard anyway
    ck("MUT-swap-must-have-failed", False)

# ---------------------------------------------------------------- H5 promotion bands
RUNGS = [D("0.05"), D("0.035355"), D("0.025"), D("0.0177"), D("0.0125")]
BANDS = [(RUNGS[i + 1], RUNGS[i]) for i in range(4)] + [(D(0), RUNGS[-1])]
def covering(lo, hi):
    return [c for c in CELLS if c[1] > lo and c[0] < hi]   # cells (a,b] meeting (lo,hi]
out("band                      covering cells                         Z_r/r^2 floor         Z_r/r^2 ceiling")
TABLE = []
for lo, hi in BANDS:
    cv = covering(lo, hi)
    ck("BAND-%s-%s-covered" % (lo, hi), cv and cv[0][0] <= lo and cv[-1][1] >= hi)
    f = min(FLOOR[c] for c in cv); g = max(CEIL[c] for c in cv)
    ck("BAND-%s-%s-floor<ceil" % (lo, hi), f < g)
    TABLE.append((lo, hi, f, g))
    out("(%s, %s]   %-38s %-21s %s" % (lo, hi, ",".join("(%s,%s]" % c for c in cv), f, g))
if MUT == "band_floor_above_cell":   # claim a band floor above one covering cell's floor -> must fail
    lo, hi, f, g = TABLE[0]
    ck("MUT-band-floor-above-cell", f <= min(FLOOR[c] for c in covering(lo, hi)) - D("1e-9"))

# ---------------------------------------------------------------- consumption statements
out("CONSUMED lo: Z_r >= floor(band) * r^2 on every H5 band; feeds ctx.Z_lo (upper-bound side)")
out("CONSUMED hi: Z_r <= ceil(band)  * r^2 on every H5 band; feeds ctx.Z_hi (I_lo side only)")
out("NOTE: band floor %s is stronger than the c_Z=%s the H5 code uses as its H3 input (factor %s)" %
    (uni_lo, "1.6155", (uni_lo / D("1.6154892676435")).quantize(D("0.0001"))))
out("STATUS: OBL-H5-ZBAND DISCHARGED at consumption grade (both sides certified uniform on (0,0.05])")
body = "\n".join(LINES).encode()
print("H5-ZBAND-CONSUME PASS digest=" + hashlib.sha256(body).hexdigest())
