"""Independent-implementation cross-check of the 100 stored zeta zeros with PARI/GP's lfunzeros.

Usage: python -I pari_cross_check.py DATA_JSON
Reads the stored values as exact decimals (not floats), computes the zeros of zeta on the
critical line with 0 < t <= T with PARI/GP (cypari2) at 60 significant digits, and checks:
  1. PARI finds exactly 100 zeros up to T = 237.147 (between zero 100, ~236.52, and zero 101, ~237.77),
     each refined by bracketed root finding on PARI's Hardy Z-function at 60 digits;
  2. each stored value lies within half a unit in its 20th significant digit of PARI's zero.
PARI's lfunzeros is documented as non-rigorous (zeros can be missed); this is a numerical
cross-check by a separately developed library, not a certificate.
"""
import json, sys
from decimal import Decimal, getcontext
import cypari2, importlib.metadata as md

getcontext().prec = 80
pari = cypari2.Pari()
pari.set_real_precision(60)
data = json.load(open(sys.argv[1]), parse_float=Decimal)
stored = data["zeros"]
assert len(stored) == 100 and all(isinstance(z, Decimal) for z in stored)
L = pari.lfuncreate(1)
T = "237.147"
zs = pari.lfunzeros(L, pari(T))
# lfunzeros returns zeros to about 19 significant digits only; refine each one inside PARI with
# its own Hardy Z-function (lfunhardy) by bracketed root finding at the full working precision.
pari("default(realprecision, 60)")
pari("Lz = lfuninit(lfuncreate(1), [1/2, 0, 240])")
pz = []
for z in zs:
    r = None
    for w in ("1e-15", "1e-12", "1e-9"):
        a, b = pari(f"{z} - {w}"), pari(f"{z} + {w}")
        fa, fb = pari.lfunhardy(pari("Lz"), a), pari.lfunhardy(pari("Lz"), b)
        if fa * fb < 0:
            r = pari(f"solve(t = {a}, {b}, lfunhardy(Lz, t))"); break
    assert r is not None, f"no sign change near {z}"
    pz.append(Decimal(str(pari.precision(r, 60)).replace(" E", "E")))
print(f"cypari2 {md.version('cypari2')}, PARI {'.'.join(map(str, pari.version()[:3]))}, realprecision 60")
print(f"zeros found by PARI with 0 < t <= {T}: {len(pz)}")
worst = Decimal(0); bad = []
for k, (s, p) in enumerate(zip(stored, pz), 1):
    e = s.adjusted()                      # exponent of the leading digit
    half_unit = Decimal(5) * Decimal(10) ** (e - 20)
    d = abs(s - p)
    worst = max(worst, d / half_unit)
    if d > half_unit:
        bad.append((k, str(s), str(p)[:30]))
print(f"stored values within half a unit in the 20th significant digit: {100 - len(bad)}/100")
print(f"largest |stored - PARI| as a fraction of that half unit: {float(worst):.3f}")
print("mismatches:", bad if bad else "none")
print("zero 1 (PARI):", str(pz[0])[:42]); print("zero 100 (PARI):", str(pz[-1])[:42])
