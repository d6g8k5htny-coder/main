"""Check the 100 stored zeta zeros against FLINT/Arb, an implementation independent of mpmath.

Usage: venv/bin/python -I check_zeros_arb.py DATA.json OUT.json
DATA is data/riemann_zeros/first_100_zeros.json from branch
claude/physics-multiscale-reconstruction-a1G1I at e999b63d (blob 16c7c489).

python-flint's acb.zeta_zeros (Arb's acb_dirichlet_zeta_zeros) returns ball enclosures of
the n-th zeros by index, so for each n this checks that the stored decimal is the correctly
rounded 20-significant-digit value of Im(rho_n). Arb's algorithm is not audited here.
Stored values are read as exact decimal text, not as float64.
"""
import decimal
import hashlib
import json
import sys

import flint
from flint import acb, arb, ctx

DATA, OUT = sys.argv[1:3]
raw = open(DATA, "rb").read()
doc = json.loads(raw, parse_float=decimal.Decimal)
stored = doc["zeros"]
assert len(stored) == 100 and all(isinstance(v, decimal.Decimal) for v in stored)

ctx.prec = 256  # bits; about 77 decimal digits
decimal.getcontext().prec = 120
SIG = 20


def ball(x):
    """Exact decimal endpoints [lo, hi] of an arb ball."""
    mid, rad = x.mid(), x.rad()
    m = decimal.Decimal(mid.str(70, radius=False, more=True))
    r = decimal.Decimal(arb(rad).str(10, radius=False, more=True)) if rad != 0 else decimal.Decimal(0)
    # str(70) of the midpoint can itself round; widen by one unit in its last printed place
    slack = decimal.Decimal(10) ** (m.adjusted() - 69)
    return m - r - slack, m + r + slack


def round_sig(d, sig=SIG):
    q = decimal.Decimal(10) ** (d.adjusted() - sig + 1)
    return d.quantize(q, rounding=decimal.ROUND_HALF_EVEN).normalize()


rows, problems = [], []
zeros = acb.zeta_zeros(1, 101)
for n in range(1, 102):
    z = zeros[n - 1]
    re_ok = z.real.contains(arb(0.5))
    lo, hi = ball(z.imag)
    rlo, rhi = round_sig(lo), round_sig(hi)
    determined = rlo == rhi
    row = {"n": n, "arb_imag": z.imag.str(30, radius=True), "re_contains_half": bool(re_ok),
           "rounding_determined": determined}
    if n <= 100:
        s = stored[n - 1]
        err_hi = max(abs(s - lo), abs(s - hi))
        row.update({"stored": str(s), "max_abs_error": "%.3e" % err_hi,
                    "correctly_rounded_%d" % SIG: (determined and s.normalize() == rlo),
                    "stored_sig_digits": len(s.as_tuple().digits)})
        if not (determined and s.normalize() == rlo):
            problems.append(n)
    if not re_ok:
        problems.append(("re", n))
    rows.append(row)

# Bracketed counts with Arb's zeta_nzeros, at the same brackets the report describes:
# b0 below the first entry, midpoints b_k between entries k and k+1, and b100 between
# entry 100 and Arb's zero 101.
vals = [arb(str(v)) for v in stored]
z101 = zeros[100].imag
brackets = [arb(10)] + [(vals[k] + vals[k + 1]) / 2 for k in range(99)] + [(vals[99] + z101) / 2]
counts = []
for k, b in enumerate(brackets):
    c = arb.zeta_nzeros(b) if hasattr(arb, "zeta_nzeros") else acb.zeta_nzeros(b)
    ok = c.contains(arb(k)) and c.rad() == 0
    counts.append({"k": k, "t": b.str(12, radius=False), "N": c.str(5), "exact_equals_k": bool(ok)})
    if not ok:
        problems.append(("count", k))

# Certified residual bound |zeta(1/2 + i s)| at the stored decimals.
res_max = arb(0)
for v in vals:
    r = abs(acb(arb(0.5), v).zeta())
    res_max = r if r.upper() > res_max.upper() else res_max
gaps = [stored[k + 1] - stored[k] for k in range(99)]

summary = {
    "data_sha256": hashlib.sha256(raw).hexdigest(),
    "python_flint": flint.__version__,
    "prec_bits": ctx.prec,
    "zeros_checked": 100,
    "all_correctly_rounded_to_%d_sig" % SIG: not [p for p in problems if isinstance(p, int)],
    "not_correctly_rounded": [p for p in problems if isinstance(p, int)],
    "all_real_parts_contain_half": all(r["re_contains_half"] for r in rows),
    "max_abs_error_stored_vs_arb": max(decimal.Decimal(r["max_abs_error"]) for r in rows if "stored" in r).to_eng_string(),
    "max_arb_radius_imag": max(zeros[n].imag.rad() for n in range(100)).str(5),
    "counts_all_exact": all(c["exact_equals_k"] for c in counts),
    "count_brackets": len(counts),
    "zero_101_arb": z101.str(25, radius=True),
    "b100": brackets[100].str(12, radius=False),
    "residual_upper_bound_max": res_max.upper().str(5, radius=False),
    "min_gap": str(min(gaps)),
    "strictly_increasing": all(g > 0 for g in gaps),
    "problems": [str(p) for p in problems],
}
json.dump({"summary": summary, "zeros": rows, "counts": counts}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1))
