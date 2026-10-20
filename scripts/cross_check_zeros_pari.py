"""Cross-check the stored zeta zeros with PARI/GP's L-function code (cypari2).

PARI is a separate code base by different authors from mpmath and FLINT, which
share their zero-finding design and principal author. Each digit string in
data/riemann_zeros/first_100_zeros.json is compared with PARI's lfunzeros at 40
and at 60 significant digits; it must lie within half a unit in its 20th
significant digit of both.

PARI documents lfunzeros as a naive sign-change search that can miss zeros
closer than 2*pi/omega, where omega = divz*(log(T/2pi) + 1 + 2*log(2/pi)) for
zeta. The script records that spacing at the top of the range, reruns the
count with a mesh twice as fine, and checks that the smallest gap between
stored values exceeds the spacing. PARI's lfun values carry a documented
absolute error below 2^-B, B = realbitprecision. This is not interval
arithmetic, and no published table is consulted.

Usage: python scripts/cross_check_zeros_pari.py [OUT_JSON]   (needs cypari2)
"""

import hashlib
import json
import math
import re
import sys
from decimal import Decimal, getcontext
from importlib.metadata import version
from pathlib import Path

import cypari2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "riemann_zeros" / "first_100_zeros.json"
OUT = ROOT / "results" / "zero_cross_check_pari.json"
SIG = 20          # the data file prints each value to 20 significant digits
DIGITS = (40, 60)  # PARI working precisions, in significant digits
DIVZ = (8, 16)     # PARI's default search mesh, and one twice as fine


def stored_strings(text):
    """The zeros exactly as written in the data file, before any float rounding."""
    body = re.search(r'"zeros":\s*\[(.*?)\]', text, re.S).group(1)
    return [s.strip() for s in body.split(",")]


def mesh_spacing(t, divz):
    """2*pi/omega, the closest zero spacing lfunzeros assumes, for zeta at height t."""
    return 2 * math.pi / (divz * (math.log(t / (2 * math.pi)) + 1 + 2 * math.log(2 / math.pi)))


def cross_check():
    getcontext().prec = 80
    raw = DATA.read_bytes()
    strings = stored_strings(raw.decode("utf-8"))
    stored = [Decimal(s) for s in strings]
    n = len(strings)
    pari = cypari2.Pari()
    # Stay below zero n + 1 (237.7698...) and above the last stored value.
    limit = "237.2"

    runs = {}
    for digits in DIGITS:
        pari.set_real_precision(digits)
        for divz in DIVZ:
            zeros = pari(f"lfunzeros(lfuncreate(1), {limit}, {divz})")
            runs[(digits, divz)] = [Decimal(str(z)) for z in zeros]
    counts = {f"{d} digits, divz {v}": len(z) for (d, v), z in runs.items()}
    base, fine = runs[(DIGITS[0], DIVZ[0])], runs[(DIGITS[1], DIVZ[0])]

    rows = []
    for k, (s, value) in enumerate(zip(strings, stored), 1):
        digits_before_point = len(s.split(".")[0])
        half_unit = Decimal(10) ** (digits_before_point - SIG) / 2
        diffs = [abs(z[k - 1] - value) if len(z) >= k else None for z in (base, fine)]
        rows.append({
            "n": k,
            "stored": s,
            "pari_60": str(fine[k - 1]) if len(fine) >= k else None,
            "abs_difference_40": float(diffs[0]) if diffs[0] is not None else None,
            "abs_difference_60": float(diffs[1]) if diffs[1] is not None else None,
            "half_unit": float(half_unit),
            "within_half_unit": all(d is not None and d <= half_unit for d in diffs),
        })

    smallest_gap = min(b - a for a, b in zip(stored, stored[1:]))
    spacing = mesh_spacing(float(limit), DIVZ[0])
    precision_shift = max(abs(a - b) for a, b in zip(base, fine)) if len(base) == len(fine) else None
    agree = sum(r["within_half_unit"] for r in rows)
    return {
        "description": "PARI/GP cross-check of data/riemann_zeros/first_100_zeros.json",
        "data_file": "data/riemann_zeros/first_100_zeros.json",
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "library": {"cypari2": version("cypari2"), "PARI": ".".join(map(str, pari.version()))},
        "reproduce": "python scripts/cross_check_zeros_pari.py",
        "method": (f"lfunzeros(lfuncreate(1), {limit}, divz) at {DIGITS} significant digits and divz "
                   f"{DIVZ}; each stored string must lie within half a unit in its 20th significant "
                   "digit of PARI's zero at both precisions"),
        "limits": ("Independent code base and authors, not interval arithmetic: lfunzeros is documented "
                   "as a naive sign-change search that may miss zeros closer than its mesh, and lfun "
                   "values carry a documented absolute error below 2^-B. No published table is consulted."),
        "summary": {
            "zeros_checked": n,
            "zeros_found_below_limit": counts,
            "counts_all_equal_n": all(c == n for c in counts.values()),
            "within_half_unit": agree,
            "all_within_half_unit": agree == n,
            "max_abs_difference": max(r["abs_difference_40"] for r in rows if r["abs_difference_40"] is not None),
            "max_shift_between_40_and_60_digits": float(precision_shift) if precision_shift is not None else None,
            "search_limit": limit,
            "mesh_spacing_at_limit_divz_8": spacing,
            "smallest_stored_gap": float(smallest_gap),
            "gap_exceeds_mesh_spacing": float(smallest_gap) > spacing,
        },
        "zeros": rows,
    }


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    result = cross_check()
    out.write_text(json.dumps(result, indent=1) + "\n")
    s = result["summary"]
    print(f"{s['within_half_unit']}/{s['zeros_checked']} stored zeros within half a unit of PARI's zeros "
          f"(max |difference| {s['max_abs_difference']:.1e}); zeros found below {s['search_limit']}: "
          f"{sorted(set(s['zeros_found_below_limit'].values()))}; smallest gap {s['smallest_stored_gap']:.3f} "
          f"vs mesh spacing {s['mesh_spacing_at_limit_divz_8']:.3f} -> {out}")
