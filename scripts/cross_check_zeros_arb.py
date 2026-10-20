"""Cross-check the stored zeta zeros with FLINT's ball arithmetic (python-flint).

Each digit string in data/riemann_zeros/first_100_zeros.json is compared with
FLINT's enclosure of the zero with the same index (acb.zeta_zeros). It must lie
within half a unit in its 20th significant digit. The bracketed zero counts of
verify_rh_local are then rechecked with arb.zeta_nzeros, which must give exactly
k at bracket k. FLINT's general contract is that a returned ball contains the
exact value.

Limits: FLINT's zero functions reuse the Turing's-method design of mpmath's
implementation (Arias de Reyna), and the two libraries share their principal
author. This is therefore an independent implementation in independent
arithmetic, not an independent algorithm, author or review. No published table
is consulted.

Usage: python scripts/cross_check_zeros_arb.py [OUT_JSON]   (needs python-flint)
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import flint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "riemann_zeros" / "first_100_zeros.json"
OUT = ROOT / "results" / "zero_cross_check_arb.json"
PREC = 128  # bits, about 38 significant digits
SIG = 20    # the data file prints each value to 20 significant digits


def stored_strings(text):
    """The zeros exactly as written in the data file, before any float rounding."""
    body = re.search(r'"zeros":\s*\[(.*?)\]', text, re.S).group(1)
    return [s.strip() for s in body.split(",")]


def cross_check():
    raw = DATA.read_bytes()
    strings = stored_strings(raw.decode("utf-8"))
    n = len(strings)
    flint.ctx.prec = PREC
    enclosures = flint.acb.zeta_zeros(1, n + 1)  # one extra places the last bracket
    stored = [flint.arb(s) for s in strings]

    rows = []
    for k, (s, value, zero) in enumerate(zip(strings, stored, enclosures), 1):
        digits_before_point = len(s.split(".")[0])
        half_unit = flint.arb(10) ** (digits_before_point - SIG) / 2
        diff = abs(zero.imag - value)
        rows.append({
            "n": k,
            "stored": s,
            "flint_imag": zero.imag.str(30, radius=False),
            "flint_radius": float(zero.imag.rad()),
            "abs_difference_bound": float(diff.mid()) + float(diff.rad()),
            "half_unit": float(half_unit),
            "within_half_unit": bool(diff <= half_unit),
            "real_part_is_half": bool(zero.real == flint.arb(0.5)),
        })

    # Brackets as in RiemannZeta.verify_rh_local: below the first value, the
    # midpoints between neighbours, and between the last value and zero n + 1.
    brackets = ([min(flint.arb(10), stored[0] / 2)]
                + [(a + b) / 2 for a, b in zip(stored, stored[1:])]
                + [(stored[-1] + enclosures[n].imag) / 2])
    counts = [b.zeta_nzeros() for b in brackets]
    bracket_rows = [{"k": k, "t": b.str(25, radius=False),
                     "nzeros": int(c.unique_fmpz()) if c.is_exact() else None}
                    for k, (b, c) in enumerate(zip(brackets, counts))]
    mismatches = [r["k"] for r in bracket_rows if r["nzeros"] != r["k"]]

    agree = sum(r["within_half_unit"] for r in rows)
    return {
        "description": "FLINT ball-arithmetic cross-check of data/riemann_zeros/first_100_zeros.json",
        "data_file": "data/riemann_zeros/first_100_zeros.json",
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "library": {"python-flint": flint.__version__, "FLINT": flint.__FLINT_VERSION__},
        "precision_bits": PREC,
        "reproduce": "python scripts/cross_check_zeros_arb.py",
        "method": ("acb.zeta_zeros(1, N + 1) and arb.zeta_nzeros in ball arithmetic; each stored "
                   "string must lie within half a unit in its 20th significant digit of the "
                   "enclosure, and zeta_nzeros(b_k) must be exactly k"),
        "limits": ("Independent implementation and arithmetic, not an independent algorithm: FLINT's "
                   "zero functions reuse the Turing's-method design of mpmath's implementation, and "
                   "the libraries share their principal author. The zero functions' own documentation "
                   "makes no separate rigor statement. No published table is consulted."),
        "summary": {
            "zeros_checked": n,
            "within_half_unit": agree,
            "all_within_half_unit": agree == n,
            "all_real_parts_half": all(r["real_part_is_half"] for r in rows),
            "max_abs_difference_bound": max(r["abs_difference_bound"] for r in rows),
            "bracket_count_mismatches": mismatches,
            "final_bracket": bracket_rows[-1]["t"],
            "zero_after_last": enclosures[n].imag.str(25, radius=False),
        },
        "zeros": rows,
        "brackets": bracket_rows,
    }


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    result = cross_check()
    out.write_text(json.dumps(result, indent=1) + "\n")
    s = result["summary"]
    print(f"{s['within_half_unit']}/{s['zeros_checked']} stored zeros within half a unit of FLINT's "
          f"enclosure (max |difference| <= {s['max_abs_difference_bound']:.1e}); "
          f"bracket count mismatches: {s['bracket_count_mismatches'] or 'none'} -> {out}")
