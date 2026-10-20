"""Compare the stored zeta zeros with published OEIS tables.

Zeros 1-10 are compared at 20 significant digits with the OEIS decimal
expansions of their imaginary parts. All stored zeros are compared at integer
resolution with A013629 (floor) and A002410 (nearest integer). The b-files are
read from a directory the caller downloaded them into (see Usage); the output
records each file's URL and SHA-256, so a later run can tell whether the OEIS
data changed.

Limits: beyond zero 10 the published tables used here give only integer parts,
so they catch gross errors only. OEIS data is curated and published, but this
comparison does not by itself establish where each table's digits came from.

Source: OEIS Foundation Inc., The On-Line Encyclopedia of Integer Sequences,
https://oeis.org (CC BY-SA 4.0). Only comparison results and short digit
excerpts are recorded.

Usage:
    for a in A058303 A065434 A065452 A065453 A192492 A305741 A305742 A305743 \
             A305744 A306004 A013629 A002410; do
        curl -sS -o "$DIR/b${a#A}.txt" "https://oeis.org/$a/b${a#A}.txt"; done
    python scripts/cross_check_zeros_oeis.py DIR [OUT_JSON]
"""

import hashlib
import json
import re
import sys
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP, getcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "riemann_zeros" / "first_100_zeros.json"
OUT = ROOT / "results" / "zero_cross_check_oeis.json"
SIG = 20
EXPANSIONS = ["A058303", "A065434", "A065452", "A065453", "A192492",
              "A305741", "A305742", "A305743", "A305744", "A306004"]  # zeros 1..10
FLOOR, NEAREST = "A013629", "A002410"


def stored_strings(text):
    body = re.search(r'"zeros":\s*\[(.*?)\]', text, re.S).group(1)
    return [s.strip() for s in body.split(",")]


def bfile(directory, anumber):
    """The (n, a(n)) rows of an OEIS b-file, and the file's provenance."""
    url = f"https://oeis.org/{anumber}/b{anumber[1:]}.txt"
    raw = (Path(directory) / f"b{anumber[1:]}.txt").read_bytes()
    rows = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            n, value = line.split()[:2]
            rows.append((int(n), int(value)))
    return rows, {"sequence": anumber, "url": url, "sha256": hashlib.sha256(raw).hexdigest(),
                  "terms": len(rows)}


def cross_check(directory):
    getcontext().prec = 80
    raw = DATA.read_bytes()
    strings = stored_strings(raw.decode("utf-8"))
    stored = [Decimal(s) for s in strings]
    sources = []

    expansion_rows = []
    for k, anumber in enumerate(EXPANSIONS, 1):
        rows, meta = bfile(directory, anumber)
        sources.append(meta)
        offset = rows[0][0]  # digits before the decimal point
        digits = "".join(str(d) for _, d in rows[:offset + 40])
        published = Decimal(digits[:offset] + "." + digits[offset:])  # truncated, error < 1e-38
        s = strings[k - 1]
        digits_before_point = len(s.split(".")[0])
        half_unit = Decimal(10) ** (digits_before_point - SIG) / 2
        diff = abs(published - stored[k - 1])
        expansion_rows.append({
            "n": k, "sequence": anumber, "stored": s,
            "published_first_25_significant_digits": digits[:offset] + "." + digits[offset:25],
            "abs_difference": float(diff), "half_unit": float(half_unit),
            "within_half_unit": diff <= half_unit + Decimal(10) ** -38,
        })

    integer_rows = {}
    for anumber, rounding in ((FLOOR, ROUND_FLOOR), (NEAREST, ROUND_HALF_UP)):
        rows, meta = bfile(directory, anumber)
        sources.append(meta)
        published = dict(rows)
        mine = [int(v.to_integral_value(rounding=rounding)) for v in stored]
        integer_rows[anumber] = {
            "published_first_n": [published.get(n) for n in range(1, len(stored) + 1)],
            "stored_values_rounded": mine,
            "mismatches": [n for n in range(1, len(stored) + 1) if published.get(n) != mine[n - 1]],
        }

    agree = sum(r["within_half_unit"] for r in expansion_rows)
    return {
        "description": "OEIS published-table comparison of data/riemann_zeros/first_100_zeros.json",
        "data_file": "data/riemann_zeros/first_100_zeros.json",
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "attribution": "OEIS Foundation Inc., The On-Line Encyclopedia of Integer Sequences, https://oeis.org (CC BY-SA 4.0)",
        "reproduce": "download the b-files with curl (see the script's docstring), then python scripts/cross_check_zeros_oeis.py DIR",
        "limits": ("Zeros 1-10 at 20 significant digits; zeros 11-100 at integer resolution only "
                   "(floor and nearest integer), which catches gross errors only."),
        "sources": sources,
        "summary": {
            "expansions_checked": len(expansion_rows),
            "expansions_within_half_unit": agree,
            "all_expansions_within_half_unit": agree == len(expansion_rows),
            "floor_mismatches": integer_rows[FLOOR]["mismatches"],
            "nearest_integer_mismatches": integer_rows[NEAREST]["mismatches"],
            "integer_checks_cover": len(stored),
        },
        "expansions": expansion_rows,
        "integer_checks": integer_rows,
    }


if __name__ == "__main__":
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    result = cross_check(sys.argv[1])
    out.write_text(json.dumps(result, indent=1) + "\n")
    s = result["summary"]
    print(f"zeros 1-{s['expansions_checked']}: {s['expansions_within_half_unit']} within half a unit of the "
          f"OEIS expansions; floor mismatches {s['floor_mismatches'] or 'none'}, nearest-integer mismatches "
          f"{s['nearest_integer_mismatches'] or 'none'} over {s['integer_checks_cover']} zeros -> {out}")
