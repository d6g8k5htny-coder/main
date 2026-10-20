"""Compare the stored zeta zeros with Odlyzko's published table of zeros.

Odlyzko's zeros6 lists the first 2,001,052 zeros to 9 decimals, stated accurate
to within 4e-9. This script reads it from the Sage project's package
database_odlyzko_zeta-20061209.tar.bz2, after checking the tarball's SHA-256
against the value Sage records in build/pkgs/database_odlyzko_zeta/checksums.ini.
Each stored value must lie within 4e-9 plus half a unit in the table's ninth
decimal of the table's entry with the same index, and the table's entry 101
must lie above the final bracket that verify_rh_local uses.

Limits: the copy is checked against Sage's recorded checksum, not against
Odlyzko's own host. A 9-decimal table checks about 11 significant digits.

Usage:
    curl -sS -O https://mirrors.mit.edu/sage/spkg/upstream/database_odlyzko_zeta/database_odlyzko_zeta-20061209.tar.bz2
    python scripts/cross_check_zeros_odlyzko.py database_odlyzko_zeta-20061209.tar.bz2 [OUT_JSON]
"""

import hashlib
import json
import re
import sys
import tarfile
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "riemann_zeros" / "first_100_zeros.json"
OUT = ROOT / "results" / "zero_cross_check_odlyzko.json"
TARBALL_SHA256 = "8919f01992718b9bf5c0602dbf16dd9d6f58b141b25f67f5cfd59f6cd0f9a0d4"
MEMBER = "database_odlyzko_zeta-20061209/zeros6"
STATED_ACCURACY = Decimal("4e-9")
HALF_LAST_DIGIT = Decimal("5e-10")


def stored_strings(text):
    body = re.search(r'"zeros":\s*\[(.*?)\]', text, re.S).group(1)
    return [s.strip() for s in body.split(",")]


def published_zeros(tarball, count):
    raw = Path(tarball).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != TARBALL_SHA256:
        raise SystemExit(f"tarball SHA-256 {digest} does not match Sage's recorded {TARBALL_SHA256}")
    with tarfile.open(tarball, "r:bz2") as archive:
        handle = archive.extractfile(MEMBER)
        lines = [handle.readline().decode("ascii").strip() for _ in range(count)]
    return lines, digest


def cross_check(tarball):
    getcontext().prec = 60
    raw = DATA.read_bytes()
    strings = stored_strings(raw.decode("utf-8"))
    n = len(strings)
    published, digest = published_zeros(tarball, n + 1)
    tolerance = STATED_ACCURACY + HALF_LAST_DIGIT

    rows = []
    for k, (s, p) in enumerate(zip(strings, published), 1):
        diff = abs(Decimal(s) - Decimal(p))
        rounded = Decimal(s).quantize(Decimal("1e-9"), rounding=ROUND_HALF_EVEN)
        rows.append({"n": k, "stored": s, "published": p, "abs_difference": float(diff),
                     "within_tolerance": diff <= tolerance,
                     "published_equals_stored_rounded_to_9_decimals": rounded == Decimal(p)})
    stored = [Decimal(s) for s in strings]
    final_bracket = (stored[-1] + Decimal(published[n])) / 2
    agree = sum(r["within_tolerance"] for r in rows)
    return {
        "description": "Odlyzko's zeros6 table compared with data/riemann_zeros/first_100_zeros.json",
        "data_file": "data/riemann_zeros/first_100_zeros.json",
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "source": {
            "table": "A. Odlyzko, zeros6: the first 2,001,052 zeros of the Riemann zeta function, accurate to within 4e-9",
            "package": "Sage database_odlyzko_zeta-20061209.tar.bz2, member " + MEMBER,
            "downloaded_from": "https://mirrors.mit.edu/sage/spkg/upstream/database_odlyzko_zeta/database_odlyzko_zeta-20061209.tar.bz2",
            "tarball_sha256": digest,
            "checksum_reference": "https://raw.githubusercontent.com/sagemath/sage/develop/build/pkgs/database_odlyzko_zeta/checksums.ini",
        },
        "reproduce": "see this script's docstring",
        "tolerance": float(tolerance),
        "limits": ("The copy is verified against Sage's recorded checksum, not against Odlyzko's host; "
                   "a 9-decimal table checks about 11 significant digits."),
        "summary": {
            "zeros_checked": n,
            "within_tolerance": agree,
            "all_within_tolerance": agree == n,
            "max_abs_difference": max(r["abs_difference"] for r in rows),
            "all_published_equal_stored_rounded_to_9_decimals": all(
                r["published_equals_stored_rounded_to_9_decimals"] for r in rows),
            "published_strictly_increasing": all(Decimal(a) < Decimal(b) for a, b in zip(published, published[1:])),
            "published_zero_after_last": published[n],
            "final_bracket_from_published": str(final_bracket),
            "zero_after_last_above_final_bracket": Decimal(published[n]) > final_bracket > stored[-1],
        },
        "zeros": rows,
    }


if __name__ == "__main__":
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    result = cross_check(sys.argv[1])
    out.write_text(json.dumps(result, indent=1) + "\n")
    s = result["summary"]
    print(f"{s['within_tolerance']}/{s['zeros_checked']} stored zeros within {result['tolerance']:.1e} of "
          f"Odlyzko's zeros6 (max |difference| {s['max_abs_difference']:.1e}); entry 101 = "
          f"{s['published_zero_after_last']} -> {out}")
