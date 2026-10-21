#!/usr/bin/env python3
"""Check a different-provider R2 replay record against the committed slice-C outputs.

Two inputs, either or both:
  --outputs DIR   a directory mirroring the packet's results/ tree with the replay's text outputs
                  (raw bytes are hashed and masked here; .npz are hashed only);
  --reported F    a JSON file mapping packet-relative paths (e.g. "results/dbn/kappa_check.out")
                  to the performer's reported sha256 (string) or {"sha256": ..., "class": ...}.
Expected values come from expected_r2.json (built by expected_r2.py from main 73a3cfcd; packet tree
bca06db8 is identical at 5cab7663). Prints one line per file and a summary; exit 0 if every file is
accounted for and nothing is DIFFERENT beyond the two known residual-trail files.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMING = [
    (re.compile(r"(setup )[0-9.]+s"), r"\1<T>s"),
    (re.compile(r"(zeta zeros in )[0-9.]+s"), r"\1<T>s"),
    (re.compile(r"^time [0-9.eE+-]+\s*$", re.M), "time <T>"),
    (re.compile(r"(scan time )[0-9.eE+-]+"), r"\1<T>"),
    (re.compile(r"( time )[0-9.eE+-]+$", re.M), r"\1<T>"),
    (re.compile(r'("seconds(?:_zero_finding|_total)?": )[0-9.eE+-]+'), r"\1<T>"),
    (re.compile(r'("sec_per_1e5_evals_at_9e5": )[0-9.eE+-]+'), r"\1<T>"),
    (re.compile(r"\([0-9.]+ ?s\)"), "(<T>s)"),
    (re.compile(r"(samples in )[0-9.]+s"), r"\1<T>s"),
    (re.compile(r"^(init )[0-9.]+( N \d+)$", re.M), r"\1<T>\2"),
]
KNOWN_RESIDUAL = {"results/dbn/indep_ht_c910.out", "results/dbn/indep_ht_c3435.out"}


def mask(s):
    for rx, rep in TIMING:
        s = rx.sub(rep, s)
    return s


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs")
    ap.add_argument("--reported")
    args = ap.parse_args()
    exp = json.loads((HERE / "expected_r2.json").read_text())
    reported = json.loads(Path(args.reported).read_text()) if args.reported else {}
    rows, counts = [], {}
    for e in exp["files"]:
        p = e["path"]
        got = None
        cls = "MISSING"
        if args.outputs and (Path(args.outputs) / p.split("results/", 1)[1]).exists():
            b = (Path(args.outputs) / p.split("results/", 1)[1]).read_bytes()
            got = sha(b)
            if got == e["committed_sha256"]:
                cls = "IDENTICAL"
            else:
                try:
                    cls = "TIMING-ONLY" if sha(mask(b.decode()).encode()) == e["committed_masked_sha256"] else "DIFFERENT"
                except UnicodeDecodeError:
                    cls = "DIFFERENT(binary)"
        elif p in reported:
            r = reported[p]
            got = r if isinstance(r, str) else r.get("sha256")
            if got == e["committed_sha256"]:
                cls = "IDENTICAL"
            else:
                claimed = None if isinstance(r, str) else r.get("class")
                masked = None if isinstance(r, str) else r.get("masked_sha256")
                if masked and masked == e["committed_masked_sha256"]:
                    cls = "TIMING-ONLY"
                else:
                    cls = f"UNVERIFIED(claimed {claimed})" if claimed else "DIFFERENT?"
        counts[cls] = counts.get(cls, 0) + 1
        rows.append((p, cls, e["same_provider_class"], (got or "-")[:16]))
    for r in rows:
        print("%-52s %-24s same-provider=%-12s got=%s" % r)
    raw = []
    for e in exp["raw"]:
        p = e["path"]
        r = reported.get(p)
        got = r if isinstance(r, str) else (r or {}).get("sha256") if r else None
        raw.append((p, got == e["manifest_sha256"] if got else None))
    print("raw:", raw)
    print("summary:", counts)
    bad = [p for p, c, _, _ in rows if (c.startswith("DIFFERENT") and p not in KNOWN_RESIDUAL) or c == "MISSING"]
    print("needs attention:", bad)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
