#!/usr/bin/env python3
"""Expected values for checking a different-provider R2 replay of the #451 slice-C package.

Reads git objects only (no checkout changes). For every committed output file listed in the
record's comparison/hashes.json: committed sha256, sha256 after the record's timing mask, the
same-provider rerun's class; for TIMING-ONLY/DIFFERENT files also the record's reproduced copy.
Raw samples: MANIFEST.json sha256/bytes. Writes expected_r2.json next to this script.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "/home/user/main"
REV = "73a3cfcd1ee94b749925462333981f5b9c59e57f"
PKT = "experiments/zeta_fold_exploration"
REC = "reviews/zeta_fold_slice_c_rerun_20261019"

# Same list as reviews/zeta_fold_slice_c_rerun_20261019/scripts/compare.py TIMING.
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


def mask(s):
    for rx, rep in TIMING:
        s = rx.sub(rep, s)
    return s


def show(path):
    r = subprocess.run(["git", "-C", REPO, "show", f"{REV}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def h(b):
    return hashlib.sha256(b).hexdigest()


def main():
    rec = json.loads(show(f"{REC}/comparison/hashes.json"))
    man = json.loads(show(f"{PKT}/MANIFEST.json"))
    out = {"rev": REV, "packet_tree": subprocess.run(
        ["git", "-C", REPO, "rev-parse", f"{REV}:{PKT}"], capture_output=True, text=True).stdout.strip(),
        "files": [], "raw": []}
    bad = 0
    for row in rec["files"]:
        p = row["path"]
        b = show(f"{PKT}/{p}")
        if b is None:
            print("MISSING committed", p)
            bad += 1
            continue
        e = dict(path=p, committed_sha256=h(b), committed_bytes=len(b), same_provider_class=row["class"])
        if e["committed_sha256"] != row["committed_sha256"]:
            print("HASH MISMATCH vs record", p)
            bad += 1
        try:
            e["committed_masked_sha256"] = h(mask(b.decode()).encode())
        except UnicodeDecodeError:
            e["committed_masked_sha256"] = None
        rp = show(f"{REC}/reproduced/{p}")
        if rp is not None:
            e["same_provider_sha256"] = h(rp)
            e["same_provider_masked_sha256"] = h(mask(rp.decode()).encode())
            if row["class"] == "TIMING-ONLY" and e["same_provider_masked_sha256"] != e["committed_masked_sha256"]:
                print("MASK CHECK FAILED", p)
                bad += 1
        out["files"].append(e)
    for r in man["raw_files"]:
        out["raw"].append(dict(path=r["path"], manifest_sha256=r["sha256"], manifest_bytes=r["bytes"]))
    Path(__file__).with_name("expected_r2.json").write_text(json.dumps(out, indent=1) + "\n")
    from collections import Counter
    print("packet tree", out["packet_tree"], "| files", len(out["files"]), Counter(f["same_provider_class"] for f in out["files"]),
          "| raw", len(out["raw"]), "| problems", bad)
    for f in out["files"]:
        if f["same_provider_class"] != "IDENTICAL":
            print(f'{f["same_provider_class"]:12s} {f["path"]:52s} committed {f["committed_sha256"][:16]} masked {str(f["committed_masked_sha256"])[:16]} sameprov {f.get("same_provider_sha256","-")[:16]}')
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
