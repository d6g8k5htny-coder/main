#!/usr/bin/env python3
"""NON-CERTIFYING, source-exposed numerical validation; no theorem acceptance.

Protocol frozen in main#458 comment 6409941584 BEFORE this execution. Execute
the exact source rs_zeta.py bytes, retain every random test point/reference/
approximation, check grid counts at two steps, and report observed errors.
The new seed is not a confirmation of a persistence-statistics hypothesis.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

SEED = 2026102601
WINDOWS = [(20000., 30000.), (200000., 300000.), (700000., 800000.)]
COUNT_WINDOWS = [(25100., 25150.), (250100., 250130.), (750100., 750120.)]
POINTS = 128


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    # No reuse of the replay's coefficient file: the source regenerates it.
    os.environ["ZETA_FOLD_RAW"] = str((a.output / "raw").resolve())
    if (a.output / "raw/rs_cheb_coeffs.npz").exists():
        raise SystemExit("fresh-check coefficient file already exists")
    sys.path.insert(0, str(a.packet.resolve() / "stats"))
    import numpy as np
    import mpmath as mp
    import rs_zeta as rs
    mp.mp.dps = 40
    rng = np.random.default_rng(SEED)
    report = dict(label="NON-CERTIFYING", scientific_effect="NONE", seed=SEED,
                  python=sys.version, numpy=np.__version__, mpmath=mp.__version__,
                  platform=platform.platform(), reference_dps=40,
                  source_sha256=hashlib.sha256(Path(rs.__file__).read_bytes()).hexdigest(),
                  source_exposed=True, organizational_independence_credit=0)
    coeff = a.output / "raw/rs_cheb_coeffs.npz"
    report["coefficients"] = dict(bytes=coeff.stat().st_size,
        sha256=hashlib.sha256(coeff.read_bytes()).hexdigest())
    tests = []
    with (a.output / "point_comparisons.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["lo", "hi", "t_float64", "siegelz_dps40", "Z_corr1", "Z_corr2", "Z_corr3"])
        for lo, hi in WINDOWS:
            t0 = time.time()
            ts = np.sort(rng.uniform(lo, hi, POINTS))
            references = [mp.siegelz(mp.mpf(float(t))) for t in ts]
            refs = np.array([float(x) for x in references])
            vals = [rs.Z(ts, nterms_corr=k) for k in (1, 2, 3)]
            for i, t in enumerate(ts):
                w.writerow([lo, hi, repr(float(t)), mp.nstr(references[i], 40),
                            *[repr(float(v[i])) for v in vals]])
            tests.append(dict(window=[lo,hi],points=POINTS,
                max_abs_errors=[float(np.max(np.abs(v-refs))) for v in vals],
                max_abs_reference=float(np.max(np.abs(refs))),wall_seconds=time.time()-t0))
            print("reference window", tests[-1], flush=True)
    report["random_point_comparisons"] = tests
    counts = []
    for lo, hi in COUNT_WINDOWS:
        # nzeros counts all nontrivial zeros, including multiplicity. A finite
        # floating sign scan is not a proof of completeness or of RH.
        n0, n1 = int(mp.nzeros(lo)), int(mp.nzeros(hi))
        grids = []
        for step in (0.01, 0.005):
            grid = np.linspace(lo, hi, round((hi-lo)/step)+1)
            z = rs.Z(grid)
            changes = int(np.count_nonzero(np.signbit(z[1:]) != np.signbit(z[:-1])))
            exact_zero_points = int(np.count_nonzero(z == 0))
            name = f"grid_{int(lo)}_{int(hi)}_{step}.csv"
            with (a.output / name).open("w", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(["t_float64","Z_corr3"])
                w.writerows((repr(float(t)),repr(float(v))) for t,v in zip(grid,z))
            grids.append(dict(step=step,points=len(grid),sign_changes=changes,
                exact_float_zero_points=exact_zero_points,endpoints=[float(z[0]),float(z[-1])],
                min_abs_Z=float(np.min(np.abs(z))),csv=name,
                csv_sha256=hashlib.sha256((a.output/name).read_bytes()).hexdigest()))
        counts.append(dict(window=[lo,hi],nzeros_lo=n0,nzeros_hi=n1,nzeros_difference=n1-n0,
            grids=grids,match=all(g["sign_changes"]==n1-n0 and g["exact_float_zero_points"]==0 for g in grids)))
        print("zero-count window", counts[-1], flush=True)
    report["zero_counts"] = counts
    report["all_counts_match"] = all(c["match"] for c in counts)
    report["limitations"] = ["Observed numerical errors, not outward enclosures",
        "mpmath is a numerical reference, not a formal proof",
        "Fixed grids cannot prove zero completeness",
        "Exact source implementation; not a second analytic method",
        "Additional validation, not confirmation of a statistical/exponent hypothesis",
        "Not full slice-C completion or scientific acceptance"]
    (a.output / "result.json").write_text(json.dumps(report,indent=2)+"\n")
    return 0 if report["all_counts_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
