#!/usr/bin/env python3
"""Slice C driver: run the README section 8 command list of experiments/zeta_fold_exploration
in a detached worktree, capture every stdout, and write a JSON-lines ledger.

Env: ZF_PKT (packet dir in the worktree), ZF_PY (pinned interpreter), ZF_LOG (ledger/log dir),
ZF_LANE (dbn | stats). Scripts write their own outputs in place; where the docstring says
"stdout saved as X" the captured stdout is copied to X unless the script itself rewrote X.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PKT = Path(os.environ["ZF_PKT"]).resolve()
ROOT = PKT.parent.parent
PY = os.environ["ZF_PY"]
LOG = Path(os.environ["ZF_LOG"]).resolve()
LANE = os.environ["ZF_LANE"]
LOG.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PYTHONDONTWRITEBYTECODE="1")
LEDGER = LOG / f"ledger_{LANE}.jsonl"
JOB_TIMEOUT = 3 * 3600


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def job(name, argv, mapped=None, cwd=None, after=()):
    return dict(name=name, argv=argv, mapped=mapped, cwd=str(cwd or PKT), after=list(after))


def P(*a):
    return [PY, "-B", *a]


DBN = [
    job("test_numeric", P("-m", "unittest", "-v", "test_exact", "test_numeric")),
    job("validate_H0", P("dbn/validate_H0.py"), "results/dbn/validate_H0.out"),
    *[job(f"count_{a}_{b}_{t}", P("dbn/count_check.py", a, b, t), f"results/dbn/count_{a}_{b}_{t}.out")
      for a, b, t in [("1", "103", "-1.87"), ("1", "103", "-1.89"), ("1", "103", "0"),
                      ("1", "270", "-0.38"), ("1", "270", "-0.39"), ("1", "270", "0")]],
    job("check_dbn_basic", P("dbn/check_dbn_basic.py"), "results/dbn/check_dbn_basic.out"),
    job("check_collision", P("dbn/check_collision.py"), "results/dbn/check_collision.out"),
    job("check_collision_persistence", P("dbn/check_collision_persistence.py"),
        "results/dbn/check_collision_persistence.out", after=["check_collision"]),
    job("refine_check", P("dbn/refine_check.py"), "results/dbn/refine_check.out"),
    job("indep_collision", P("dbn/indep_collision.py"), "results/dbn/indep_collision.out"),
    job("kappa_check", P("dbn/kappa_check.py"), "results/dbn/kappa_check.out"),
    job("kappa_indep_degree6", P("dbn/kappa_indep_degree6.py"), "results/dbn/kappa_indep_degree6.out"),
    job("kappa_indep_random", P("dbn/kappa_indep_random.py"), "results/dbn/kappa_indep_random.out"),
    job("zero_velocity", P("dbn/zero_velocity.py"), "results/dbn/zero_velocity.out"),
    job("smoke_hflow", P("dbn/smoke_hflow.py"), "results/dbn/smoke_hflow.out"),
    job("indep_ht_xi", P("dbn/indep_ht.py", "60", "xi"), "results/dbn/indep_ht_xi.out"),
    job("indep_ht_c910", P("dbn/indep_ht.py", "60", "c910"), "results/dbn/indep_ht_c910.out"),
    job("indep_ht_c3435", P("dbn/indep_ht.py", "60", "c3435"), "results/dbn/indep_ht_c3435.out"),
    job("lobe_identity", P("dbn/lobe_identity.py"), "results/dbn/lobe_identity.out"),
    job("track_z270", P("dbn/track.py", "270"), "results/dbn/track_z270.out"),
    job("track_low10", P("dbn/track.py", "103", "-40", "0.02", "low10"), "results/dbn/track_low10.out"),
    *[job(f"track_dt{dt}", P("dbn/track.py", "103", "-40", dt, f"dt{dt}"))
      for dt in ["0.01", "0.02", "0.025", "0.04", "0.05"]],
    job("collision_table", P("dbn/collision_table.py"), after=["track_z270", "track_low10"]),
    job("fold_pair34_35", P("dbn/fold_check.py", "-0.3845611166", "223.2757801856", "pair34_35"),
        "results/dbn/fold_pair34_35.out", after=["collision_table"]),
    job("fold_pair9_10", P("dbn/fold_check.py", "-1.8779090730", "99.3284693056", "pair9_10"),
        "results/dbn/fold_pair9_10.out", after=["collision_table"]),
    job("fold_analyze", P("dbn/fold_analyze.py", "pair34_35", "pair9_10"), "results/dbn/fold_analyze.out",
        after=["fold_pair34_35", "fold_pair9_10"]),
    job("check_theta", P("theta/check_theta.py"), "results/theta/check_theta.out"),
    job("project_theta_checks", P("theta/project_theta_checks.py"), "results/theta/project_theta_checks.out"),
    job("check_gue_transfer", P("stats/check_gue_transfer.py"), "results/stats/check_gue_transfer.out"),
    job("fold_controls", P("experiments/universality/fold_controls.py"),
        "results/theta/fold_controls_rerun.out", cwd=ROOT),
]

STATS_SEQ = [
    job("rs_validate", P("stats/rs_zeta.py", "validate")),
    job("rs_A", P("stats/rs_zeta.py", "A", "1e4", "1e5")),
    job("rs_B", P("stats/rs_zeta.py", "B", "1e5", "1e6")),
    job("gauss", P("stats/gauss_proc.py", "500", "4")),
    job("cue12", P("stats/cue.py", "12", "86016", "4")),
    job("cue128", P("stats/cue.py", "128", "8064", "4")),
    job("analyze", P("stats/analyze.py")),
    job("matched_tail", P("stats/matched_tail.py")),
    job("make_plots", P("stats/make_plots.py")),
    job("sample_600", P("stats/sample_Z.py", "600", "900", "0.02"), "results/stats/sample_600.log"),
    job("sample_5000", P("stats/sample_Z.py", "5000", "5200", "0.015"), "results/stats/sample_5000.log"),
    job("checkZ_600", P("stats/check_Z_persistence.py", "Z_grid_600_900.npz"),
        "results/stats/Z_grid_600_900_persistence.out"),
    job("checkZ_5000", P("stats/check_Z_persistence.py", "Z_grid_5000_5200.npz"),
        "results/stats/Z_grid_5000_5200_persistence.out"),
]
STATS_CROSS = [
    job("check_gauss", P("stats/check_gauss.py")),
    job("check_zeta_quad", P("stats/check_zeta_quad.py")),
    job("barcode_bruteforce", P("stats/barcode_bruteforce.py"), "results/stats/barcode_bruteforce.out"),
    job("gauss_hidden", P("stats/gauss_hidden.py"), "results/stats/gauss_hidden.out"),
    job("localamp_baseline", P("stats/localamp_baseline.py"), "results/stats/localamp_baseline.out"),
    job("cue_gap_shape", P("stats/cue_gap_shape.py"), "results/stats/cue_gap_shape.out"),
    job("rs_spot_check", P("stats/rs_spot_check.py"), "results/stats/rs_spot_check.out"),
    job("cue_endpoint_check", P("stats/cue_endpoint_check.py"), "results/stats/cue_endpoint_check.out"),
    job("block_sensitivity", P("stats/block_sensitivity.py")),
]


def run(j):
    mapped = (PKT / j["mapped"]) if j["mapped"] else None
    before = mapped.stat().st_mtime_ns if (mapped and mapped.exists()) else None
    log = LOG / f"{j['name']}.stdout"
    err = LOG / f"{j['name']}.stderr"
    t0 = time.time()
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))
    with open(log, "wb") as fo, open(err, "wb") as fe:
        try:
            rc = subprocess.run(j["argv"], cwd=j["cwd"], env=ENV, stdout=fo, stderr=fe,
                                timeout=JOB_TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
    wall = time.time() - t0
    mode = None
    if mapped is not None:
        after = mapped.stat().st_mtime_ns if mapped.exists() else None
        if after is not None and after != before:
            mode = "self-written"
        elif rc == 0:
            mapped.parent.mkdir(parents=True, exist_ok=True)
            mapped.write_bytes(log.read_bytes())
            mode = "stdout-copied"
        else:
            mode = "not-copied(rc!=0)"
    rec = dict(lane=LANE, name=j["name"], argv=j["argv"], cwd=j["cwd"], start=start, wall_s=round(wall, 1),
               rc=rc, stdout=str(log), stdout_sha256=sha256(log), stderr_bytes=err.stat().st_size,
               mapped=j["mapped"], mode=mode)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{LANE}] {j['name']}: rc={rc} wall={wall:.1f}s mode={mode}", flush=True)
    return rec


def run_pool(jobs, width):
    done = {}
    pending = {j["name"]: j for j in jobs}
    with ThreadPoolExecutor(max_workers=width) as ex:
        futs = {}
        while pending or futs:
            ready = [n for n, j in pending.items() if all(a in done for a in j["after"])]
            for n in ready:
                futs[ex.submit(run, pending.pop(n))] = n
            if not futs:
                raise RuntimeError(f"dependency cycle or missing job: {list(pending)}")
            for f in list(futs):
                if f.done():
                    rec = f.result()
                    done[futs.pop(f)] = rec
            time.sleep(0.5)
    return done


def main():
    if "--list" in sys.argv:
        for j in (DBN if LANE == "dbn" else STATS_SEQ + STATS_CROSS):
            print(j["name"], " ".join(j["argv"][1:]), "->", j["mapped"] or "(own outputs)")
        return 0
    LEDGER.write_text("")
    if LANE == "dbn":
        run_pool(DBN, 3)
    elif LANE == "stats":
        for j in STATS_SEQ:
            run(j)
        run_pool(STATS_CROSS, 3)
    else:
        raise SystemExit(f"unknown lane {LANE}")
    print(f"[{LANE}] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
