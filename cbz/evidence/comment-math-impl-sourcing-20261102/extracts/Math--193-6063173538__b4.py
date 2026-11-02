"""Readback verification only: no Gaussian fields, persistence or fits regenerated."""
import collections
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integral(a, b, alpha):
    q = 1-alpha
    return math.log(b/a) if q == 0 else a**q*math.expm1(q*math.log(b/a))/q


def main():
    sys.path.insert(0, str(ROOT.parent / "analytic-control/packages"))
    import numpy as np
    p = json.loads((ROOT/"PROTOCOL.json").read_text())
    f = json.loads((ROOT/"PROTOCOL_FREEZE.json").read_text())
    e = json.loads((ROOT/"EXECUTION_FREEZE.json").read_text())
    result = json.loads((ROOT/"RESULTS.json").read_text())
    rows = [json.loads(line) for line in (ROOT/"observations.jsonl").read_text().splitlines()]
    assert sha(ROOT/"PROTOCOL.json") == f["protocol_sha256"] == result["protocol_sha256"]
    assert sha(ROOT/"run_heldout.py") == e["driver_sha256"] == result["driver_sha256"]
    assert sha(ROOT/"observations.jsonl") == result["observations_sha256"]
    for b in p["input_bindings"]:
        assert len(Path(b["path"]).read_bytes()) == b["bytes"] and sha(b["path"]) == b["sha256"]
    assert all(r["recorded_at_utc"] >= e["frozen_at_utc"] > f["actual_frozen_at_utc"] for r in rows)
    assert len([r for r in rows if r["kind"] == "runtime"]) == 1
    assert not any(r["kind"] == "execution_failure" for r in rows)
    kinds = collections.Counter(r["kind"] for r in rows)
    ensemble = [r for r in rows if r["kind"] == "heldout_ensemble"]
    indexed = {(r["seed"], r["grid"], r["method"]): r for r in ensemble}
    assert len(indexed) == len(ensemble) == 384
    assert set(indexed) == set(itertools.product(p["seeds"], p["grids"], p["methods"]))
    bank_rows = [r for r in rows if r["kind"] == "ensemble_bank"]
    assert [r["seed"] for r in bank_rows] == p["seeds"]
    assert all(len(r["coefficient_modes"]) == 1201 for r in bank_rows)
    oracle_comparisons = 0
    lifetimes = 0
    for r in ensemble:
        life = [b-d for b, d in r["intervals"]]
        assert all(math.isfinite(b) and math.isfinite(d) and b > d for b, d in r["intervals"])
        assert len(r["essential"]) == 1 and math.isfinite(r["essential"][0])
        counts = [sum(a <= x < b for x in life) for a, b in zip(p["bins"], p["bins"][1:])]
        assert counts == r["counts"] and not r["failed_predicates"]
        lifetimes += len(life)
        assert r["grid_sha256"] == indexed[r["seed"], r["grid"], "cubical"]["grid_sha256"]
        if r["oracle"] is not None:
            assert r["oracle"]["intervals"] == r["intervals"]
            assert r["oracle"]["essential"] == r["essential"]
            oracle_comparisons += 1
    assert oracle_comparisons == 64
    cal = [r for r in rows if r["kind"] == "calibration_field"]
    assert [r["seed"] for r in cal] == p["calibration"]["seeds"]
    summary = next(r for r in rows if r["kind"] == "covariance_calibration")
    points = np.array([r["point_values"] for r in cal])
    products = points[:, :1]*points
    se = products.std(axis=0, ddof=1)/math.sqrt(512)
    z = (products.mean(axis=0)-summary["reference"])/se
    assert np.max(np.abs(z-summary["product_z"])) < 1e-12
    assert np.max(np.abs(z)) <= 4.5
    mean_z = points.mean(axis=0)/(points.std(axis=0, ddof=1)/math.sqrt(512))
    assert np.max(np.abs(mean_z-summary["mean_z"])) < 1e-12
    assert np.max(np.abs(mean_z)) <= 4.5
    analytic = [r for r in rows if r["kind"] == "analytic_fejer"]
    assert [r["grid"] for r in analytic] == p["grids"]
    for r in analytic:
        assert not r["failed_predicates"]
        assert r["engines"]["gudhi_h0"]["intervals"] == r["engines"]["oracle_h0"]["intervals"]
        assert r["engines"]["gudhi_h0"]["essential"] == r["engines"]["oracle_h0"]["essential"]
        for engine in r["engines"].values():
            assert len(engine["intervals"]) == 2 and len(engine["essential"]) == 1
            assert engine["optimal_two_bar_endpoint_error"] <= r["endpoint_allowance"]
            assert engine["essential_birth_error"] <= r["E_n"]+1e-12
    fits = [r for r in rows if r["kind"] == "free_alpha"]
    assert len(fits) == 12 and result["fits"] == fits
    failure = []
    for r in fits:
        expected = [indexed[seed, r["grid"], r["method"]]["counts"] for seed in p["seeds"]]
        pooled = [sum(c[j] for c in expected) for j in range(8)]
        assert pooled == r["pooled_counts"]
        checks = [(r["fit"], pooled, 32*576)]
        for deleted, seed, counts in zip(r["delete_one"], p["seeds"], expected):
            assert deleted["omitted_seed"] == seed
            checks.append((deleted["fit"], [x-y for x, y in zip(pooled, counts)], 31*576))
        for fit, counts, exposure in checks:
            predicted = [fit["A"]*exposure*integral(a, b, fit["alpha"])+fit["B"]*exposure*(b-a)
                         for a, b in zip(p["bins"], p["bins"][1:])]
            assert all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)
                       for x, y in zip(predicted, fit["predicted_counts"]))
            assert fit["A_positive"] == (fit["A"] > 0)
        if not r["fit"]["A_positive"]: failure.append(f"free_alpha:{r['grid']}/{r['method']}:A_positive")
        if not r["fit"]["density_nonnegative"]: failure.append(f"free_alpha:{r['grid']}/{r['method']}:density_nonnegative")
        if r["fit"]["boundary_minimum"]: failure.append(f"free_alpha:{r['grid']}/{r['method']}:profile_boundary")
        if r["grid"] == 512 and r["delete_one_alpha_range"][1]-r["delete_one_alpha_range"][0] > .2:
            failure.append(f"free_alpha:{r['grid']}/{r['method']}:delete_one_instability")
    fit_index = {(r["grid"], r["method"]): r for r in fits}
    for method in p["methods"]:
        for n, m in zip(p["grids"], p["grids"][1:]):
            a, b = fit_index[n, method], fit_index[m, method]
            if abs(a["fit"]["alpha"]-b["fit"]["alpha"]) > .1:
                failure.append(f"free_alpha:{method}:grid_{n}_{m}")
            for j, (x, y) in enumerate(zip(a["pooled_counts"], b["pooled_counts"])):
                if abs(x-y)/max(1, x, y) > .1: failure.append(f"count_convergence:{method}:{n}_{m}:bin{j}:relative")
                if abs(x-y)/32 > .25: failure.append(f"count_convergence:{method}:{n}_{m}:bin{j}:paired_mean")
    finest = [fit_index[512, m]["fit"]["alpha"] for m in p["methods"]]
    if max(finest)-min(finest) > .05: failure.append("free_alpha:finest_method_spread")
    for a, b in itertools.combinations(p["methods"], 2):
        for j, (x, y) in enumerate(zip(fit_index[512, a]["pooled_counts"], fit_index[512, b]["pooled_counts"])):
            if abs(x-y)/max(1, x, y) > .05: failure.append(f"count_convergence:finest:{a}_{b}:bin{j}")
    assert failure == result["failed_predicates"] and len(failure) == 113
    historical = json.loads((ROOT.parent/"free-alpha/RESULTS.json").read_text())
    assert len(historical["failed_diagnostic_predicates"]) == 14 and historical["diagnostic_verdict"] == "FAIL"
    assert result["complete_execution"] and result["scientific_decision"] == "INDETERMINATE"
    assert result["decisions"]["independently_blind_coefficient_test"] == "NOT_RUN/UNINTERPRETABLE"
    assert result["peak_rss_MiB"] <= 768 and result["elapsed_seconds"] < 1200
    out = dict(verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               actual_verifier="/root/i2_incidence_review", source_exposed=True, org_independence=0,
               compiler_calls=0, fresh_fields_regenerated=0, fits_rerun=0,
               protocol_sha256=sha(ROOT/"PROTOCOL.json"), driver_sha256=sha(ROOT/"run_heldout.py"),
               observations_sha256=sha(ROOT/"observations.jsonl"), results_sha256=sha(ROOT/"RESULTS.json"),
               input_bindings_matched=9, row_kinds=dict(kinds), ensemble_rows=384,
               fields=32, positive_lifetimes_rebinned=lifetimes, oracle_endpoint_comparisons=64,
               primary_fits_checked=12, delete_one_fits_checked=384,
               all_113_failure_labels_reproduced=True, original_14_failures_retained=True,
               verification_status="PASS", numerical_convergence="FAIL", scientific_decision="INDETERMINATE")
    (ROOT/"VERIFICATION.json").open("x").write(json.dumps(out, indent=2, sort_keys=True)+"\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

