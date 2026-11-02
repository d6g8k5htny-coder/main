"""Prospective held-out finite float64 diagnostics; no scientific promotion."""
import datetime
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parent
PROTOCOL_SHA = "0283222a1dbe8b4dda691a339550b731758ba1af1a04832eef3369f60180ac1f"
START = time.monotonic()
DEADLINE = datetime.datetime(2026, 10, 8, 15, 28, tzinfo=datetime.timezone.utc)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def record(kind, **kwargs):
    row = dict(kind=kind, recorded_at_utc=now(), **kwargs)
    with (ROOT / "observations.jsonl").open("a") as out:
        out.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return row


def budget():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    if datetime.datetime.now(datetime.timezone.utc) >= DEADLINE:
        raise RuntimeError("Fixed UTC computational stop reached")
    if time.monotonic() - START > 1200:
        raise RuntimeError("Fixed 1200-second wall budget reached")
    if rss > 768:
        raise RuntimeError("Fixed 768-MiB observed peak RSS limit exceeded")


def same(a, b, tol=1e-12):
    return (len(a["intervals"]) == len(b["intervals"])
            and len(a["essential"]) == len(b["essential"])
            and all(abs(x-y) <= tol for p, q in zip(a["intervals"], b["intervals"])
                    for x, y in zip(p, q))
            and all(abs(x-y) <= tol for x, y in zip(a["essential"], b["essential"])))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    assert digest(ROOT / "PROTOCOL.json") == PROTOCOL_SHA
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    for bound in p["input_bindings"]:
        path = Path(bound["path"])
        assert len(path.read_bytes()) == bound["bytes"] and digest(path) == bound["sha256"]
    assert not (ROOT / "observations.jsonl").exists()
    assert not (ROOT / "RESULTS.json").exists()
    sys.path.insert(0, str(ROOT.parent / "analytic-control/packages"))
    import numpy as np
    import gudhi
    assert np.__version__ == "2.5.3" and gudhi.__version__ == "3.13.0"
    source = Path(p["input_bindings"][0]["path"])
    sys.path.insert(0, str(source.parent))
    import experiment as ex
    import refinement as rf
    fit_source = p["input_bindings"][3]["path"]
    fa = load_module("bound_free_alpha", fit_source)
    runtime = record("runtime", protocol_sha256=PROTOCOL_SHA,
                     driver_sha256=digest(__file__), python=sys.version,
                     executable=sys.executable, numpy=np.__version__, gudhi=gudhi.__version__,
                     platform=platform.platform(), argv=sys.argv,
                     thread_environment={k:os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                     process_nice=os.getpriority(os.PRIO_PROCESS, 0))
    failures = []
    analytic = []
    ensemble = []
    fits = []
    stage = "sampler_basis"
    completed = False
    try:
        # A new cutoff24/lag control, not a replay of old cutoff12 controls.
        cfg = p["sampler_basis"]
        lags = cfg["index_lags"]
        actual = np.zeros(len(lags))
        modes = [(0, 0)] + [(x, y) for x in range(25) for y in range(-24, 25)
                            if x > 0 or (x == 0 and y > 0)]
        responses = 0
        for mode in modes:
            for component in range(1 if mode == (0, 0) else 2):
                bank = {(0, 0): (0.0, 0.0)}
                bank[mode] = (1.0, 0.0) if component == 0 else (0.0, 1.0)
                values = ex.field_grid(bank, 64, 24)
                actual += values[0, 0] * np.array([values[x, y] for x, y in lags])
                responses += 1
            budget()
        reference = np.array([ex.reference_covariance(24, (x*24/64, y*24/64))
                              for x, y in lags])
        error = float(np.max(np.abs(actual-reference)))
        if error > cfg["max_absolute_covariance_error"]:
            failures.append("sampler_basis:covariance_error")
        basis = record(stage, responses=responses, actual_covariance=actual.tolist(),
                       reference_covariance=reference.tolist(), max_error=error,
                       passed=error <= cfg["max_absolute_covariance_error"],
                       scope="Exact model covariance identity evaluated in float64, no full-spectrum certificate")
        print(json.dumps(dict(stage=stage, max_error=error, responses=responses)), flush=True)
        stage = "covariance_calibration"
        points = []
        for seed in p["calibration"]["seeds"]:
            budget()
            bank = ex.mode_bank(seed, 24)
            values = ex.field_grid(bank, 64, 24)
            sample = [float(values[x, y]) for x, y in lags]
            points.append(sample)
            record("calibration_field", seed=seed, point_values=sample,
                   grid_sha256=hashlib.sha256(values.tobytes()).hexdigest())
        points = np.asarray(points)
        products = points[:, :1] * points
        means = points.mean(axis=0)
        mean_se = points.std(axis=0, ddof=1) / math.sqrt(len(points))
        product_means = products.mean(axis=0)
        product_se = products.std(axis=0, ddof=1) / math.sqrt(len(points))
        mean_z = means / mean_se
        product_z = (product_means-reference) / product_se
        for j, z in enumerate(mean_z):
            if abs(z) > p["calibration"]["fixed_max_absolute_studentized_mean_deviation"]:
                failures.append(f"covariance_calibration:mean_{j}")
        for j, z in enumerate(product_z):
            if abs(z) > p["calibration"]["fixed_max_absolute_studentized_product_deviation"]:
                failures.append(f"covariance_calibration:product_{j}")
        calibration = record(stage, fields=len(points), reference=reference.tolist(),
                             mean_values=means.tolist(), se_values=mean_se.tolist(),
                             mean_z=mean_z.tolist(), product_means=product_means.tolist(),
                             product_se=product_se.tolist(), product_z=product_z.tolist(),
                             scope=p["calibration"]["claim"])
        print(json.dumps(dict(stage=stage, fields=len(points), product_z=product_z.tolist())), flush=True)
        stage = "analytic_fejer"
        tx, ty = math.sqrt(2)/7, math.sqrt(3)/11
        truth = [(1.0021, -0.9969), (1.0041, -0.9989)]
        for n in p["analytic_control"]["grids"]:
            budget()
            x = 2*np.pi*np.arange(n)/n-tx
            y = 2*np.pi*np.arange(n)/n-ty
            g = np.cos(3*x)
            for j, v in enumerate(p["analytic_control"]["v"]):
                hj = np.full(n, 1/6)
                for r in range(1, 6):
                    hj += (6-r)/18*np.cos(r*(x-j*np.pi/3))
                g += v*hj/1000
            values = g[:, None]+np.cos(y)[None, :]/10000
            E = (9+119/1200+1/10000)*math.pi**2/(2*n*n)
            out = {}
            failed = []
            for name in p["analytic_control"]["engines"]:
                tick = time.monotonic()
                result = getattr(ex, name)(values)
                result["elapsed_seconds"] = time.monotonic()-tick
                if len(result["intervals"]) != 2:
                    failed.append(name+":finite_count")
                    match_error = None
                else:
                    match_error = min(max(abs(a-b) for pair, target in zip(order, truth)
                                          for a, b in zip(pair, target))
                                      for order in itertools.permutations(result["intervals"]))
                    if match_error > 2*E+1e-12:
                        failed.append(name+":continuum_endpoint_allowance")
                if len(result["essential"]) != 1:
                    failed.append(name+":essential_count")
                    essential_error = None
                else:
                    essential_error = abs(result["essential"][0]-1.0061)
                    if essential_error > E+1e-12:
                        failed.append(name+":essential_birth_allowance")
                result.update(optimal_two_bar_endpoint_error=match_error,
                              essential_birth_error=essential_error)
                out[name] = result
            if not same(out["gudhi_h0"], out["oracle_h0"]):
                failed.append("engine_agreement")
            row = record(stage, grid=n, shift=[tx, ty], E_n=E, endpoint_allowance=2*E+1e-12,
                         grid_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                         engines=out, failed_predicates=failed)
            analytic.append(row)
            failures.extend(f"analytic_fejer:{n}:{x}" for x in failed)
            print(json.dumps(dict(stage=stage, grid=n, failed=failed)), flush=True)
            del values
            gc.collect()
        stage = "heldout_ensemble"
        for seed in p["seeds"]:
            budget()
            bank = ex.mode_bank(seed, p["cutoff"])
            record("ensemble_bank", seed=seed, coefficient_modes=[
                [x, y, a, b] for (x, y), (a, b) in bank.items()])
            hessian = rf.hessian_diagnostic(bank, p["cutoff"])
            for n in p["grids"]:
                budget()
                values = ex.field_grid(bank, n, p["cutoff"])
                grid_hash = hashlib.sha256(values.tobytes()).hexdigest()
                epsilon = rf.interpolation_diagnostic(hessian, n)
                for method in p["methods"]:
                    budget()
                    tick = time.monotonic()
                    result = ex.gudhi_h0(values) if method == "cubical" else rf.triangulated_h0(values, method)
                    elapsed = time.monotonic()-tick
                    life = [b-d for b, d in result["intervals"]]
                    assert all(math.isfinite(x) and x > 0 for x in life)
                    failed = []
                    if len(result["essential"]) != 1:
                        failed.append("essential_count")
                    if method == "cubical" and n in p["random_engine_controls"]["oracle_grids"]:
                        oracle = ex.oracle_h0(values)
                        if not same(result, oracle):
                            failed.append("oracle_endpoint_agreement")
                    else:
                        oracle = None
                    row = record(stage, seed=seed, grid=n, method=method,
                                 grid_sha256=grid_hash, intervals=result["intervals"],
                                 essential=result["essential"], zero_count=result["zero_count"],
                                 counts=ex.bin_counts(life, p["bins"]), elapsed_seconds=elapsed,
                                 oracle=oracle, failed_predicates=failed,
                                 floating_hessian_majorant=hessian, floating_diagram_budget=epsilon,
                                 bin_lower_endpoints_exceed_twice_budget=[a > 2*epsilon for a in p["bins"][:-1]],
                                 approximation_scope="Floating finite-polynomial diagnostic, not certified samples/matching/count sandwich")
                    ensemble.append(row)
                    failures.extend(f"heldout_ensemble:{seed}:{n}:{method}:{x}" for x in failed)
                    gc.collect()
                del values
            print(json.dumps(dict(stage=stage, completed_seed=seed, rows=len(ensemble),
                                  peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)), flush=True)
        stage = "free_alpha"
        indexed = {(r["seed"], r["grid"], r["method"]): r for r in ensemble}
        assert len(indexed) == len(p["seeds"])*len(p["grids"])*len(p["methods"])
        for n, method in itertools.product(p["grids"], p["methods"]):
            budget()
            counts = [indexed[(seed, n, method)]["counts"] for seed in p["seeds"]]
            pooled = [sum(xs[j] for xs in counts) for j in range(8)]
            primary = fa.fit(p["bins"], pooled, len(counts)*24**2)
            deletes = []
            for seed, row_counts in zip(p["seeds"], counts):
                deletes.append(dict(omitted_seed=seed, fit=fa.fit(p["bins"],
                    [c-x for c, x in zip(pooled, row_counts)], (len(counts)-1)*24**2)))
            alphas = [r["fit"]["alpha"] for r in deletes]
            failed = []
            if not primary["A_positive"]: failed.append("A_positive")
            if not primary["density_nonnegative"]: failed.append("density_nonnegative")
            if primary["boundary_minimum"]: failed.append("profile_boundary")
            if n == 512 and max(alphas)-min(alphas) > .2: failed.append("delete_one_instability")
            failures.extend(f"free_alpha:{n}/{method}:{x}" for x in failed)
            row = record(stage, grid=n, method=method, pooled_counts=pooled, fit=primary,
                         delete_one=deletes, delete_one_alpha_range=[min(alphas), max(alphas)],
                         failed_predicates=failed)
            fits.append(row)
        print(json.dumps(dict(stage=stage, primary_fits=len(fits), delete_one_fits=32*len(fits))), flush=True)
        stage = "diagnostic_decisions"
        fit_index = {(r["grid"], r["method"]): r for r in fits}
        convergence = []
        for method in p["methods"]:
            for n, m in zip(p["grids"], p["grids"][1:]):
                left, right = fit_index[n, method], fit_index[m, method]
                delta = abs(left["fit"]["alpha"]-right["fit"]["alpha"])
                if delta > .1: failures.append(f"free_alpha:{method}:grid_{n}_{m}")
                rel = [abs(a-b)/max(1, a, b) for a, b in zip(left["pooled_counts"], right["pooled_counts"])]
                paired = [(right["pooled_counts"][j]-left["pooled_counts"][j])/32 for j in range(8)]
                for j, (r, d) in enumerate(zip(rel, paired)):
                    if r > .1: failures.append(f"count_convergence:{method}:{n}_{m}:bin{j}:relative")
                    if abs(d) > .25: failures.append(f"count_convergence:{method}:{n}_{m}:bin{j}:paired_mean")
                convergence.append(dict(method=method, source_grid=n, target_grid=m,
                                        alpha_difference=delta, relative_count_difference=rel,
                                        paired_mean_count_difference=paired))
        finest_alpha = [fit_index[512, method]["fit"]["alpha"] for method in p["methods"]]
        spread = max(finest_alpha)-min(finest_alpha)
        if spread > .05: failures.append("free_alpha:finest_method_spread")
        method_comparisons = []
        eligibility_failed = ["certified_approximation_MISSING", "remainder_window_MISSING", "blind_coefficient_enclosure_MISSING"]
        for a, b in itertools.combinations(p["methods"], 2):
            ca, cb = fit_index[512, a]["pooled_counts"], fit_index[512, b]["pooled_counts"]
            rel = [abs(x-y)/max(1, x, y) for x, y in zip(ca, cb)]
            for j, r in enumerate(rel):
                if r > .05: failures.append(f"count_convergence:finest:{a}_{b}:bin{j}")
            method_comparisons.append(dict(methods=[a, b], relative_count_difference=rel))
        for method in p["methods"]:
            for j, c in enumerate(fit_index[512, method]["pooled_counts"]):
                if c < 30: eligibility_failed.append(f"count_power:{method}:bin{j}:count{c}")
        decisions = record(stage, adjacent=convergence, finest_method_alpha_spread=spread,
                           finest_method_comparisons=method_comparisons,
                           interpretation_failed_obligations=eligibility_failed,
                           independently_blind_coefficient_test="NOT_RUN/UNINTERPRETABLE",
                           legacy_coefficient_diagnostic="SKIPPED: does not replace actual frozen blind prediction",
                           scientific_decision="INDETERMINATE", failed_predicates=failures)
        completed = True
    except Exception as error:
        record("execution_failure", stage=stage, error=repr(error))
        failures.append("execution_failure:"+stage+":"+repr(error))
        decisions = {"scientific_decision":"INDETERMINATE", "incomplete_stage":stage,
                     "independently_blind_coefficient_test":"NOT_RUN/UNINTERPRETABLE"}
    result = dict(task_id=p["task_id"], actual_assigner=p["actual_assigner"], actual_executor=p["actual_executor"],
                  protocol_sha256=PROTOCOL_SHA, driver_sha256=digest(__file__),
                  observations_sha256=digest(ROOT/"observations.jsonl"), completed_at_utc=now(),
                  runtime=runtime, elapsed_seconds=time.monotonic()-START,
                  peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                  complete_execution=completed, analytic_rows=analytic,
                  ensemble_rows=len(ensemble), independent_ensemble_fields=len({r["seed"] for r in ensemble}),
                  fits=fits, decisions=decisions, failed_predicates=failures,
                  scientific_effect="NONE", scientific_decision="INDETERMINATE",
                  org_independence=0, blind_credit=0, compiler_calls=0)
    (ROOT/"RESULTS.json").open("x").write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(dict(complete_execution=completed, failed_predicates=len(failures),
                          ensemble_rows=len(ensemble), result_sha256=digest(ROOT/"RESULTS.json"),
                          scientific_decision="INDETERMINATE", elapsed_seconds=result["elapsed_seconds"])), flush=True)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())

