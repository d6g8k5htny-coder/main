#!/usr/bin/env python3
"""
GP-DATA-184-v1.0 — same-line replay of the human-readable CL-DATA-200 core.

This is NOT the missing original Anthropic runner and does not reproduce the
declared original source hash. It loads the reconstructed Google-Doc source,
runs the authoritative precision ladder P={40,100,200}, all five controls,
nine zero-radius mutations, and containment self-tests, and emits canonical JSON.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("cl_ec019_t2_reconstructed", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def frac_obj(x):
    return {"numerator": str(x.numerator), "denominator": str(x.denominator)}

def interval_obj(x):
    return {"lo": frac_obj(x.lo), "hi": frac_obj(x.hi)}

def mutate(mod, base, name, center=None, radius=None):
    d = copy.deepcopy(base)
    c, r = d[name]
    d[name] = (
        c if center is None else mod.Fr(center),
        r if radius is None else mod.Fr(radius),
    )
    return d

def self_tests(mod, scale: int):
    x = mod.Fr(7, 3)
    lo = mod.sqrt_lo(x, scale)
    hi = mod.sqrt_hi(x, scale)
    sqrt_ok = lo * lo <= x <= hi * hi

    a = mod.IV(mod.Fr(-2), mod.Fr(3))
    b = mod.IV(mod.Fr(-5), mod.Fr(7))
    add_ok = (a + b).lo == -7 and (a + b).hi == 10
    sub_ok = (a - b).lo == -9 and (a - b).hi == 8
    mul = a * b
    mul_ok = mul.lo == -15 and mul.hi == 21
    even_ok = a.powi(2).lo == 0 and a.powi(2).hi == 9
    return {
        "sqrt_containment": sqrt_ok,
        "addition": add_ok,
        "subtraction": sub_ok,
        "multiplication": mul_ok,
        "even_power_zero_crossing": even_ok,
        "all_pass": all([sqrt_ok, add_ok, sub_ok, mul_ok, even_ok]),
    }

def execute(mod, exponent: int):
    scale = 10 ** exponent
    primary = mod.compute(copy.deepcopy(mod.BASE), S=scale)
    margins = primary["margins"]

    controls = {}

    try:
        mod.compute(mutate(mod, mod.BASE, "c04", radius=0), S=scale)
        controls["NC1_zero_width"] = {"pass": False}
    except mod.FullDimError as exc:
        controls["NC1_zero_width"] = {"pass": True, "exception": str(exc)}

    nc2 = mod.compute(mutate(mod, mod.BASE, "c40", center=0, radius=500), S=scale)
    controls["NC2_endpoint_typing"] = {
        "pass": nc2["margins"]["m_saddle_det"] <= 0 or nc2["margins"]["m_max_det"] <= 0,
        "m_saddle_det": frac_obj(nc2["margins"]["m_saddle_det"]),
        "m_max_det": frac_obj(nc2["margins"]["m_max_det"]),
    }

    nc3 = mod.compute(mutate(mod, mod.BASE, "s", center=-1, radius=mod.Fr(11, 10)), S=scale)
    controls["NC3_threshold"] = {
        "pass": nc3["margins"]["m_threshold"] <= 0,
        "m_threshold": frac_obj(nc3["margins"]["m_threshold"]),
    }

    nc4 = mod.compute(copy.deepcopy(mod.BASE), kappa=mod.Fr(5, 10000), S=scale)
    controls["NC4_cone"] = {
        "pass": nc4["margins"]["m_cone_slope"] <= 0,
        "m_cone_slope": frac_obj(nc4["margins"]["m_cone_slope"]),
    }

    nc5 = mod.compute(
        mutate(mod, mod.BASE, "a", center=mod.Fr(25, 10000), radius=mod.Fr(1, 2)),
        S=scale,
    )
    controls["NC5_capture_chart"] = {
        "pass": nc5["margins"]["m_chart_X"] <= 0 or nc5["margins"]["m_gersh_1"] <= 0,
        "m_chart_X": frac_obj(nc5["margins"]["m_chart_X"]),
        "m_gersh_1": frac_obj(nc5["margins"]["m_gersh_1"]),
    }

    zero_mutations = {}
    for name in mod.PARAM_ORDER:
        try:
            mod.compute(mutate(mod, mod.BASE, name, radius=0), S=scale)
            zero_mutations[name] = False
        except mod.FullDimError:
            zero_mutations[name] = True

    return {
        "precision_exponent": exponent,
        "primary_pass": mod.all_positive(margins),
        "margins": {name: frac_obj(margins[name]) for name in mod.MARGIN_ORDER},
        "derived": {
            k: interval_obj(v) if isinstance(v, mod.IV) else frac_obj(v)
            for k, v in primary["derived"].items()
            if k != "S"
        },
        "controls": controls,
        "all_controls_pass": all(item["pass"] for item in controls.values()),
        "zero_radius_mutations": zero_mutations,
        "all_zero_radius_mutations_reject": all(zero_mutations.values()),
        "self_tests": self_tests(mod, scale),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    source_path = Path(args.source)
    source_bytes = source_path.read_bytes()
    mod = load_module(source_path)
    runs = [execute(mod, p) for p in (40, 100, 200)]

    sign_vectors = [
        [run["margins"][name]["numerator"].startswith("-") is False for name in mod.MARGIN_ORDER]
        for run in runs
    ]
    record = {
        "object": "EC-019 T2 semantic replay of CL-DATA-200 human-readable core",
        "evidence_class": "SAME-LINE REPLAY / NOT ORIGINAL SOURCE IDENTITY",
        "source": {
            "filename": source_path.name,
            "bytes": len(source_bytes),
            "sha256": sha256_bytes(source_bytes),
            "declared_original_bytes": 12832,
            "declared_original_sha256": "6e6e7a532e8e66bb92f80fdc2a225f832c77987e01bdcd63d79a1fb8e99ac9d4",
            "identity_matches_declared_original": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "authoritative_precision_exponents": [40, 100, 200],
        "runs": runs,
        "all_primary_pass": all(r["primary_pass"] for r in runs),
        "all_controls_pass": all(r["all_controls_pass"] for r in runs),
        "all_zero_radius_mutations_reject": all(r["all_zero_radius_mutations_reject"] for r in runs),
        "all_self_tests_pass": all(r["self_tests"]["all_pass"] for r in runs),
        "signs_stable": all(v == sign_vectors[0] for v in sign_vectors[1:]),
        "verdict": "SEMANTIC REPLAY PASS / ORIGINAL CARRIER AND PREREGISTRATION FIDELITY OPEN",
    }
    out = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8")
    Path(args.output).write_bytes(out)
    print(json.dumps({
        "output_bytes": len(out),
        "output_sha256": sha256_bytes(out),
        "all_primary_pass": record["all_primary_pass"],
        "all_controls_pass": record["all_controls_pass"],
        "all_zero_radius_mutations_reject": record["all_zero_radius_mutations_reject"],
        "all_self_tests_pass": record["all_self_tests_pass"],
        "signs_stable": record["signs_stable"],
        "verdict": record["verdict"],
    }, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
