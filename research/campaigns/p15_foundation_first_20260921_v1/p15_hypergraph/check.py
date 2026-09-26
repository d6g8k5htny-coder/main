#!/usr/bin/env python3
"""Authenticated, exact, bounded P15 finite replay. No source archive executes."""
import argparse
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

from finite import (antichain, colorable, complete_witnesses, cover_cost, family,
                    good, occupancy_cover, obstruction_cover, primitive_palette,
                    probability, require, require_complete_actual,
                    singleton_restriction, verify_composition,
                    verify_scalar_sandwich)

HERE = Path(__file__).resolve().parent
SOURCE_MANIFEST_SHA = "c2c74d00f3c8a79789c8587c2e6347adffedaaa5ee24cb70c339834b8a44b091"


def identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def source_guard(repo, artifact=HERE):
    """Exact source identities + fresh exclusion metadata. Never reads vault."""
    manifest_path = artifact / "SOURCES.json"
    require(identity(manifest_path)["sha256"] == SOURCE_MANIFEST_SHA,
            "source manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    for m in manifest["current_admission_metadata"]:
        require(identity(repo / m["path"]) == {k: m[k] for k in ("bytes", "sha256")},
                "admission metadata changed; re-review required")
    exclusions = json.loads((repo / "quarantine/EXCLUSIONS.json").read_text())["exclusions"]
    excluded = {e["carrier_id"] for e in exclusions}
    wanted = {s["drive_id"] for s in manifest["sources"]}
    rows = {}
    for line in (repo / "drive/inventory.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("id") in wanted:
            require(row["id"] not in rows, "duplicate source inventory identity")
            rows[row["id"]] = row
    observed = []
    for s in manifest["sources"]:
        require(s["drive_id"] not in excluded, "selected source excluded")
        row = rows.get(s["drive_id"])
        require(row is not None, "selected source absent from inventory")
        require(row["path"] == s["drive_path"] and row["path"].startswith("01_ACTIVE_RESEARCH_PACKAGES/"),
                "selected source lacks expected active path")
        expected = {k: s[k] for k in ("bytes", "sha256")}
        require({k: row[k] for k in expected} == expected, "source inventory identity differs")
        require(identity(repo / s["repository_path"]) == expected, "repository source identity differs")
        require(identity(artifact / s["copy_path"]) == expected, "copied source identity differs")
        observed.append({"drive_id": s["drive_id"], "name": s["name"], **expected})
    return observed


def rejects(operation, expected_text):
    try:
        operation()
    except ValueError as exc:
        require(expected_text in str(exc), "wrong rejection")
        return str(exc)
    raise ValueError("negative control was accepted")


def run_cases():
    blocks = {i: frozenset((2 * i, 2 * i + 1)) for i in range(3)}
    vertices = frozenset(range(6))
    local = {i: family([b]) for i, b in blocks.items()}
    macro = family([{0, 1, 2}])
    prices = {v: F(1, 2) for v in vertices}
    for i, b in blocks.items():
        verify_scalar_sandwich(b, local[i], {v: F(1, 2) for v in b}, F(1))
    reports = {}
    reports["rank3_complete_one_color_interface"] = verify_composition(
        blocks, local, macro, 1, local, {i: 0 for i in blocks}, macro, prices)
    reports["rank3_proper_macro_two_color_interface"] = verify_composition(
        blocks, local, macro, 1, local, {0: 0, 1: 0, 2: 1}, family([]), prices)
    actual = complete_witnesses(blocks, local, macro)
    mu_local = F(1)
    q = {}
    for i, b in blocks.items():
        mu_local *= probability(b, {v: prices[v] for v in b}, lambda s: good(s, local[i]))
        q[i] = 1 - probability(b, {v: prices[v] for v in b}, lambda s: not s)
    mu_global = probability(vertices, prices, lambda s: good(s, actual))
    mu_macro = probability(blocks, q, lambda s: good(s, macro))
    require(mu_global <= mu_local and mu_global <= mu_macro, "probability direction failed")
    require(mu_global != mu_local * mu_macro, "shared-event negative control lost dependence")
    reports["shared_events_not_independent"] = {
        "actual_good_probability": mu_global, "local_good_probability": mu_local,
        "macro_good_probability": mu_macro, "invalid_independence_product": mu_local * mu_macro}

    singleton_blocks = {i: frozenset([i]) for i in range(5)}
    universal_local = {i: family([]) for i in singleton_blocks}
    all_triples = family(combinations(range(5), 3))
    colors = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
    macro_cover = family([{0, 1, 2}])
    reports["nonempty_two_color_obstruction"] = verify_composition(
        singleton_blocks, universal_local, all_triples, 1, universal_local,
        colors, macro_cover, {i: F(1, 5) for i in range(5)})
    require(reports["nonempty_two_color_obstruction"]["global_obstructions"] == 1,
            "nontrivial obstruction fixture changed")
    zero = {i: F(0) for i in range(5)}
    reports["zero_probability_sets_retained"] = verify_composition(
        singleton_blocks, universal_local, all_triples, 1, universal_local,
        colors, macro_cover, zero)
    require(cover_cost(macro_cover, zero) == 0, "zero price failed")
    reports["zero_probability_sets_retained"]["deleting_zero_generators_rejected"] = rejects(
        lambda: obstruction_cover(range(5), all_triples, 2, family([])), "uncovered obstruction")

    capped = {v: F(3, 4) for v in vertices}
    reports["empty_occupancy_generator"] = verify_composition(
        blocks, local, macro, 1, local, {i: 0 for i in blocks}, macro, capped)
    require(reports["empty_occupancy_generator"]["lifted_cost"] == 1, "empty generator must cost one")
    require(cover_cost(family([set()]), capped) == 1 and cover_cost(family([]), capped) == 0,
            "empty generator/family distinction failed")
    require(cover_cost(occupancy_cover(blocks[0], prices), prices) == 1 > F(3, 4),
            "occupancy price must not be replaced by probability")
    reports["empty_occupancy_generator"]["refine_after_lift_halving_false"] = (F(1) > F(1, 2))

    # Positive global failure but zero macro failure: all original macro edges
    # remain zero-price setwise certificates, exactly the endpoint addendum.
    mixed = {v: F(0) if v in blocks[2] else F(1, 2) for v in vertices}
    mixed_q = {i: 1 - probability(b, {v: mixed[v] for v in b}, lambda s: not s)
               for i, b in blocks.items()}
    mixed_Q = 1 - probability(vertices, mixed, lambda s: good(s, actual))
    mixed_QH = 1 - probability(blocks, mixed_q, lambda s: good(s, macro))
    require(mixed_Q > 0 and mixed_QH == 0, "intermediate zero-Q fixture failed")
    reports["zero_macro_positive_global"] = verify_composition(
        blocks, local, macro, 1, local, {i: 0 for i in blocks}, macro, mixed)
    reports["zero_macro_positive_global"].update(global_failure=mixed_Q, macro_failure=mixed_QH)

    sb = {0: frozenset([0]), 1: frozenset([1, 2]), 2: frozenset([3, 4])}
    sl = {0: family([{0}]), 1: family([]), 2: family([])}
    removed, rb, rl, rm = singleton_restriction(sb, sl, macro)
    require(removed == frozenset([0]) and set(rb) == {1, 2} and not rm,
            "incident edge must be deleted, never shrunk")
    mutant = family([{1, 2}])
    require(good(frozenset([1, 2]), rm) and not good(frozenset([1, 2]), mutant),
            "shrunk-edge counterexample failed")
    p_single = {v: F(1, 3) for v in range(5)}
    mu_single = probability(range(5), p_single,
                            lambda s: good(s, complete_witnesses(sb, sl, macro)))
    require(mu_single == 1 - p_single[0], "singleton factor identity failed")
    reports["singleton_empty_block_deletion"] = {"removed": [0], "remaining_macro_edges": 0,
                                                "actual_good_probability": mu_single}

    # A macro-singleton forbids EVERY original coordinate of that block.
    mj, mb, ml, mm = singleton_restriction(blocks, {i: family([]) for i in blocks}, family([{0}]))
    require(mj == blocks[0] and set(mb) == {1, 2} and not mm, "macro singleton reduction failed")
    ej, eb, el, em = singleton_restriction({0: blocks[0]}, {0: family([])}, family([{0}]))
    require(ej == blocks[0] and not eb and not el and not em, "empty remainder endpoint failed")
    reports["macro_singleton_and_empty_remainder"] = {"original_singletons_removed": 2,
                                                     "empty_remainder_good_probability": F(1)}

    # Incomplete lifting: each block surely occupied, but one rare transversal
    # is the sole actual cross witness. Macro failure 1 is not <= actual 1/1000.
    rare = family([{0, 2, 4}])
    rare_p = {v: F(1, 10) if v % 2 == 0 else F(1) for v in vertices}
    rare_Q = 1 - probability(vertices, rare_p, lambda s: good(s, rare))
    require(rare_Q == F(1, 1000), "incomplete lift fixture failed")
    reports["incomplete_lift_rejected"] = {
        "actual_failure": rare_Q, "macro_failure": F(1),
        "rejection": rejects(lambda: require_complete_actual(
            blocks, {i: family([]) for i in blocks}, macro, rare), "complete-transversal")}

    pb = {0: frozenset([0, 1]), 1: frozenset([2, 3])}
    pl = {i: family([b]) for i, b in pb.items()}
    pm = family([{0, 1}])
    empty_covers = {i: family([]) for i in pb}
    reports["product_not_max_palette"] = verify_composition(
        pb, pl, pm, 2, empty_covers, {0: 0, 1: 1}, family([]), {v: F(1, 2) for v in range(4)})
    clique4 = complete_witnesses(pb, pl, pm)
    require(not colorable(range(4), clique4, 2) and colorable(range(4), clique4, 4),
            "product palette counterexample failed")
    reports["product_not_max_palette"]["max_palette_rejection"] = rejects(
        lambda: obstruction_cover(range(4), clique4, 2, family([])), "uncovered obstruction")

    empty_edge = family([set()])
    require(obstruction_cover(range(3), empty_edge, 4, empty_edge) == 7,
            "globally empty family requires universal cover")
    require(obstruction_cover(range(3), family([]), 1, family([])) == 0,
            "universally good family needs no cover")
    reports["empty_family_endpoints"] = {"globally_empty_downset_cost": F(1),
                                        "universal_downset_cost": F(0)}
    require(primitive_palette(2) == 768, "rank-two recurrence")
    require(primitive_palette(3) == 4706657280256, "rank-three recurrence")
    require(F(1, 2) + F(3761, 9216) == F(8369, 9216) < 1, "rank-three budget")
    reports["exact_palette_and_budget_arithmetic"] = {
        "general_H3_2": primitive_palette(3), "sharper_rank3_macro_palette": 2 * 368640,
        "local_budget_coefficient": F(1, 2), "macro_rank3_budget_coefficient": F(3761, 9216),
        "singleton_free_rank3_hazard_coefficient": F(8369, 9216)}
    return reports


def encode(value):
    if isinstance(value, F):
        return {"numerator": value.numerator, "denominator": value.denominator}
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    require(not output.exists(), "output must be fresh")
    require(repo != output and repo not in output.parents, "output cannot modify repository")
    require(HERE / "sources" not in output.parents and output.name not in
            {"PROOF.md", "SOURCES.json", "finite.py", "check.py", "test_finite.py"},
            "output cannot overwrite source material")
    inputs = {name: identity(HERE / name) for name in ("PROOF.md", "SOURCES.json", "finite.py", "check.py")}
    before = source_guard(repo)
    cases = run_cases()
    after = source_guard(repo)
    require(before == after, "source identities changed during run")
    require(inputs == {name: identity(HERE / name) for name in inputs}, "implementation changed during run")
    result = {"schema": "P15_FINITE_RANK_EXACT_FINITE_REPLAY_V1", "status": "PASS",
              "claim": "CLAIM-P15-FINITE-RANK-20260921-ROOT01", "inputs": inputs,
              "source_guard_before_after": before, "cases": cases,
              "case_count": len(cases), "original_prize_closed": False,
              "q0_changes": False, "scientific_status_changed": False,
              "organizational_independence_credit": 0,
              "analytic_source_primitives_proved_by_this_replay": False,
              "scope": "Exact finite examples and negative controls; analytic theorem and foundations require proof review."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, default=encode, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "case_count": len(cases), "output": str(output), **identity(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
