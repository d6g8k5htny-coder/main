#!/usr/bin/env python3
"""Task 2 focused unittest suite; source-exposed, org/blind0.

The original 24 tests were authored from the contract and plan before source
implementation. Additional regressions and diagnostic budget retention were
authored by the implementer after actual self-review findings. Use Python 3.12.
No old population module, CDF, pairing oracle, or scientific driver is consumed.
Each mathematical prediction is asserted only after select_h0 and verify_h0 pass.
"""

import copy
from fractions import Fraction as Q
import time
import unittest
from unittest.mock import patch

from surgery_field import Budget
import global_h0_selector as selector


ROOT_KEYS = {"factor", "ordinal", "isolator", "exact_value", "multiplicity", "branch_splits"}
ROOT_SET_KEYS = {"status", "polynomial", "domain", "factorization", "sturm_chains", "roots", "checks", "work"}
SELECTION_KEYS = {"schema_version", "model_id", "u", "v", "status", "guard", "critical_roots", "old_criticals", "critical_heights", "height_groups", "levels", "incidence", "events", "actual_barcode", "population_projection", "checks", "work"}
WORK_KEYS = {"polynomial_evaluations", "gcd_steps", "polynomial_divisions", "max_branch_splits", "field_calls", "primitive_requests"}
U0 = Q(3, 2**32)
V0 = Q(1, 2**47)
A = Q(1, 2**18)
BUDGET_OBSERVATIONS = []  # Test diagnostics only; outside recorder consumes these.


def evaluate(polynomial, point):
    """Evaluate only the supplied polynomial witness; never select roots or bars."""
    return sum((coefficient * point**power for power, coefficient in enumerate(polynomial)), Q(0))


def multiply(left, right):
    """Replay a factorization identity; no factor finding or root selection."""
    product = [Q(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            product[i + j] += a * b
    return tuple(product)


def variations(values):
    signs = [1 if value > 0 else -1 for value in values if value != 0]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def roots_in(record):
    """Controller ruling: critical_roots is the complete Root-set dictionary."""
    return record["roots"]


class ContractCase(unittest.TestCase):
    def budget(self, **limits):
        budget = Budget(time.monotonic(), **limits)
        budget.begin_fixture(self.id())
        BUDGET_OBSERVATIONS.append((self.id(), budget))
        return budget

    def sound_selection(self, u, v):
        budget = self.budget()
        certificate = selector.select_h0(u, v, budget=budget)
        self.assertIsInstance(certificate, dict)
        self.assertTrue(SELECTION_KEYS <= certificate.keys())
        self.assertEqual(certificate["status"], "CERTIFIED")
        verification = selector.verify_h0(u, v, certificate, budget=budget)
        self.assertEqual(verification["status"], "PASS")
        self.assertTrue({"status", "checks", "work"} <= verification.keys())
        self.assertTrue(WORK_KEYS <= verification["work"].keys())
        self.assertEqual(certificate["schema_version"], 1)
        self.assertEqual(certificate["model_id"], "explicit_periodic_cusp_v1")
        self.assertEqual((certificate["u"], certificate["v"]), (u, v))
        self.assertEqual(certificate["actual_barcode"]["status"], "CERTIFIED")
        self.assertEqual(certificate["actual_barcode"]["tie_policy"], "smallest_axial_root_ordinal_survives")
        essential = certificate["actual_barcode"]["essential"]
        self.assertIsInstance(essential, list)
        self.assertEqual(len(essential), 1)
        self.assertTrue({"birth_root", "birth_centered_interval", "tied_birth"} <= essential[0].keys())
        return certificate, budget

    def ordered_roots(self, certificate):
        roots = roots_in(certificate["critical_roots"])
        ordered = sorted(roots, key=lambda root: root["ordinal"])
        # Controller ruling: the sorted root labels are zero-based and contiguous.
        self.assertEqual([root["ordinal"] for root in ordered], list(range(len(ordered))))
        return ordered

    def contains(self, interval, exact):
        self.assertIsInstance(interval, tuple)
        self.assertEqual(len(interval), 2)
        self.assertTrue(all(type(endpoint) is Q for endpoint in interval))
        self.assertLessEqual(interval[0], exact)
        self.assertLessEqual(exact, interval[1])

    def rejected_certificate(self, u, v, certificate):
        # Fresh verifier fixture has the full ceiling; malformed input must fail
        # substantive verification, not succeed by exhausting a prior budget.
        result = selector.verify_h0(u, v, certificate, budget=self.budget())
        self.assertEqual(result["status"], "FAIL_IMPLEMENTATION")
        self.assertTrue(result["checks"])
        self.assertTrue({"status", "checks", "work"} <= result.keys())


class ExactRootTests(ContractCase):
    def check_root_set(self, result, p, domain):
        self.assertTrue(ROOT_SET_KEYS <= result.keys())
        self.assertEqual(result["status"], "CERTIFIED")
        self.assertEqual(result["polynomial"], p)
        self.assertEqual(result["domain"], domain)
        # Controller ruling: preserve the scalar unit lost by monic factors.
        self.assertIs(type(result["scalar_unit"]), Q)
        rebuilt = (result["scalar_unit"],)
        for factor in result["factorization"]:
            self.assertIs(type(factor["multiplicity"]), int)
            self.assertGreater(factor["multiplicity"], 0)
            for _ in range(factor["multiplicity"]):
                rebuilt = multiply(rebuilt, factor["factor"])
        self.assertEqual(rebuilt, p)
        self.assertTrue(WORK_KEYS <= result["work"].keys())
        self.assertEqual([root["ordinal"] for root in result["roots"]], list(range(len(result["roots"]))))
        for root in result["roots"]:
            self.assertTrue(ROOT_KEYS <= root.keys())
            self.assertIs(type(root["ordinal"]), int)
            self.assertIs(type(root["multiplicity"]), int)
            self.assertGreater(root["multiplicity"], 0)
            self.assertLessEqual(root["branch_splits"], 256)
        for chain in result["sturm_chains"]:
            self.assertTrue({"factor", "sequence", "left", "right", "left_variations", "right_variations", "root_count"} <= chain.keys())
            # Exact endpoint roots must have been extracted before this count.
            self.assertNotEqual(evaluate(chain["factor"], chain["left"]), 0)
            self.assertNotEqual(evaluate(chain["factor"], chain["right"]), 0)
            left = variations([evaluate(p, chain["left"]) for p in chain["sequence"]])
            right = variations([evaluate(p, chain["right"]) for p in chain["sequence"]])
            self.assertEqual(chain["left_variations"], left)
            self.assertEqual(chain["right_variations"], right)
            self.assertEqual(chain["root_count"], left - right)

    def test_repeated_endpoint_and_midpoint_roots_are_retained_once(self):
        # Break caught: duplicate midpoint roots or lost original gcd multiplicity.
        # x^3 (x+1)^2 (x-1), with both domain endpoints and its midpoint roots.
        p = (Q(0), Q(0), Q(0), Q(-1), Q(-1), Q(1), Q(1))
        domain = (Q(-1), Q(1))
        result = selector.isolate_roots(p, domain, budget=self.budget())
        self.check_root_set(result, p, domain)
        self.assertEqual([(r["exact_value"], r["multiplicity"]) for r in result["roots"]], [(Q(-1), 2), (Q(0), 3), (Q(1), 1)])
        self.assertEqual(len({r["ordinal"] for r in result["roots"]}), 3)

    def test_irrational_roots_have_separate_certified_sturm_isolators(self):
        # Break caught: treating interval overlap as an exact root or missing a root.
        p, domain = (Q(-2), Q(0), Q(1)), (Q(-2), Q(2))
        result = selector.isolate_roots(p, domain, budget=self.budget())
        self.check_root_set(result, p, domain)
        self.assertTrue(result["sturm_chains"])
        self.assertEqual(len(result["roots"]), 2)
        left, right = result["roots"]
        self.assertLess(left["isolator"][1], Q(0))
        self.assertGreater(right["isolator"][0], Q(0))
        for root in (left, right):
            self.assertIsNone(root["exact_value"])
            self.assertEqual(root["multiplicity"], 1)
            lo, hi = root["isolator"]
            self.assertTrue(type(lo) is Q and type(hi) is Q)
            self.assertLess(lo, hi)
            self.assertLessEqual(hi - lo, Q(1, 2**160))
            self.assertLess(evaluate(p, lo) * evaluate(p, hi), 0)

    def test_negative_leading_scalar_and_repeated_irrational_root_counts(self):
        # Break caught: discarding the leading scalar during square-free/gcd
        # factorization. The original polynomial is -2 (x^2-2)^2.
        p, domain = (Q(-8), Q(0), Q(8), Q(0), Q(-2)), (Q(-2), Q(2))
        result = selector.isolate_roots(p, domain, budget=self.budget())
        self.check_root_set(result, p, domain)
        self.assertEqual(len(result["roots"]), 2)
        self.assertEqual([root["multiplicity"] for root in result["roots"]], [2, 2])
        self.assertLess(result["roots"][0]["isolator"][1], Q(0))
        self.assertGreater(result["roots"][1]["isolator"][0], Q(0))
        self.assertEqual(evaluate(result["polynomial"], Q(0)), Q(-8))
        self.assertEqual(evaluate(result["polynomial"], Q(1)), Q(-2))

    def test_sturm_variation_discards_zero_nonleading_entries(self):
        # Break caught: counting an internal chain zero as a sign change.
        # For x^2-1 at x=0 the derivative vanishes, but the factor does not.
        p, domain = (Q(-1), Q(0), Q(1)), (Q(0), Q(2))
        result = selector.isolate_roots(p, domain, budget=self.budget())
        self.check_root_set(result, p, domain)
        self.assertEqual(len(result["roots"]), 1)
        root = result["roots"][0]
        if root["exact_value"] is None:
            self.contains(root["isolator"], Q(1))
        else:
            self.assertEqual(root["exact_value"], Q(1))


class SelectorBehaviorTests(ContractCase):
    def check_level_and_incidence_witnesses(self, certificate, u, v):
        levels = {level["id"]: level for level in certificate["levels"]}
        for level in levels.values():
            h = level["centered_height"]
            self.assertIs(type(h), Q)
            original = (-h, v, u / 2, Q(0), Q(-1, 4))
            for component in level["components"]:
                witness = component["interior_witness"]
                value = evaluate(original, witness)
                self.assertGreater(value, 0)
                self.assertEqual(component["witness_polynomial_value"], value)
        for edge in certificate["incidence"]:
            upper, lower = levels[edge["upper_level"]], levels[edge["lower_level"]]
            self.assertGreater(upper["centered_height"], lower["centered_height"])
            self.assertIn(edge["upper_component"], {c["id"] for c in upper["components"]})
            self.assertIn(edge["lower_component"], {c["id"] for c in lower["components"]})
            lower_p = (-lower["centered_height"], v, u / 2, Q(0), Q(-1, 4))
            value = evaluate(lower_p, edge["witness"])
            self.assertGreater(value, 0)
            self.assertEqual(edge["lower_polynomial_value"], value)

    def test_hostile_younger_right_is_derived_from_two_to_one_incidence(self):
        # Break caught: fixed-left lifetime, reversed polynomial signs, or elder swap.
        u, v = Q(7, 2**36), Q(-6, 2**54)
        certificate, _ = self.sound_selection(u, v)
        roots = self.ordered_roots(certificate)
        self.assertEqual([r["exact_value"] for r in roots], [-3 * A, A, 2 * A])
        bars = certificate["actual_barcode"]["finite"]
        self.assertEqual(len(bars), 1)
        bar = bars[0]
        self.assertEqual(bar["birth_root"], roots[2]["ordinal"])
        self.assertEqual(certificate["actual_barcode"]["essential"][0]["birth_root"], roots[0]["ordinal"])
        self.assertEqual(bar["lifetime_exact"], Q(3, 2**74))
        self.assertNotEqual(bar["lifetime_exact"], Q(1, 2**67))
        self.assertFalse(bar["tied_birth"])
        self.contains(bar["lifetime_interval"], Q(3, 2**74))
        levels = sorted(certificate["levels"], key=lambda level: level["centered_height"], reverse=True)
        self.assertEqual([len(level["components"]) for level in levels], [0, 1, 2, 1])
        groups = sorted(certificate["height_groups"], key=lambda group: group["centered_interval"][0], reverse=True)
        self.assertEqual(len(groups), 3)
        for upper, lower in zip(groups, groups[1:]):
            self.assertGreater(upper["centered_interval"][0], lower["centered_interval"][1])
        self.assertEqual(certificate["population_projection"]["count"], 1)
        self.assertEqual(certificate["population_projection"]["actual_finite_count"], 1)
        self.check_level_and_incidence_witnesses(certificate, u, v)

    def test_reflection_changes_younger_to_left_without_changing_lifetime(self):
        # Break caught: one fixed birth label is reused under reflection.
        certificate, _ = self.sound_selection(Q(7, 2**36), Q(6, 2**54))
        roots = self.ordered_roots(certificate)
        self.assertEqual([r["exact_value"] for r in roots], [-2 * A, -A, 3 * A])
        self.assertEqual(len(certificate["actual_barcode"]["finite"]), 1)
        bar = certificate["actual_barcode"]["finite"][0]
        self.assertEqual(bar["birth_root"], roots[0]["ordinal"])
        self.assertEqual(bar["lifetime_exact"], Q(3, 2**74))

    def test_irrational_equal_births_merge_but_are_excluded_from_population_law(self):
        # Break caught: treating tie exclusion as no actual bar, or equality by overlap.
        certificate, _ = self.sound_selection(Q(3, 2**33), Q(0))
        roots = self.ordered_roots(certificate)
        self.assertEqual(len(roots), 3)
        self.assertIsNone(roots[0]["exact_value"])
        self.assertIsNone(roots[2]["exact_value"])
        self.assertEqual(roots[1]["exact_value"], Q(0))
        groups = certificate["height_groups"]
        self.assertEqual(len(groups), 2)
        ties = [group for group in groups if len(group["root_ordinals"]) == 2]
        self.assertEqual(len(ties), 1)
        tie = ties[0]
        self.assertEqual(set(tie["root_ordinals"]), {roots[0]["ordinal"], roots[2]["ordinal"]})
        self.assertIsNotNone(tie["equality_witness"])
        self.contains(tie["centered_interval"], Q(9, 2**68))
        self.assertEqual(len(certificate["actual_barcode"]["finite"]), 1)
        bar = certificate["actual_barcode"]["finite"][0]
        self.assertTrue(bar["tied_birth"])
        self.assertEqual(bar["birth_root"], roots[2]["ordinal"])
        self.assertEqual(bar["lifetime_exact"], Q(9, 2**68))
        self.assertEqual(certificate["actual_barcode"]["essential"][0]["birth_root"], roots[0]["ordinal"])
        self.contains(bar["lifetime_interval"], Q(9, 2**68))
        projection = certificate["population_projection"]
        self.assertEqual(projection["count"], 0)
        self.assertEqual(projection["actual_finite_count"], 1)
        self.assertIn("EXCLUDED_TIE", (projection["status"], projection["reason"]))
        self.assertEqual(sorted(len(level["components"]) for level in certificate["levels"]), [0, 1, 2])

    def test_rational_tie_retains_actual_bar_under_the_same_label_convention(self):
        # Break caught: tie handling works only for irrational spatial roots.
        certificate, _ = self.sound_selection(Q(1, 2**34), Q(0))
        roots = self.ordered_roots(certificate)
        self.assertEqual([r["exact_value"] for r in roots], [Q(-1, 2**17), Q(0), Q(1, 2**17)])
        self.assertEqual(len(certificate["actual_barcode"]["finite"]), 1)
        bar = certificate["actual_barcode"]["finite"][0]
        self.assertEqual(bar["birth_root"], roots[2]["ordinal"])
        self.assertEqual(bar["lifetime_exact"], Q(1, 2**70))
        self.assertTrue(bar["tied_birth"])
        self.assertEqual(certificate["population_projection"]["count"], 0)
        self.assertEqual(certificate["population_projection"]["actual_finite_count"], 1)

    def test_fold_multiplicities_do_not_create_fictitious_zero_length_bars(self):
        # Break caught: treating a repeated critical root as an ordinary Morse event.
        for u, v, expected in (
            (Q(3, 2**36), Q(2, 2**54), [(-A, 2), (2 * A, 1)]),
            (Q(12, 2**36), Q(-16, 2**54), [(-4 * A, 1), (2 * A, 2)]),
            (Q(0), Q(0), [(Q(0), 3)]),
        ):
            with self.subTest(u=u, v=v):
                certificate, _ = self.sound_selection(u, v)
                self.assertEqual([(r["exact_value"], r["multiplicity"]) for r in self.ordered_roots(certificate)], expected)
                self.assertEqual(certificate["actual_barcode"]["finite"], [])
                self.assertEqual(certificate["population_projection"]["count"], 0)
                self.assertEqual(certificate["population_projection"]["actual_finite_count"], 0)
                repeated = {r["ordinal"] for r in self.ordered_roots(certificate) if r["multiplicity"] > 1}
                for height in certificate["critical_heights"]:
                    if height["root_ordinal"] in repeated:
                        # The inertia's representation is not frozen: multiplicity
                        # and no invented positive finite bar remain ABI assertions.
                        self.assertGreater(height["multiplicity"], 1)

    def test_nonwedge_has_no_finite_bar(self):
        # Break caught: selecting two maxima without three distinct cubic roots.
        certificate, _ = self.sound_selection(-U0 / 2, V0 / 2)
        self.assertEqual(len(self.ordered_roots(certificate)), 1)
        self.assertEqual(certificate["actual_barcode"]["finite"], [])
        self.assertEqual(certificate["population_projection"]["count"], 0)


class VerificationTamperingTests(ContractCase):
    def hostile(self):
        u, v = Q(7, 2**36), Q(-6, 2**54)
        certificate, budget = self.sound_selection(u, v)
        return u, v, certificate, budget

    def test_verifier_replays_without_calling_selector(self):
        # Break caught: verifier delegates to producer instead of checking records.
        u, v, certificate, budget = self.hostile()
        with patch.object(selector, "select_h0", side_effect=AssertionError("verifier called selector")):
            self.assertEqual(selector.verify_h0(u, v, certificate, budget=budget)["status"], "PASS")

    def test_changed_root_multiplicity_is_rejected_despite_pass_flags(self):
        u, v, certificate, _ = self.hostile()
        bad = copy.deepcopy(certificate)
        roots_in(bad["critical_roots"])[0]["multiplicity"] += 1
        self.rejected_certificate(u, v, bad)

    def test_changed_quartic_factorization_scalar_is_rejected(self):
        # Break caught: verifier checks monic roots but ignores original sign.
        u, v, certificate, _ = self.hostile()
        bad = copy.deepcopy(certificate)
        bad["levels"][0]["root_certificate"]["scalar_unit"] *= -1
        self.rejected_certificate(u, v, bad)

    def test_omitted_regular_component_is_rejected_despite_pass_flags(self):
        u, v, certificate, _ = self.hostile()
        bad = copy.deepcopy(certificate)
        next(level for level in bad["levels"] if len(level["components"]) == 2)["components"].pop()
        self.rejected_certificate(u, v, bad)

    def test_changed_original_polynomial_sign_witness_is_rejected(self):
        u, v, certificate, _ = self.hostile()
        bad = copy.deepcopy(certificate)
        component = next(c for level in bad["levels"] for c in level["components"])
        component["witness_polynomial_value"] = -component["witness_polynomial_value"]
        self.rejected_certificate(u, v, bad)

    def test_changed_incidence_sign_is_rejected(self):
        u, v, certificate, _ = self.hostile()
        bad = copy.deepcopy(certificate)
        bad["incidence"][0]["lower_polynomial_value"] *= -1
        self.rejected_certificate(u, v, bad)

    def test_swapped_elder_birth_is_rejected(self):
        # Break caught: barcode and independently derived elder graph disagree.
        u, v, certificate, _ = self.hostile()
        bad = copy.deepcopy(certificate)
        bad["actual_barcode"]["finite"][0]["birth_root"] = self.ordered_roots(bad)[0]["ordinal"]
        self.rejected_certificate(u, v, bad)

    def test_swapped_elder_graph_decision_is_rejected(self):
        # Break caught: trusting producer elder decisions despite older birth height.
        u, v, certificate, _ = self.hostile()
        bad = copy.deepcopy(certificate)
        merge = next(merge for event in bad["events"] for merge in event["merges"])
        self.assertEqual(len(merge["dying_births"]), 1)
        survivor, dying = merge["survivor_birth"], merge["dying_births"][0]
        merge["survivor_birth"], merge["dying_births"] = dying, [survivor]
        self.rejected_certificate(u, v, bad)

    def test_changed_finite_bar_death_group_is_rejected(self):
        u, v, certificate, _ = self.hostile()
        bad = copy.deepcopy(certificate)
        bar = bad["actual_barcode"]["finite"][0]
        bar["death_group"] = next(group["id"] for group in bad["height_groups"] if group["id"] != bar["death_group"])
        self.rejected_certificate(u, v, bad)

    def test_omitted_merge_batch_is_rejected(self):
        # Break caught: verifier accepts a barcode without its merger event.
        u, v, certificate, _ = self.hostile()
        bad = copy.deepcopy(certificate)
        merger = next(event for event in bad["events"] if event["merges"])
        merger["merges"] = []
        self.rejected_certificate(u, v, bad)

    def test_missing_irrational_equality_witness_is_rejected(self):
        u, v = Q(3, 2**33), Q(0)
        certificate, _ = self.sound_selection(u, v)
        bad = copy.deepcopy(certificate)
        next(group for group in bad["height_groups"] if len(group["root_ordinals"]) == 2)["equality_witness"] = None
        self.rejected_certificate(u, v, bad)


class RejectionAndBudgetTests(ContractCase):
    def test_selector_rejects_inexact_controls_and_closed_rectangle_boundary(self):
        for u, v in ((0.0, Q(0)), (True, Q(0)), (0, Q(0)), (Q(0), False), (Q(0), 0.0), (U0, Q(0)), (-U0, Q(0)), (Q(0), V0), (Q(0), -V0)):
            with self.subTest(u=u, v=v):
                error = TypeError if type(u) is not Q or type(v) is not Q else ValueError
                with self.assertRaises(error):
                    selector.select_h0(u, v, budget=self.budget())

    def test_root_isolator_rejects_inexact_coefficients_and_bad_domains(self):
        for p, domain in (
            ((Q(-1), 0.0, Q(1)), (Q(-2), Q(2))),
            ((Q(-1), True, Q(1)), (Q(-2), Q(2))),
            ((Q(-1), 0, Q(1)), (Q(-2), Q(2))),
            ((Q(-1), Q(0), Q(1)), (Q(2), Q(-2))),
            ((Q(-1), Q(0), Q(1)), (-2.0, Q(2))),
            ((Q(-1), Q(0), Q(1)), (False, Q(2))),
        ):
            with self.subTest(p=p, domain=domain):
                error = TypeError if any(type(x) is not Q for x in (*p, *domain)) else ValueError
                with self.assertRaises(error):
                    selector.isolate_roots(p, domain, budget=self.budget())

    def test_reduced_polynomial_budget_retains_partial_selection_without_acceptance(self):
        # Break caught: exhaustion escapes without partial records or leaves a
        # nested barcode CERTIFIED. Uses only the frozen Budget constructor.
        budget = self.budget(polynomial_limit=1)
        result = selector.select_h0(Q(7, 2**36), Q(-6, 2**54), budget=budget)
        self.assertEqual(result["status"], "INCONCLUSIVE_BUDGET")
        self.assertTrue(SELECTION_KEYS <= result.keys())
        self.assertTrue(WORK_KEYS <= result["work"].keys())
        self.assertNotIn(result["actual_barcode"]["status"], ("CERTIFIED", "PASS"))
        self.assertNotIn(result["population_projection"]["status"], ("CERTIFIED", "PASS"))
        self.assertIsInstance(budget.snapshot(), dict)
        self.assertGreater(result["work"]["polynomial_evaluations"], 0)


class AdditionalCertificateTests(ContractCase):
    def test_elder_deadline_retains_already_computed_events_and_finite_bar(self):
        # R1: a deadline at the internal elder checkpoint must preserve graph
        # and bar observations that precede the rejection, without inventing
        # the later essential class or population projection.
        u, v = Q(7, 2**36), Q(-6, 2**54)
        budget = self.budget()
        actual_checkpoint, actual_bar = budget.checkpoint, selector._bar
        computed_bars = []
        def observe_bar(*args, **kwargs):
            bar = actual_bar(*args, **kwargs)
            computed_bars.append(bar)
            return bar
        def controlled_checkpoint(phase="work"):
            if phase == "elder/group2":
                with patch("surgery_field.time.monotonic", return_value=float(budget.snapshot()["monotonic_deadline"])):
                    return actual_checkpoint(phase)
            return actual_checkpoint(phase)
        with patch.object(selector, "_bar", side_effect=observe_bar), patch.object(budget, "checkpoint", side_effect=controlled_checkpoint):
            certificate = selector.select_h0(u, v, budget=budget)
        barcode, projection = certificate["actual_barcode"], certificate["population_projection"]
        print("R1_RETENTION", {"computed_finite_count": len(computed_bars),
            "computed_lifetime_exact": str(computed_bars[0]["lifetime_exact"]) if computed_bars else None,
            "retained_event_count": len(certificate["events"]), "retained_finite_count": len(barcode["finite"]),
            "retained_essential_count": len(barcode["essential"]), "selection_status": certificate["status"],
            "barcode_status": barcode["status"], "projection_status": projection["status"],
            "failure_phase": certificate["failure"]["phase"], "work": certificate["work"]})
        self.assertEqual(len(computed_bars), 1)
        self.assertEqual(computed_bars[0]["lifetime_exact"], Q(3, 2**74))
        self.assertEqual(certificate["status"], "INCONCLUSIVE_BUDGET")
        self.assertEqual(certificate["failure"]["phase"], "elder/group2")
        self.assertEqual(barcode["status"], "INCONCLUSIVE_BUDGET")
        self.assertEqual(projection["status"], "INCONCLUSIVE_BUDGET")
        self.assertEqual(barcode["finite"], computed_bars)
        self.assertEqual(len(certificate["events"]), 3)
        merge = certificate["events"][-1]["merges"][0]
        self.assertEqual(merge["incoming_births"], [0, 2])
        self.assertEqual(merge["survivor_birth"], 0)
        self.assertEqual(merge["dying_births"], [2])
        self.assertEqual(barcode["finite"][0]["birth_root"], 2)
        self.assertEqual(barcode["finite"][0]["death_group"], certificate["events"][-1]["group_id"])
        self.assertEqual(barcode["essential"], [])
        self.assertEqual(projection["reason"], "UNCOMPUTED")
        self.assertEqual((projection["count"], projection["actual_finite_count"]), (0, 0))
        self.assertEqual(barcode["computed_status_before_withdrawal"], "COMPUTING")
        self.assertEqual(projection["computed_status_before_withdrawal"], "COMPUTING")

    def test_missing_required_selection_and_root_records_are_rejected(self):
        # Break caught: a mathematically plausible but incomplete ABI record
        # silently passes after required status/work/check fields are erased.
        u, v = Q(7, 2**36), Q(-6, 2**54)
        certificate, _ = self.sound_selection(u, v)
        for owner, key in (("selection", "checks"), ("critical_roots", "work")):
            with self.subTest(owner=owner, key=key):
                bad = copy.deepcopy(certificate)
                record = bad if owner == "selection" else bad[owner]
                del record[key]
                self.rejected_certificate(u, v, bad)

    def test_verification_charges_the_same_fixture_without_resetting_deadline(self):
        # Break caught: verification silently opens fresh counters or clock.
        u, v = Q(7, 2**36), Q(-6, 2**54)
        certificate, budget = self.sound_selection(u, v)
        before = budget.snapshot()
        result = selector.verify_h0(u, v, certificate, budget=budget)
        after = budget.snapshot()
        self.assertEqual(result["status"], "PASS")
        for key in ("polynomial_evaluations", "gcd_steps", "polynomial_divisions"):
            self.assertGreater(after[key], before[key])
        self.assertEqual(after["fixture_id"], before["fixture_id"])
        self.assertEqual(after["monotonic_deadline"], before["monotonic_deadline"])
        self.assertEqual(after["root_refinements"], before["root_refinements"])

    def test_each_actual_refinement_has_unique_branch_phase_and_actual_depth(self):
        # Break caught: branch counters reset or conflate distinct refinements.
        budget = self.budget()
        result = selector.isolate_roots((Q(-2), Q(0), Q(1)), (Q(-2), Q(2)), budget=budget)
        self.assertEqual(result["status"], "CERTIFIED")
        refinements = budget.snapshot()["root_refinements"]
        self.assertTrue(refinements)
        self.assertEqual(len({r["phase"] for r in refinements}), len(refinements))
        for refinement in refinements:
            branch = refinement["phase"].split("/")[-2]
            self.assertEqual(len(branch), refinement["depth"])
        self.assertEqual(max(r["depth"] for r in refinements), result["work"]["max_branch_splits"])

    def test_extracted_endpoint_is_retained_when_the_next_actual_evaluation_exhausts(self):
        # Break caught: a discovered exact root stays only in temporary state
        # and disappears from the returned partial certificate at exhaustion.
        budget = self.budget(polynomial_limit=1)
        result = selector.isolate_roots((Q(1), Q(1)), (Q(-1), Q(1)), budget=budget)
        self.assertEqual(result["status"], "INCONCLUSIVE_BUDGET")
        self.assertEqual(result["work"]["polynomial_evaluations"], 1)
        self.assertEqual(len(result["roots"]), 1)
        self.assertEqual(result["roots"][0]["exact_value"], Q(-1))
        self.assertEqual(result["roots"][0]["ordinal"], 0)
        self.assertEqual(result["roots"][0]["multiplicity"], 1)

    def test_irrational_isolator_requires_its_retained_sturm_witness(self):
        # Break caught: accepting an erased required algebra certificate merely
        # because the verifier can independently recompute the missing witness.
        u, v = Q(3, 2**33), Q(0)
        certificate, _ = self.sound_selection(u, v)
        bad = copy.deepcopy(certificate)
        bad["critical_roots"]["sturm_chains"] = []
        self.rejected_certificate(u, v, bad)

    def test_integer_polynomial_coefficients_in_a_certificate_are_rejected(self):
        # Break caught: Python numeric equality masks a forbidden mathematical
        # type while comparing the supplied factor with a Fraction polynomial.
        u, v = Q(7, 2**36), Q(-6, 2**54)
        certificate, _ = self.sound_selection(u, v)
        bad = copy.deepcopy(certificate)
        factor = list(bad["critical_roots"]["roots"][0]["factor"])
        factor[-1] = 1
        bad["critical_roots"]["roots"][0]["factor"] = tuple(factor)
        self.rejected_certificate(u, v, bad)

    def test_verifier_also_avoids_public_root_isolation(self):
        # Break caught: producer rerun replaces independent supplied-proof replay.
        u, v = Q(7, 2**36), Q(-6, 2**54)
        certificate, budget = self.sound_selection(u, v)
        with patch.object(selector, "select_h0", side_effect=AssertionError("producer called")), patch.object(selector, "isolate_roots", side_effect=AssertionError("isolation called")):
            self.assertEqual(selector.verify_h0(u, v, certificate, budget=budget)["status"], "PASS")

    def test_repeated_roots_report_zero_axial_hessian(self):
        # Break caught: assigning an ordinary Morse inertia to a repeated root.
        certificate, _ = self.sound_selection(Q(3, 2**36), Q(2, 2**54))
        repeated = next(h for h in certificate["critical_heights"] if h["multiplicity"] == 2)
        self.assertEqual(repeated["axial_inertia"], "zero")
        self.assertEqual(repeated["transverse_inertia"], "negative")

    def test_late_actual_deadline_withdraws_computed_bars_without_erasing_values(self):
        # Break caught: late shared-clock exhaustion leaves accepted nested
        # bars, loses the numerical result or forgets its computed disposition.
        u, v = Q(7, 2**36), Q(-6, 2**54)
        budget = self.budget()
        actual_checkpoint = budget.checkpoint
        deadline = float(budget.snapshot()["monotonic_deadline"])
        def controlled_checkpoint(phase="work"):
            if phase == "selector/final_acceptance":
                # Fault injection is at the real external clock; the original
                # Budget deadline comparison remains the code under test.
                with patch("surgery_field.time.monotonic", return_value=deadline):
                    return actual_checkpoint(phase)
            return actual_checkpoint(phase)
        with patch.object(budget, "checkpoint", side_effect=controlled_checkpoint):
            certificate = selector.select_h0(u, v, budget=budget)
        self.assertEqual(certificate["status"], "INCONCLUSIVE_BUDGET")
        self.assertEqual(certificate["failure"]["phase"], "selector/final_acceptance")
        barcode = certificate["actual_barcode"]
        self.assertEqual(barcode["status"], "INCONCLUSIVE_BUDGET")
        self.assertEqual(barcode["computed_status_before_withdrawal"], "CERTIFIED")
        self.assertEqual(barcode["finite"][0]["lifetime_exact"], Q(3, 2**74))
        projection = certificate["population_projection"]
        self.assertEqual(projection["status"], "INCONCLUSIVE_BUDGET")
        self.assertEqual(projection["computed_status_before_withdrawal"], "CERTIFIED")
        self.assertEqual((projection["count"], projection["actual_finite_count"]), (1, 1))
        self.assertGreater(certificate["work"]["polynomial_evaluations"], 0)
        self.assertTrue(certificate["events"])


if __name__ == "__main__":
    unittest.main()
