"""Synthetic engineering controls for the opt-in, nonauthoritative interlock."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "claim_selection_transfer.py"
SPEC = importlib.util.spec_from_file_location("claim_selection_transfer", TOOL)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

ID = {
    "law": "Q_r_weighted_palm", "conditioning": "six_pin",
    "event": "E_r", "count": "N_B", "region": "fixed_B",
    "selection": "event_weighted",
}


def sample(kind="none", order=2, beta="0", target="3"):
    info = {"kind": kind, "identity": copy.deepcopy(ID)}
    if kind == "unconditional_p_moment":
        info.update(order=order, beta=beta)
    return {
        "schema": M.SCHEMA,
        "target": {"identity": copy.deepcopy(ID), "observable": "event_weighted_count_expectation", "requested_power": target},
        "event_bound": {"identity": copy.deepcopy(ID), "rate": "r^3"},
        "count_information": info,
    }


class ClaimSelectionEngineering(unittest.TestCase):
    def test_event_probability_alone_does_not_control_expected_count(self):
        d = M.evaluate(sample())
        self.assertIsNone(d["derived_power"])
        self.assertEqual(d["exponent_comparison"], "no_count_rate_implied")

    def test_holder_p2_is_three_halves_not_three(self):
        d = M.evaluate(sample("unconditional_p_moment", 2))
        self.assertEqual((d["derived_power"], d["exponent_comparison"]),
                         ("3/2", "insufficient_for_declared_target"))

    def test_holder_p4_is_nine_quarters(self):
        d = M.evaluate(sample("unconditional_p_moment", 4))
        self.assertEqual(d["derived_power"], "9/4")

    def test_holder_growing_moment_has_beta_penalty(self):
        d = M.evaluate(sample("unconditional_p_moment", 4, "2"))
        self.assertEqual(d["derived_power"], "7/4")

    def test_holder_inequality_comparison_only(self):
        d = M.evaluate(sample("unconditional_p_moment", 4, target="2"))
        self.assertEqual(d["exponent_comparison"], "meets_declared_target")
        self.assertFalse(d["premises_independently_verified"])
        self.assertFalse(d["scientific_acceptance"])
        self.assertFalse(d["graph_binding_created"])

    def test_conditional_mean_requires_explicit_premise(self):
        d = M.evaluate(sample("uniform_conditional_mean"))
        self.assertEqual(d["derived_power"], "3")
        self.assertEqual(d["route"], "explicit_uniform_conditional_mean")
        self.assertFalse(d["source_identity_runtime_verified"])

    def test_finite_moment_cannot_satisfy_cubic_goal_without_more(self):
        for p in range(2, 65):
            self.assertLess(Fraction(M.evaluate(sample("unconditional_p_moment", p))["derived_power"]), 3)

    def test_rare_count_exact_counterexample(self):
        # Existing Math- source rare_count(n): r=2^-n, P(E)=r^3, N=n on E.
        # The ratio E[N 1_E]/r^3 is n; finite samples are NOT a theorem replay.
        for n in (1, 2, 8, 32):
            r = Fraction(1, 2**n)
            probability = r**3
            self.assertEqual((n * probability) / probability, n)
            self.assertGreaterEqual((n * probability) / r**3, 1)

    def test_scope_dimensions_reject_cross_population(self):
        for name in M.IDENTITY:
            t = sample("unconditional_p_moment", 2)
            t["count_information"]["identity"][name] = "other_scope"
            with self.subTest(name=name), self.assertRaises(M.Refused):
                M.evaluate(t)

    def test_event_scope_mismatch_refuses(self):
        t = sample("uniform_conditional_mean")
        t["event_bound"]["identity"]["conditioning"] = "unconditioned"
        with self.assertRaises(M.Refused):
            M.evaluate(t)

    def test_unconditional_count_not_supported(self):
        t = sample("uniform_conditional_mean")
        t["target"]["observable"] = "whole_count_expectation"
        with self.assertRaises(M.Refused):
            M.evaluate(t)

    def test_float_and_boolean_moment_orders_refuse(self):
        for bad in (1, 0, 2.0, True, -1, 65, "2"):
            t = sample("unconditional_p_moment", bad)
            with self.subTest(bad=bad), self.assertRaises(M.Refused):
                M.evaluate(t)

    def test_nonexact_or_out_of_scope_exponents_refuse(self):
        for bad in (1.0, True, "3.0", "-1", "65", "1/0", "1e3", "1/100000", "01"):
            t = sample("unconditional_p_moment", 2, bad)
            with self.subTest(bad=bad), self.assertRaises(M.Refused):
                M.evaluate(t)

    def test_invalid_identity_value_refuses(self):
        for bad in ("", "x y", "a" * 81, 7, True, "secret\nabc"):
            t = sample()
            t["target"]["identity"]["selection"] = bad
            with self.subTest(bad=bad), self.assertRaises(M.Refused):
                M.evaluate(t)

    def test_wrong_source_rate_refuses(self):
        t = sample()
        t["event_bound"]["rate"] = "r^2"
        with self.assertRaises(M.Refused):
            M.evaluate(t)

    def test_unknown_key_and_extra_count_premise_refuse(self):
        t = sample()
        t["new_acceptance_flag"] = True
        with self.assertRaises(M.Refused):
            M.evaluate(t)
        t = sample("none")
        t["count_information"]["order"] = 2
        with self.assertRaises(M.Refused):
            M.evaluate(t)

    def test_source_pins_are_fixed_but_not_implicitly_verified(self):
        d = M.evaluate(sample())
        self.assertEqual(d["source"]["program_blob"], "d64e119adad0dfb3da5ff033a81234129fccb253")
        self.assertFalse(d["source_identity_runtime_verified"])

    def test_duplicate_json_keys_and_scientific_claim_injection_refuse(self):
        text = json.dumps(sample()).replace('"schema": ', '"schema": "false", "schema": ', 1)
        with self.assertRaises(M.Refused):
            M.parse(text.encode("utf-8"))

    def test_json_float_constant_and_length_refuse(self):
        for raw in (b'{"x":1.0}', b'{"x":NaN}', b'{"x":Infinity}', b'', b'[' + b' ' * M.LIMIT + b']'):
            with self.subTest(raw=raw[:20]), self.assertRaises(M.Refused):
                M.parse(raw)

    def test_cli_success_is_deterministic_and_nonadmitting(self):
        raw = json.dumps(sample("unconditional_p_moment", 2)).encode()
        runs = [subprocess.run([sys.executable, "-B", "-S", str(TOOL)], input=raw, capture_output=True)
                for _ in range(2)]
        for run in runs:
            self.assertEqual((run.returncode, run.stderr), (0, b""))
            data = json.loads(run.stdout)
            self.assertFalse(data["scientific_acceptance"])
            self.assertEqual(data["derived_power"], "3/2")
        self.assertEqual(runs[0].stdout, runs[1].stdout)

    def test_cli_refusal_opaque_and_no_stdout(self):
        raw = b'{"personal_secret":"this must not echo","personal_secret":42}'
        run = subprocess.run([sys.executable, "-B", "-S", str(TOOL)], input=raw, capture_output=True)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(run.stdout, b"")
        self.assertEqual(run.stderr, b"CLAIM_SELECTION_TRANSFER_REFUSED\n")


if __name__ == "__main__":
    unittest.main()
