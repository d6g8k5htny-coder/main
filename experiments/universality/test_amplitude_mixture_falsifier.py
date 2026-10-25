"""Adversarial exact controls for the covariance-only amplitude specimen.

Scientific effect NONE. No field/barcode simulation or source-status acceptance.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "universality" / "amplitude_mixture_falsifier.py"
SPEC = importlib.util.spec_from_file_location("amplitude_mixture_falsifier", SCRIPT)
engine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engine)


class ExactAmplitudeTests(unittest.TestCase):
    def sample(self):
        return copy.deepcopy(engine.EXAMPLE_SPEC)

    def test_second_moment_preserved_fourth_separates(self):
        result = engine.witness(self.sample())
        model = result["scale_model"]
        self.assertEqual(model["second_scale_moment"], "1")
        self.assertEqual(model["fourth_scale_moment"], "25/9")
        self.assertEqual(model["normalized_mixture_fourth_moment"], "25/3")
        self.assertEqual(model["fourth_scale_moment_minus_one"], "16/9")
        self.assertEqual(model["inverse_two_thirds_scale_moment"], "DIVERGES")
        self.assertTrue(model["same_covariance_for_independent_centered_H"])
        self.assertNotEqual(model["normalized_mixture_fourth_moment"],
                            model["normalized_gaussian_fourth_moment"])

    def test_exact_conditional_witness(self):
        r = engine.witness(self.sample())
        self.assertEqual(r["premise_status"], "CALLER_SUPPLIED_NOT_VERIFIED")
        self.assertEqual(r["conditional_short_bar_witness"]["growth_exponent"], "1/3")
        self.assertEqual(r["conditional_short_bar_witness"]["grid"][0], {
            "m": 2, "t_over_sqrt_2kplus1_times_upper_bar": "1/4",
            "probability_R_at_m_threshold": "1/2",
            "expected_short_bar_count_lower_bound": "1/8",
        })
        self.assertEqual(r["conditional_short_bar_witness"]["grid"][-1][
            "expected_short_bar_count_lower_bound"], "1/36")

    def test_general_integer_k_is_exact(self):
        obj = self.sample()
        obj["power"] = 3
        r = engine.witness(obj)
        self.assertEqual(r["scale_model"]["small_amplitude_cdf_exponent"], "1/3")
        self.assertEqual(r["scale_model"]["fourth_scale_moment"], "49/13")
        self.assertEqual(r["conditional_short_bar_witness"]["growth_exponent"], "1")
        self.assertEqual(r["conditional_short_bar_witness"]["grid"][0][
            "t_over_sqrt_2kplus1_times_upper_bar"], "1/8")

    def test_canonical_fractions(self):
        obj = self.sample()
        obj["bar_interval"] = {"lower": "2/4", "upper": "9/6"}
        obj["base_expected_bar_count_lower_bound"] = "4/16"
        r = engine.witness(obj)
        self.assertEqual(r["premises"]["bar_interval_lower"], "1/2")
        self.assertEqual(r["premises"]["bar_interval_upper"], "3/2")
        self.assertEqual(r["premises"]["expected_base_bar_count_lower_bound"], "1/4")

    def test_witness_does_not_mutate_input(self):
        obj = self.sample()
        before = copy.deepcopy(obj)
        engine.witness(obj)
        self.assertEqual(obj, before)

    def test_invalid_power_and_rationals_refuse(self):
        cases = []
        for power in (0, 1, 17, True, 2.0, "2", None):
            obj = self.sample()
            obj["power"] = power
            cases.append(obj)
        for bad in ("0", "-1/2", "1/0", "0.2", "nan", "1/01", 1, None):
            obj = self.sample()
            obj["base_expected_bar_count_lower_bound"] = bad
            cases.append(obj)
        for obj in cases:
            with self.subTest(payload=str(obj)[:60]):
                with self.assertRaises(engine.SpecRefused):
                    engine.witness(obj)

    def test_invalid_intervals_refuse(self):
        for a,b in (("1","1"),("2","1"),("0","1"),("1","0")):
            with self.subTest(a=a,b=b):
                obj=self.sample()
                obj["bar_interval"]={"lower":a,"upper":b}
                with self.assertRaises(engine.SpecRefused):
                    engine.witness(obj)

    def test_grid_validation_refuses_empty_repeated_unsorted(self):
        for bad in ([],[1],[2,2],[4,3],[True,2],[2,"3"],[2.5],[100001],[2]*21):
            with self.subTest(bad=bad):
                obj=self.sample()
                obj["grid"] = bad
                with self.assertRaises(engine.SpecRefused):
                    engine.witness(obj)

    def test_strict_json_duplicate_unknown_float_nonfinite_refuse(self):
        raw_cases = (
            b'{"schema_version":1,"schema_version":1}',
            b'{"k":2}',
            json.dumps(dict(self.sample(), surprise="http://bad.example")).encode(),
            json.dumps(dict(self.sample(), power=2.0)).encode(),
            json.dumps(dict(self.sample(), power=None)).encode(),
            b'NaN',
            b'Infinity',
            b'[]',
            b'\xff\xfe',
            b'0' * (engine.MAX_INPUT + 1),
        )
        for raw in raw_cases:
            with self.subTest(raw=raw[:35]):
                with self.assertRaises(engine.SpecRefused):
                    engine.load_spec(raw)

    def test_source_identity_and_limits_explicit(self):
        obj=engine.witness(self.sample())
        self.assertEqual(obj["scientific_effect"], "NONE")
        self.assertFalse(obj["scientific_status_authority"])
        self.assertFalse(obj["source_identity_runtime_verified"])
        self.assertEqual(len(obj["source_identity"]["git_blob"]), 40)
        self.assertEqual(obj["covariance_only_two_thirds_transfer"],
                         "REFUSED_BY_COUNTEREXAMPLE")
        self.assertGreaterEqual(len(obj["scope_limits"]), 3)


class CLIControls(unittest.TestCase):
    def run_program(self, mode=(), args=(), stdin=None):
        return subprocess.run([sys.executable,"-B",*mode,"-S",str(SCRIPT),*args],
                              input=stdin, capture_output=True, timeout=10)

    def test_example_is_inert_and_roundtrips(self):
        p=self.run_program(args=("--example",))
        self.assertEqual(p.returncode,0,p.stderr.decode())
        self.assertEqual(json.loads(p.stdout),engine.EXAMPLE_SPEC)
        self.assertEqual(p.stderr,b"")

    def test_normal_optimized_identical_stdout(self):
        sample=json.dumps(engine.EXAMPLE_SPEC).encode()
        a=self.run_program(args=("--spec","-"),stdin=sample)
        b=self.run_program(mode=("-O",),args=("--spec","-"),stdin=sample)
        self.assertEqual(a.returncode,0,a.stderr.decode())
        self.assertEqual(b.returncode,0,b.stderr.decode())
        self.assertEqual(a.stdout,b.stdout)
        self.assertEqual(a.stderr,b"")
        self.assertEqual(json.loads(a.stdout)["scale_model"]["fourth_scale_moment"],"25/9")

    def test_require_covariance_transfer_vetoes_both_modes(self):
        sample=json.dumps(engine.EXAMPLE_SPEC).encode()
        for mode in ((),("-O",)):
            with self.subTest(mode=mode):
                p=self.run_program(mode=mode,args=("--spec","-",
                              "--require-covariance-transfer"),stdin=sample)
                self.assertEqual(p.returncode,1)
                self.assertEqual(p.stderr,b"")
                self.assertEqual(json.loads(p.stdout)["covariance_only_two_thirds_transfer"],
                                 "REFUSED_BY_COUNTEREXAMPLE")

    def test_malformed_cli_is_opaque_in_both_modes(self):
        raw=b'{"schema_version":1,"url":"https://secret.example/a-key"}'
        for mode in ((),("-O",)):
            with self.subTest(mode=mode):
                p=self.run_program(mode=mode,args=("--spec","-"),stdin=raw)
                self.assertEqual(p.returncode,2)
                self.assertEqual(p.stdout,b"")
                self.assertEqual(p.stderr,b"AMP_TWIN_REFUSED: invalid-input\n")
                self.assertNotIn(b"secret",p.stderr)

    def test_input_file_and_stdin_equivalent(self):
        raw=json.dumps(engine.EXAMPLE_SPEC).encode()
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"fixture.json"
            path.write_bytes(raw)
            by_path=self.run_program(args=("--spec",str(path)))
        by_stdin=self.run_program(args=("--spec","-"),stdin=raw)
        self.assertEqual(by_path.returncode,0)
        self.assertEqual(by_path.stdout,by_stdin.stdout)

    def test_mutation_to_constant_scale_breaks_counterexample(self):
        # The naive R=1 replacement keeps covariance but has E[R^4]=1.
        # That replacement MUST fail the counterexample's 4th-moment witness.
        obj=engine.witness(engine.EXAMPLE_SPEC)
        self.assertEqual(Fraction(obj["scale_model"]["second_scale_moment"]),1)
        actual_fourth=Fraction(obj["scale_model"]["fourth_scale_moment"])
        mutant_fourth=Fraction(1)
        self.assertNotEqual(actual_fourth,mutant_fourth)
        self.assertGreater(actual_fourth,mutant_fourth)


if __name__ == "__main__":
    unittest.main()
