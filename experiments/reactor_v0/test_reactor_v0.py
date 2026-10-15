import copy
import json
import os
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class ReactorFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluator import evaluate_package
        from validator import load_and_validate

        cls.evaluate_package = staticmethod(evaluate_package)
        cls.load_and_validate = staticmethod(load_and_validate)
        cls.package = load_and_validate(HERE)

    def test_exposed_control_is_exact_five_sixteenths(self):
        result = self.evaluate_package(self.package)
        self.assertEqual(result["expected_short_bar_count"], Fraction(5, 16))
        self.assertEqual(result["finite_bars"], ((1, 0), (8, 0)))
        self.assertEqual(result["essential_birth"], 27)
        self.assertEqual(result["included_lifetimes"], (1, 8))

    def test_tampered_expected_control_is_rejected(self):
        package = copy.deepcopy(self.package)
        package["control"]["expected_control"] = {"numerator": 1, "denominator": 2}
        with self.assertRaisesRegex(ValueError, "declared expected control"):
            self.evaluate_package(package)

    def test_threshold_is_inclusive_and_essential_class_is_excluded(self):
        from evaluator import short_bar_lifetimes

        barcode = {"intervals": [[3, 1], [8, 0]], "essential": [27]}
        self.assertEqual(short_bar_lifetimes(barcode, Fraction(2)), (2,))

    def test_unsupported_exact_root_is_refused(self):
        from exact_helpers import rational_power_exact

        with self.assertRaisesRegex(ValueError, "rational root"):
            rational_power_exact(Fraction(1, 2), Fraction(2, 3))

    def test_schema_rejects_missing_extra_wrong_type_and_status_effect(self):
        from validator import validate_document

        good = json.loads((HERE / "control.json").read_text(encoding="utf-8"))
        mutations = []
        missing = copy.deepcopy(good)
        del missing["scientific_status_effect"]
        mutations.append(missing)
        extra = copy.deepcopy(good)
        extra["unexpected"] = False
        mutations.append(extra)
        wrong_type = copy.deepcopy(good)
        wrong_type["threshold"] = 1.0
        mutations.append(wrong_type)
        wrong_status = copy.deepcopy(good)
        wrong_status["scientific_status_effect"] = "ACCEPTED"
        mutations.append(wrong_status)
        for document in mutations:
            with self.subTest(document=document), self.assertRaises(ValueError):
                validate_document(document)

    def test_source_manifest_separates_raw_hash_from_git_membership(self):
        sources = self.package["sources"]
        self.assertEqual(len(sources), 2)
        for source in sources:
            self.assertEqual(source["repository"], "d6g8k5htny-coder/main")
            self.assertEqual(len(source["commit"]), 40)
            self.assertEqual(len(source["raw_sha256"]), 64)
            self.assertEqual(len(source["git_blob_sha"]), 40)
            self.assertFalse(source["theorem_conclusion_consumed"])

    def test_consumed_closure_hashes_match_exact_bytes(self):
        from validator import verify_closure

        self.assertTrue(verify_closure(HERE, self.package["closure"]))

    def test_optimized_child_process_runs_validator_and_evaluator(self):
        completed = subprocess.run(
            [sys.executable, "-O", "-B", str(HERE / "run_control.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["expected_short_bar_count"], "5/16")
        self.assertTrue(payload["validated"])


if __name__ == "__main__":
    unittest.main()
