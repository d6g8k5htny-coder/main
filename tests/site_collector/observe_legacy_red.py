"""Prospective hosted observer; Source only until a separate execution pickup.

An intended RED exits 1. Setup/import/errors/incomplete observations exit 2.
All regressions passing exits 0 with NO_RED_OBSERVED, never a baseline RED claim.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

MODULE_NAME = "test_legacy_collector_contract"
EXPECTED_MODULE_PIN = (12877, "834b6bea540d31b49d37a6a2e11e73acc0e301709b2b8efd5eb6abdde702a843")
EXPECTED_SETUP_NAMES = (
    "test_all_six_fixture_pins",
    "test_unchanged_function_extractions",
    "test_static_35_node_aware_membership",
    "test_normalizer_recognizes_imports_and_preserves_non_url_literal",
    "test_correct_imported_js_is_accepted",
    "test_correct_imported_css_is_accepted",
    "test_changed_material_bodies_are_refused",
    "test_valid_historical_four_page_control",
    "test_wrong_historical_index_digest_is_refused",
    "test_independent_literal_leaf_key_oracles",
)
EXPECTED_MARKERS = {
    "test_wrong_imported_js_key_is_refused": "LEGACY_COLLECTOR_ACCEPTED_WRONG_IMPORTED_JS_KEY",
    "test_wrong_imported_css_key_is_refused": "LEGACY_COLLECTOR_ACCEPTED_WRONG_IMPORTED_CSS_KEY",
    "test_explore_wrong_digest_is_refused": "LEGACY_COLLECTOR_ACCEPTED_WRONG_EXPLORE_DIGEST",
    "test_formal_wrong_size_is_refused": "LEGACY_COLLECTOR_ACCEPTED_WRONG_FORMAL_SIZE",
    "test_all_ten_maintained_pages_are_checked": "LEGACY_COLLECTOR_CHECKED_FOUR_OF_TEN_PAGES",
}


class ObservationResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.started = []
        self.observations = []

    def startTest(self, test):
        self.started.append(test.id())
        super().startTest(test)

    def addSuccess(self, test):
        self.observations.append({"id": test.id(), "outcome": "pass"})
        super().addSuccess(test)

    def _exception(self, test, err, outcome):
        kind, value, _ = err
        self.observations.append({"id": test.id(), "outcome": outcome,
                                  "exception_type": kind.__name__, "message": str(value),
                                  "builtin_assertion_error": kind is AssertionError})

    def addFailure(self, test, err):
        self._exception(test, err, "failure")
        super().addFailure(test, err)

    def addError(self, test, err):
        self._exception(test, err, "error")
        super().addError(test, err)

    def addSkip(self, test, reason):
        self.observations.append({"id": test.id(), "outcome": "skip", "reason": reason})
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, err):
        self._exception(test, err, "expected_failure")
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self.observations.append({"id": test.id(), "outcome": "unexpected_success"})
        super().addUnexpectedSuccess(test)

    def as_json(self):
        return {"testsRun": self.testsRun, "started_ids": self.started,
                "failures": len(self.failures), "errors": len(self.errors),
                "skips": len(self.skipped), "expected_failures": len(self.expectedFailures),
                "unexpected_successes": len(self.unexpectedSuccesses),
                "observations": self.observations}


def run_class(cls):
    names = unittest.defaultTestLoader.getTestCaseNames(cls)
    result = ObservationResult()
    unittest.TestSuite(cls(name) for name in names).run(result)
    return names, result.as_json()


def clean_pass(result, expected_ids):
    return (result["testsRun"] == len(expected_ids) and
            set(result["started_ids"]) == set(expected_ids) and
            len(result["started_ids"]) == len(expected_ids) and
            result["failures"] == result["errors"] == result["skips"] ==
            result["expected_failures"] == result["unexpected_successes"] == 0 and
            len(result["observations"]) == len(expected_ids) and
            all(item["outcome"] == "pass" for item in result["observations"]))


def observe(phase, report):
    path = Path(__file__).resolve().with_name(MODULE_NAME + ".py")
    raw = path.read_bytes()
    report["test_source"] = {"path": path.name, "bytes": len(raw),
                             "sha256": hashlib.sha256(raw).hexdigest()}
    if (len(raw), hashlib.sha256(raw).hexdigest()) != EXPECTED_MODULE_PIN:
        raise RuntimeError("Test module differs from independently pinned first-RED Source")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Test module specification unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report["source_commit"] = module.SOURCE_COMMIT
    report["fixture_pins"] = {name: {"bytes": pin[0], "sha256": pin[1]}
                              for name, pin in module.PINS.items()}
    if module.EXPECTED_MARKERS != EXPECTED_MARKERS:
        raise RuntimeError("Observer and regression marker contracts disagree")
    setup_names = unittest.defaultTestLoader.getTestCaseNames(module.BaselineSetupControls)
    if len(setup_names) != 10 or set(setup_names) != set(EXPECTED_SETUP_NAMES):
        raise RuntimeError("Positive setup method inventory differs from independent contract")
    setup_names, setup = run_class(module.BaselineSetupControls)
    report["setup"] = setup
    setup_ids = [MODULE_NAME + ".BaselineSetupControls." + name for name in EXPECTED_SETUP_NAMES]
    if len(setup_names) != 10 or not clean_pass(setup, setup_ids):
        report["status"] = "SETUP_OR_IMPORT_ERROR"
        return 2
    if phase == "setup":
        report["status"] = "SETUP_PASS"
        return 0
    names, regression = run_class(module.CollectorRegressionTests)
    report["regression"] = regression
    expected_ids = {MODULE_NAME + ".CollectorRegressionTests." + name: marker
                    for name, marker in EXPECTED_MARKERS.items()}
    if set(names) != set(EXPECTED_MARKERS):
        report["status"] = "INCOMPLETE_OR_UNEXPECTED_OBSERVATION"
        return 2
    if clean_pass(regression, expected_ids):
        report["status"] = "NO_RED_OBSERVED"
        return 0
    observations = regression["observations"]
    exact_red = (
        regression["testsRun"] == 5 and len(regression["started_ids"]) == 5 and
        set(regression["started_ids"]) == set(expected_ids) and
        regression["failures"] == 5 and regression["errors"] == 0 and
        regression["skips"] == regression["expected_failures"] == regression["unexpected_successes"] == 0 and
        len(observations) == 5 and {item["id"] for item in observations} == set(expected_ids) and
        all(item["outcome"] == "failure" and item["exception_type"] == "AssertionError" and
            item["builtin_assertion_error"] is True and
            item["message"] == expected_ids[item["id"]] for item in observations)
    )
    report["status"] = "GENUINE_RED_FROZEN_BASELINE" if exact_red else "INCOMPLETE_OR_UNEXPECTED_OBSERVATION"
    return 1 if exact_red else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("setup", "regression"))
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "phase": args.phase, "status": "NOT_OBSERVED",
              "scope": "frozen unchanged functions with declared synthetic helpers and fixtures",
              "browser_runtime": "NOT_RUN", "scientific_acceptance": "NONE"}
    try:
        exit_code = observe(args.phase, report)
    except Exception as error:
        report["status"] = "SETUP_OR_IMPORT_ERROR"
        report["driver_error"] = {"exception_type": type(error).__name__, "message": str(error)}
        exit_code = 2
    report["exit_code"] = exit_code
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as error:
        print("REPORT_WRITE_ERROR: " + type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 2
    print(report["status"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
