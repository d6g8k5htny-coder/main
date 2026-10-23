"""Tests for tools/evidence_path.py (LANES-20261022 Lane 1, main#475).

Scientific effect NONE. The complete path uses a synthetic fixture node; the
real punctured-pin node must stay explicitly missing.
"""
import copy
import importlib.util
import io
import json
from pathlib import Path
import contextlib
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evidence_path", ROOT / "tools" / "evidence_path.py")
evidence_path_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence_path_module)
evidence_path = evidence_path_module.evidence_path
Refused = evidence_path_module.EvidencePathRefused

PIN = "math.d5-component.punctured-pin-proof"
PIN_MISSING = "public projection not admitted (#327 exporter held by Claude017)"
FIXTURE_NODE = "fixture.lane1.bound-node"
COMMIT = "a" * 40
GRAPH = {
    "data": {"edges": [
        {"from": FIXTURE_NODE, "relation": "requires_evidence", "required": True, "to": PIN},
    ]},
    "identity": {"repository": "example/fixture", "commit": COMMIT, "path": "GRAPH.json"},
}
BINDINGS = [{"node_id": FIXTURE_NODE, "artifact_id": "FIXTURE-ARTIFACT-v1", "roles": ["source"]}]
RECEIPTS = {"FIXTURE-ARTIFACT-v1": {"url": "https://example.invalid/receipt", "sha": "b" * 64}}
REVIEWS = {"FIXTURE-ARTIFACT-v1": {"url": "https://example.invalid/review"}}


class EvidencePathTests(unittest.TestCase):
    def test_complete_path_for_synthetic_bound_node(self):
        result = evidence_path(FIXTURE_NODE, GRAPH, BINDINGS, RECEIPTS, REVIEWS)
        self.assertEqual(result["missing"], [])
        self.assertTrue(result["complete"])
        self.assertEqual(result["source"], {
            "url": "https://github.com/example/fixture/blob/" + COMMIT + "/GRAPH.json", "sha": COMMIT})
        self.assertEqual(result["artifacts"], [{"artifact_id": "FIXTURE-ARTIFACT-v1", "roles": ["source"]}])
        self.assertEqual(result["receipts"][0]["sha"], "b" * 64)
        self.assertEqual(result["reviews"][0]["url"], "https://example.invalid/review")
        self.assertEqual(result["scientific_effect"], "NONE")

    def test_punctured_pin_projection_is_explicitly_missing(self):
        result = evidence_path(PIN, GRAPH, BINDINGS, RECEIPTS, REVIEWS)
        self.assertFalse(result["complete"])
        self.assertEqual(result["artifacts"], [])
        self.assertIn(PIN_MISSING, result["missing"])
        self.assertNotIn("artifact index binding not recorded", result["missing"])
        self.assertEqual(result["receipts"], [])
        self.assertEqual(result["reviews"], [])

    def test_missing_receipt_and_review_are_named_not_padded(self):
        result = evidence_path(FIXTURE_NODE, GRAPH, BINDINGS)
        self.assertEqual(result["missing"], [
            "execution receipt not recorded for FIXTURE-ARTIFACT-v1",
            "review link not recorded for FIXTURE-ARTIFACT-v1"])

    def test_generic_unbound_node_reports_missing_binding(self):
        graph = copy.deepcopy(GRAPH)
        graph["data"]["edges"].append(
            {"from": FIXTURE_NODE, "relation": "requires_evidence", "to": "fixture.lane1.unbound-node"})
        result = evidence_path("fixture.lane1.unbound-node", graph, BINDINGS, RECEIPTS, REVIEWS)
        self.assertFalse(result["complete"])
        self.assertEqual(result["artifacts"], [])
        self.assertEqual(result["missing"], [
            "artifact index binding not recorded",
            "execution receipt not recorded",
            "review link not recorded"])

    def test_non_sha_commit_in_identity_is_not_a_pinned_source(self):
        for bad in ("main", "HEAD", "a" * 39, "A" * 40, "g" * 40, "a" * 41, 7, None):
            graph = copy.deepcopy(GRAPH)
            graph["identity"]["commit"] = bad
            result = evidence_path(FIXTURE_NODE, graph, BINDINGS, RECEIPTS, REVIEWS)
            self.assertIsNone(result["source"], bad)
            self.assertIn("pinned source (url+sha) not recorded", result["missing"])
            self.assertFalse(result["complete"])

    def test_missing_empty_or_traversal_path_is_not_a_pinned_source(self):
        cases = {"missing": None, "empty": "", "traversal": "../GRAPH.json",
                 "inner-traversal": "docs/../../GRAPH.json", "absolute": "/GRAPH.json",
                 "non-string": 7}
        for label, bad in cases.items():
            graph = copy.deepcopy(GRAPH)
            if bad is None:
                del graph["identity"]["path"]
            else:
                graph["identity"]["path"] = bad
            result = evidence_path(FIXTURE_NODE, graph, BINDINGS, RECEIPTS, REVIEWS)
            self.assertIsNone(result["source"], label)
            self.assertIn("pinned source (url+sha) not recorded", result["missing"], label)
            self.assertFalse(result["complete"], label)

    def test_unknown_node_is_refused(self):
        with self.assertRaises(Refused) as caught:
            evidence_path("fixture.lane1.not-in-graph", GRAPH, BINDINGS)
        self.assertEqual(caught.exception.code, "unknown-node")

    def test_malformed_node_id_is_refused(self):
        for bad in ("", "../x", 7, None):
            with self.assertRaises(Refused):
                evidence_path(bad, GRAPH, BINDINGS)

    def test_inputs_are_not_mutated(self):
        before = copy.deepcopy((GRAPH, BINDINGS, RECEIPTS, REVIEWS))
        evidence_path(FIXTURE_NODE, GRAPH, BINDINGS, RECEIPTS, REVIEWS)
        self.assertEqual(before, (GRAPH, BINDINGS, RECEIPTS, REVIEWS))

    def test_repository_punctured_pin_is_missing_via_cli(self):
        # Reads the checked-in graph and bindings only; no network, no writes.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = evidence_path_module.main([PIN])
        self.assertEqual(code, 0)
        result = json.loads(out.getvalue())
        self.assertIn(PIN_MISSING, result["missing"])
        self.assertFalse(result["complete"])

    def test_cli_refuses_unknown_node(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = evidence_path_module.main(["fixture.lane1.not-in-graph"])
        self.assertEqual(code, 2)
        self.assertIn("unknown-node", err.getvalue())


if __name__ == "__main__":
    unittest.main()
