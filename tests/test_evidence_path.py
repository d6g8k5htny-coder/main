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

    def test_source_path_is_serialized_as_url_path_data(self):
        cases = (
            ("GRAPH.json", "GRAPH.json"),
            ("docs/GRAPH.json", "docs/GRAPH.json"),
            ("%2e%2e/GRAPH.json", "%252e%252e/GRAPH.json"),
            ("%2E%2E/GRAPH.json", "%252E%252E/GRAPH.json"),
            (".%2e/GRAPH.json", ".%252e/GRAPH.json"),
            ("%2e./GRAPH.json", "%252e./GRAPH.json"),
            ("docs/%2e%2e/%2e%2e/GRAPH.json", "docs/%252e%252e/%252e%252e/GRAPH.json"),
            ("docs/a#b?.json", "docs/a%23b%3F.json"),
            ("docs/a b%.json", "docs/a%20b%25.json"),
            ("docs/a%2Fb.json", "docs/a%252Fb.json"),
            ("docs/caf\u00e9.json", "docs/caf%C3%A9.json"),
            ("docs/a\tb\n.json", "docs/a%09b%0A.json"),
        )
        for path, encoded in cases:
            with self.subTest(path=path):
                graph = copy.deepcopy(GRAPH)
                graph["identity"]["path"] = path
                result = evidence_path(FIXTURE_NODE, graph, BINDINGS, RECEIPTS, REVIEWS)
                self.assertEqual(result["source"], {
                    "url": "https://github.com/example/fixture/blob/" + COMMIT + "/" + encoded,
                    "sha": COMMIT,
                })
                self.assertEqual(result["missing"], [])
                self.assertTrue(result["complete"])
                self.assertEqual(graph["identity"]["path"], path)

    def test_malformed_https_evidence_authorities_are_missing(self):
        bad_urls = (
            "https://", "https://?r=1", "https://#review", "https://:443/r",
            "https://[::1/r", "https://example.invalid:notaport/r",
            "https://example.invalid:65536/r", "https://bad host.invalid/r",
            "https://[::1]suffix/r", "https://prefix[::1]/r",
            "https://[v1.a]/r", "https://bad|host.invalid/r",
            "https://%20/r", "https://example.invalid:\uff14\uff14\uff13/r",
        )
        for channel, label in (("receipts", "execution receipt"), ("reviews", "review link")):
            for value in bad_urls:
                with self.subTest(channel=channel, value=value):
                    tables = {"receipts": copy.deepcopy(RECEIPTS), "reviews": copy.deepcopy(REVIEWS)}
                    tables[channel]["FIXTURE-ARTIFACT-v1"]["url"] = value
                    before = copy.deepcopy(tables)
                    result = evidence_path(FIXTURE_NODE, GRAPH, BINDINGS, **tables)
                    self.assertFalse(result["complete"])
                    self.assertEqual(result[channel], [])
                    self.assertEqual(result["missing"], [label + " not recorded for FIXTURE-ARTIFACT-v1"])
                    self.assertEqual(result["scientific_effect"], "NONE")
                    self.assertEqual(tables, before)

    def test_raw_url_whitespace_controls_and_backslashes_are_missing(self):
        bad_urls = (
            "https://example.invalid/r\n", "https://exa\tmple.invalid/r",
            "https://example.invalid/r\r", "https://example.invalid/\x00r",
            "https://example.invalid/\x1fr", "https://example.invalid/\x7fr",
            "https://example.invalid/a b", "https://example.invalid/\u00a0r",
            "https://example.invalid\\other/r", "https://example.invalid/r\\x",
        )
        for channel, label in (("receipts", "execution receipt"), ("reviews", "review link")):
            for value in bad_urls:
                with self.subTest(channel=channel, value=value):
                    tables = {"receipts": copy.deepcopy(RECEIPTS), "reviews": copy.deepcopy(REVIEWS)}
                    tables[channel]["FIXTURE-ARTIFACT-v1"]["url"] = value
                    result = evidence_path(FIXTURE_NODE, GRAPH, BINDINGS, **tables)
                    self.assertFalse(result["complete"])
                    self.assertEqual(result[channel], [])
                    self.assertEqual(result["missing"], [label + " not recorded for FIXTURE-ARTIFACT-v1"])

    def test_https_evidence_values_and_existing_limits_are_preserved(self):
        base = "https://example.invalid/"
        good_urls = (
            base, base + "receipt?attempt=1#L2", "https://EXAMPLE.invalid:443/r",
            "https://[::1]:443/r", "https://[2001:db8::1]/r",
            base + "a%20b?q=%23#review", "https://b\u00fccher.example/r",
            base + "x" * (2048 - len(base)),
        )
        for channel, label in (("receipts", "execution receipt"), ("reviews", "review link")):
            for value in good_urls + ("", None, True, 7, "http://example.invalid/r", base + "x" * 2048):
                valid = value in good_urls
                with self.subTest(channel=channel, value=value):
                    tables = {"receipts": copy.deepcopy(RECEIPTS), "reviews": copy.deepcopy(REVIEWS)}
                    tables[channel]["FIXTURE-ARTIFACT-v1"]["url"] = value
                    before = copy.deepcopy(tables)
                    result = evidence_path(FIXTURE_NODE, GRAPH, BINDINGS, **tables)
                    self.assertEqual(result["complete"], valid)
                    self.assertEqual(result["missing"], [] if valid else [label + " not recorded for FIXTURE-ARTIFACT-v1"])
                    if valid:
                        self.assertEqual(result[channel][0]["url"], value)
                    else:
                        self.assertEqual(result[channel], [])
                    self.assertEqual(tables, before)
                    self.assertEqual(result["scientific_effect"], "NONE")

    def test_invalid_numeric_https_hosts_are_missing(self):
        # Regression for AST493-F1's numeric-host residual: letters allowed in
        # the hostname are not enough when a special-URL host ends in a number.
        bad_hosts = (
            "256.0.0.1", "127.0.0.999", "1.2.3.4.5", "1.2.3.256",
            "4294967296", "0x100000000", "example.123", "09",
            "08.0.0.1", "1.16777216", "1.2.65536", "1.256.1",
            "1.2.3.0x100", "0xg.1", "example.0x1", "example.09",
            "1.2.3.4.5.", "1.2.3.4.0", "1.2.3.4.0.", "0x100000000.", "1..2", ".1",
            "\uff11\uff12\uff17.0.0.999", "example\u3002123",
        )
        for channel, label in (("receipts", "execution receipt"), ("reviews", "review link")):
            other = "reviews" if channel == "receipts" else "receipts"
            for host in bad_hosts:
                with self.subTest(channel=channel, host=host):
                    tables = {"receipts": copy.deepcopy(RECEIPTS), "reviews": copy.deepcopy(REVIEWS)}
                    tables[channel]["FIXTURE-ARTIFACT-v1"]["url"] = "https://" + host + "/evidence"
                    before = copy.deepcopy(tables)
                    result = evidence_path(FIXTURE_NODE, GRAPH, BINDINGS, **tables)
                    self.assertFalse(result["complete"])
                    self.assertEqual(result[channel], [])
                    self.assertEqual(result["missing"], [label + " not recorded for FIXTURE-ARTIFACT-v1"])
                    self.assertEqual(result[other][0]["url"], before[other]["FIXTURE-ARTIFACT-v1"]["url"])
                    self.assertEqual(result["scientific_effect"], "NONE")
                    self.assertEqual(tables, before)

    def test_valid_numeric_and_nonnumeric_https_hosts_are_preserved(self):
        good_hosts = (
            "example.invalid", "127.0.0.1", "[2001:db8::1]", "2130706433",
            "0x7f000001", "127.1", "0", "0x", "00", "127.0.0.1.",
            "0xffffffff", "255.255.255.255", "127.0.65535", "127.16777215",
            "1.2.65535", "0.0.0.0", "0177.0.0.1", "0X7f.1",
            "08.example", "example.0xg", "example.1a", "1.2.3.4a",
            "123.example", "\uff11\uff12\uff17\uff0e0\uff0e0\uff0e1",
        )
        for channel in ("receipts", "reviews"):
            for host in good_hosts:
                with self.subTest(channel=channel, host=host):
                    value = "https://" + host + "/evidence?attempt=1#review"
                    tables = {"receipts": copy.deepcopy(RECEIPTS), "reviews": copy.deepcopy(REVIEWS)}
                    tables[channel]["FIXTURE-ARTIFACT-v1"]["url"] = value
                    before = copy.deepcopy(tables)
                    result = evidence_path(FIXTURE_NODE, GRAPH, BINDINGS, **tables)
                    self.assertTrue(result["complete"])
                    self.assertEqual(result["missing"], [])
                    self.assertEqual(result[channel][0]["url"], value)
                    self.assertEqual(result["scientific_effect"], "NONE")
                    self.assertEqual(tables, before)

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
