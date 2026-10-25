"""Exact, synthetic, nonauthoritative 1-skeleton topology diagnostic controls."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# A detached/replayed checkout preserves this tests/ and tools/ arrangement.
TOOL = Path(__file__).resolve().parents[1] / "tools" / "evidence_topology_diagnostic.py"


def graph(nodes, edges, **metadata):
    return {"nodes": {node: {"kind": "TEST"} for node in nodes},
            "edges": edges, "schema_version": 1, **metadata}


def edge(a, b, required=True, relation="TEST.depends"):
    return {"from": a, "to": b, "required": required, "relation": relation}


class EvidenceTopologyDiagnostic(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, name, document):
        path = self.root / name
        raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def call(self, source, digest, *extra):
        cmd = [sys.executable, "-B"] + (["-O"] if sys.flags.optimize else []) + [
            "-S", str(TOOL), "--graph", str(source), "--sha256", digest, *map(str, extra)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return p.returncode, json.loads(p.stdout) if p.stdout else None, p.stderr

    def check(self, document):
        src, digest = self.source("graph.json", document)
        rc, data, err = self.call(src, digest)
        self.assertEqual((rc, err), (0, ""))
        self.assertEqual(data["before"]["raw_sha256"], digest)
        self.assertIs(data["scientific_status_authority"], False)
        self.assertEqual(data["scientific_effect"], "NONE")
        return data["before"]

    def reject(self, document):
        path, digest = self.source("bad.json", document)
        rc, result, err = self.call(path, digest)
        self.assertEqual(rc, 2)
        self.assertIsNone(result)
        self.assertEqual(err, "EVIDENCE_TOPOLOGY_REFUSED\n")

    def test_single_vertex_and_no_edges(self):
        a = self.check(graph(["TEST.a"], []))["all_edges"]
        self.assertEqual((a["vertices"], a["edges"], a["beta0"], a["beta1"]), (1, 0, 1, 0))
        self.assertEqual(a["isolated_nodes"], ["TEST.a"])

    def test_two_disconnected_edges(self):
        a = self.check(graph(list("abcd"), [edge("a", "b"), edge("c", "d")]))["all_edges"]
        self.assertEqual((a["beta0"], a["beta1"], a["component_sizes"]), (2, 0, [2, 2]))

    def test_connected_tree(self):
        a = self.check(graph(list("abcd"), [edge("a", "b"), edge("b", "c"), edge("c", "d")]))["all_edges"]
        self.assertEqual((a["beta0"], a["beta1"]), (1, 0))

    def test_undirected_cycle(self):
        a = self.check(graph(list("abc"), [edge("a", "b"), edge("b", "c"), edge("c", "a")]))["all_edges"]
        self.assertEqual((a["beta0"], a["beta1"]), (1, 1))

    def test_parallel_directed_cells_are_not_collapsed(self):
        a = self.check(graph(["a", "b"], [edge("a", "b", relation="different"), edge("b", "a")]))["all_edges"]
        self.assertEqual((a["beta0"], a["beta1"], a["edges"]), (1, 1, 2))

    def test_loop_is_one_cell(self):
        a = self.check(graph(["a"], [edge("a", "a")]))["all_edges"]
        self.assertEqual((a["beta0"], a["beta1"], a["isolated_nodes"]), (1, 1, []))

    def test_required_subgraph_keeps_all_vertices(self):
        a = self.check(graph(list("abcd"), [edge("a", "b"), edge("b", "c", False), edge("c", "a", False)]))
        self.assertEqual((a["all_edges"]["beta0"], a["all_edges"]["beta1"]), (2, 1))
        self.assertEqual((a["required_edges_only"]["beta0"], a["required_edges_only"]["beta1"]), (3, 0))
        self.assertEqual(a["required_edges_only"]["isolated_nodes"], ["c", "d"])

    def test_dynamic_snapshot_changes(self):
        before = graph(["a", "b", "c"], [edge("a", "b")], epoch="old")
        after = graph(["a", "b", "c"], [edge("a", "b"), edge("b", "c"), edge("c", "a", False)], epoch="new")
        p, pd = self.source("a.json", before)
        q, qd = self.source("b.json", after)
        rc, output, err = self.call(p, pd, "--compare", q, "--compare-sha256", qd)
        self.assertEqual((rc, err), (0, ""))
        self.assertEqual(output["delta"]["beta_delta"]["all_edges"], {"beta0": -1, "beta1": 1})
        self.assertEqual(output["delta"]["beta_delta"]["required_edges_only"], {"beta0": -1, "beta1": 0})
        self.assertEqual(len(output["delta"]["typed_edges_added"]), 2)
        self.assertTrue(output["delta"]["other_metadata_changed"])

    def test_deleted_node_and_edge(self):
        p, pd = self.source("a.json", graph(["a", "b"], [edge("a", "b")]))
        q, qd = self.source("b.json", graph(["b"], []))
        rc, report, _ = self.call(p, pd, "--compare", q, "--compare-sha256", qd)
        self.assertEqual(rc, 0)
        self.assertEqual(report["delta"]["nodes_removed"], ["a"])
        self.assertEqual(len(report["delta"]["typed_edges_removed"]), 1)

    def test_node_metadata_changes_not_scientifically_interpreted(self):
        a = graph(["a"], [])
        b = graph(["a"], [])
        b["nodes"]["a"]["classification"] = "TEST.FORGED_ACCEPT"
        p, pd = self.source("a.json", a)
        q, qd = self.source("b.json", b)
        rc, r, _ = self.call(p, pd, "--compare", q, "--compare-sha256", qd)
        self.assertEqual(rc, 0)
        self.assertEqual(r["delta"]["node_attributes_changed"], ["a"])
        self.assertEqual((r["before"]["all_edges"]["beta0"], r["after"]["all_edges"]["beta0"]), (1, 1))
        self.assertNotIn("classification", r)

    def compare_documents(self, before, after):
        sources = []
        for name, document in (("before.json", before), ("after.json", after)):
            # Preserve insertion order so key-order controls reach the real CLI.
            raw = json.dumps(document, separators=(",", ":")).encode("utf-8")
            path = self.root / name
            path.write_bytes(raw)
            sources.append((path, hashlib.sha256(raw).hexdigest()))
        (p, pd), (q, qd) = sources
        rc, report, err = self.call(p, pd, "--compare", q, "--compare-sha256", qd)
        self.assertEqual((rc, err), (0, ""))
        self.assertEqual(report["before"]["raw_sha256"], pd)
        self.assertEqual(report["after"]["raw_sha256"], qd)
        self.assertEqual(report["scientific_effect"], "NONE")
        self.assertIs(report["scientific_status_authority"], False)
        self.assertEqual(report["before"]["all_edges"], report["after"]["all_edges"])
        self.assertEqual(report["before"]["required_edges_only"],
                         report["after"]["required_edges_only"])
        self.assertEqual(report["delta"]["nodes_added"], [])
        self.assertEqual(report["delta"]["nodes_removed"], [])
        self.assertEqual(report["delta"]["typed_edges_added"], [])
        self.assertEqual(report["delta"]["typed_edges_removed"], [])
        return report["delta"]

    def test_boolean_number_node_changes_are_reported(self):
        for old, new in ((True, 1), (False, 0), (True, 1.0), (False, 0.0)):
            for left, right in ((old, new), (new, old)):
                with self.subTest(left=left, right=right):
                    before, after = graph(["a", "b"], []), graph(["a", "b"], [])
                    before["nodes"]["a"]["value"] = left
                    after["nodes"]["a"]["value"] = right
                    delta = self.compare_documents(before, after)
                    self.assertEqual(delta["node_attributes_changed"], ["a"])
                    self.assertIs(delta["other_metadata_changed"], False)

    def test_nested_boolean_number_node_changes_are_reported(self):
        for old, new in ((True, 1), (False, 0), (True, 1.0), (False, 0.0)):
            for left, right in ((old, new), (new, old)):
                with self.subTest(left=left, right=right):
                    before, after = graph(["a", "b"], []), graph(["a", "b"], [])
                    before["nodes"]["b"]["value"] = {"outer": [{"inner": [left]}]}
                    after["nodes"]["b"]["value"] = {"outer": [{"inner": [right]}]}
                    delta = self.compare_documents(before, after)
                    self.assertEqual(delta["node_attributes_changed"], ["b"])
                    self.assertIs(delta["other_metadata_changed"], False)

    def test_boolean_number_metadata_changes_are_reported(self):
        for old, new in ((True, 1), (False, 0), (True, 1.0), (False, 0.0)):
            for left, right in ((old, new), (new, old)):
                with self.subTest(left=left, right=right):
                    delta = self.compare_documents(graph(["a"], [], value=left),
                                                   graph(["a"], [], value=right))
                    self.assertEqual(delta["node_attributes_changed"], [])
                    self.assertIs(delta["other_metadata_changed"], True)

    def test_nested_boolean_number_metadata_changes_are_reported(self):
        for old, new in ((True, 1), (False, 0), (True, 1.0), (False, 0.0)):
            for left, right in ((old, new), (new, old)):
                with self.subTest(left=left, right=right):
                    before = graph(["a"], [], value=[{"outer": [left]}])
                    after = graph(["a"], [], value=[{"outer": [right]}])
                    delta = self.compare_documents(before, after)
                    self.assertEqual(delta["node_attributes_changed"], [])
                    self.assertIs(delta["other_metadata_changed"], True)

    def test_parsed_value_equivalence_does_not_invent_changes(self):
        cases = ((True, True), (False, False), (None, None), ([], []), ({}, {}),
                 (1, 1.0), (0, -0.0), (1.0, 1), (-0.0, 0),
                 ({"a": [True, None, 1], "b": {"x": False}},
                  {"b": {"x": False}, "a": [True, None, 1.0]}))
        for left, right in cases:
            with self.subTest(left=left, right=right):
                before, after = graph(["a"], [], value=left), graph(["a"], [], value=right)
                before["nodes"]["a"]["value"] = left
                after["nodes"]["a"]["value"] = right
                delta = self.compare_documents(before, after)
                self.assertEqual(delta["node_attributes_changed"], [])
                self.assertIs(delta["other_metadata_changed"], False)

    def test_other_parsed_value_changes_remain_reported(self):
        cases = ((1, 2), (None, False), ("1", 1), ([], {}),
                 ([1], [1, 2]), ([1, 2], [2, 1]),
                 ({"a": None}, {"b": None}), ({"a": []}, {"a": [], "b": 0}),
                 (True, False), (False, True), (None, 0), ("a", "A"),
                 ({}, []), ({"x": 0}, ["x"]), ("ab", ["a", "b"]),
                 ([1, 2], [1]), ([], [1]),
                 ({"a": [], "b": 0}, {"a": []}), ({}, {"a": None}))
        for left, right in cases:
            with self.subTest(left=left, right=right):
                before, after = graph(["a"], [], value=left), graph(["a"], [], value=right)
                before["nodes"]["a"]["value"] = left
                after["nodes"]["a"]["value"] = right
                delta = self.compare_documents(before, after)
                self.assertEqual(delta["node_attributes_changed"], ["a"])
                self.assertIs(delta["other_metadata_changed"], True)

    def test_added_node_is_reported_without_attribute_change(self):
        p, pd = self.source("a.json", graph(["a"], []))
        q, qd = self.source("b.json", graph(["a", "b"], []))
        rc, report, err = self.call(p, pd, "--compare", q, "--compare-sha256", qd)
        self.assertEqual((rc, err), (0, ""))
        delta = report["delta"]
        self.assertEqual((delta["nodes_added"], delta["nodes_removed"]), (["b"], []))
        self.assertEqual(delta["node_attributes_changed"], [])

    def test_top_level_metadata_keys_are_compared_by_name(self):
        for old, new, changed in (({"epoch": 1}, {"era": 1}, True),
                                 ({"epoch": 1, "era": 2}, {"era": 2, "epoch": 1}, False)):
            with self.subTest(old=old, new=new):
                delta = self.compare_documents(graph(["a"], [], **old),
                                               graph(["a"], [], **new))
                self.assertEqual(delta["node_attributes_changed"], [])
                self.assertIs(delta["other_metadata_changed"], changed)

    def test_deeply_nested_parsed_values_keep_comparison_semantics(self):
        for right, changed in ((True, False), (1, True)):
            with self.subTest(changed=changed):
                left = True
                for _ in range(300):
                    left, right = [left], [right]
                delta = self.compare_documents(graph(["a"], [], value=left),
                                               graph(["a"], [], value=right))
                self.assertEqual(delta["node_attributes_changed"], [])
                self.assertIs(delta["other_metadata_changed"], changed)

    def test_no_comparison_without_both_flags(self):
        p, pd = self.source("a.json", graph(["a"], []))
        q, qd = self.source("b.json", graph(["b"], []))
        for args in (("--compare", q), ("--compare-sha256", qd)):
            with self.subTest(args=args):
                rc, out, err = self.call(p, pd, *args)
                self.assertEqual((rc, out, err), (2, None, "EVIDENCE_TOPOLOGY_REFUSED\n"))

    def test_wrong_source_digest_refuses(self):
        p, pd = self.source("a.json", graph(["a"], []))
        rc, out, err = self.call(p, "0" * 64)
        self.assertEqual((rc, out, err), (2, None, "EVIDENCE_TOPOLOGY_REFUSED\n"))

    def test_wrong_comparison_digest_refuses(self):
        p, pd = self.source("a.json", graph(["a"], []))
        q, qd = self.source("b.json", graph(["b"], []))
        rc, out, err = self.call(p, pd, "--compare", q, "--compare-sha256", "0"*64)
        self.assertEqual((rc, out, err), (2, None, "EVIDENCE_TOPOLOGY_REFUSED\n"))

    def test_duplicate_graph_json_object_key_refuses(self):
        p = self.root / "raw.json"
        raw = b'{"nodes":{"a":{}},"edges":[],"edges":[]}'
        p.write_bytes(raw)
        rc, out, err = self.call(p, hashlib.sha256(raw).hexdigest())
        self.assertEqual((rc, out, err), (2, None, "EVIDENCE_TOPOLOGY_REFUSED\n"))

    def test_unknown_endpoint_and_duplicate_edge_refuse(self):
        self.reject(graph(["a"], [edge("a", "missing")]))
        self.reject(graph(["a", "b"], [edge("a", "b"), edge("a", "b")]))

    def test_reject_nonboolean_edge_required_and_unknown_field(self):
        item = edge("a", "b"); item["required"] = 1
        self.reject(graph(["a", "b"], [item]))
        item = edge("a", "b"); item["extra"] = "unknown"
        self.reject(graph(["a", "b"], [item]))

    def test_reject_nonfinite_and_bare_empty_nodes(self):
        self.reject(graph([], []))
        p = self.root / "nan.json"
        raw = b'{"nodes":{"a":{}},"edges":[],"metadata":NaN}'
        p.write_bytes(raw)
        rc, out, err = self.call(p, hashlib.sha256(raw).hexdigest())
        self.assertEqual((rc, out, err), (2, None, "EVIDENCE_TOPOLOGY_REFUSED\n"))
        self.reject(graph(["a"], [], nan=float("inf")))

    def test_source_size_limit(self):
        p = self.root / "big.json"
        raw = b" " * (4 * 1024 * 1024 + 1)
        p.write_bytes(raw)
        rc, out, err = self.call(p, hashlib.sha256(raw).hexdigest())
        self.assertEqual((rc, out, err), (2, None, "EVIDENCE_TOPOLOGY_REFUSED\n"))

    def test_deterministic_output(self):
        p, pd = self.source("a.json", graph(["z", "a", "b"], [edge("z", "a"), edge("b", "a")]))
        x = self.call(p, pd)
        y = self.call(p, pd)
        self.assertEqual(x, y)
        self.assertEqual(x[0], 0)
        self.assertEqual(x[1]["before"]["all_edges"]["components"], [["a", "b", "z"]])


if __name__ == "__main__":
    unittest.main()
