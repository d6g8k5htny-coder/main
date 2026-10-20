"""Offline readback of the dependency atlas's node fingerprints against pinned sources.

Scientific effect: NONE. These controls check byte identity and graph structure.
A digest match is not proof validity, review independence or scientific acceptance;
the page's own "binding unresolved" wording is not changed by anything here.

Inputs, all committed:
- docs/site/dependency-source/GRAPH.json and PROVENANCE.json (the 3 October snapshot);
- docs/site/current.json (the committed inventory observation);
- tests/fixtures/dependency_source_bindings.json, written by
  tools/dependency_source_bindings.py from a Math- checkout at the pinned commit.
"""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from dependency_source_bindings import fingerprint_kind, is_repository_path  # noqa: E402

SITE = ROOT / "docs" / "site"
GRAPH_PATH = SITE / "dependency-source" / "GRAPH.json"
PROVENANCE_PATH = SITE / "dependency-source" / "PROVENANCE.json"
CURRENT_PATH = SITE / "current.json"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "dependency_source_bindings.json"
MATH_REPOSITORY = "d6g8k5htny-coder/Math-"

EXPECTED_NODES = 49
EXPECTED_EDGES = 55
EXPECTED_REQUIRED_EDGES = 42
EXPECTED_CENSUS = {"ABSENT": 3, "blob40": 1, "no_field": 2, "sha256": 23, "symbolic": 20}
EXPECTED_RECORD_ONLY = 15


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class DependencySourceBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph_bytes = GRAPH_PATH.read_bytes()
        cls.graph = json.loads(cls.graph_bytes)
        cls.provenance = load_json(PROVENANCE_PATH)
        cls.current = load_json(CURRENT_PATH)
        cls.fixture = load_json(FIXTURE_PATH)
        cls.inventory = {(row["repository"], row["path"]): row for row in cls.current["sources"]}

    def test_graph_bytes_match_provenance_and_fixture(self):
        digest = hashlib.sha256(self.graph_bytes).hexdigest()
        self.assertEqual(digest, self.provenance["files"]["GRAPH.json"]["sha256"])
        self.assertEqual(len(self.graph_bytes), self.provenance["files"]["GRAPH.json"]["bytes"])
        self.assertEqual(digest, self.fixture["graph_sha256"])
        self.assertEqual(self.fixture["graph_bytes"], len(self.graph_bytes))
        self.assertEqual(self.fixture["pinned_commit"], self.provenance["commit"])
        identity = self.current["graph"]["identity"]
        self.assertEqual(identity["sha256"], digest)
        self.assertEqual(identity["bytes"], len(self.graph_bytes))
        # The inventory's blob id is Git's object name for these same bytes.
        git_blob = hashlib.sha1(b"blob %d\0" % len(self.graph_bytes) + self.graph_bytes).hexdigest()
        self.assertEqual(identity["blob"], git_blob)

    def test_node_and_edge_structure(self):
        nodes = self.graph["nodes"]
        edges = self.graph["edges"]
        self.assertEqual(len(nodes), EXPECTED_NODES)
        self.assertEqual(len(edges), EXPECTED_EDGES)
        self.assertEqual(self.fixture["node_count"], EXPECTED_NODES)
        self.assertEqual(self.fixture["edge_count"], EXPECTED_EDGES)
        pairs = set()
        required = 0
        for edge in edges:
            self.assertIn(edge["from"], nodes, edge)
            self.assertIn(edge["to"], nodes, edge)
            pair = (edge["from"], edge["to"])
            self.assertNotIn(pair, pairs, "duplicate edge")
            pairs.add(pair)
            if edge.get("required") is True:
                required += 1
        self.assertEqual(required, EXPECTED_REQUIRED_EDGES)
        self.assertEqual(len(edges) - required, EXPECTED_EDGES - EXPECTED_REQUIRED_EDGES)

    def test_fingerprint_census_is_pinned(self):
        census: dict[str, int] = {}
        for node in self.graph["nodes"].values():
            kind = fingerprint_kind(node.get("fingerprint"))
            census[kind] = census.get(kind, 0) + 1
        self.assertEqual(census, EXPECTED_CENSUS)
        self.assertEqual(self.fixture["fingerprint_census"], EXPECTED_CENSUS)

    def test_absent_carriers_are_fail_closed(self):
        for node_id, node in self.graph["nodes"].items():
            if node.get("fingerprint") == "ABSENT":
                self.assertEqual(node.get("classification"), "BLOCKED_ABSENT", node_id)
                self.assertFalse(node.get("controlling"), node_id)

    def test_every_digest_node_binds_to_its_pinned_bytes(self):
        file_nodes = self.fixture["file_nodes"]
        digest_nodes = {node_id for node_id, node in self.graph["nodes"].items()
                        if fingerprint_kind(node.get("fingerprint")) in ("sha256", "blob40")}
        self.assertEqual(set(file_nodes), digest_nodes)
        for node_id, pinned in file_nodes.items():
            node = self.graph["nodes"][node_id]
            with self.subTest(node=node_id):
                self.assertEqual(node["source"], pinned["source"])
                self.assertTrue(is_repository_path(pinned["source"]))
                expected = pinned["sha256"] if pinned["kind"] == "sha256" else pinned["blob"]
                self.assertEqual(node["fingerprint"], expected, "fingerprint is not the pinned identity")
                row = self.inventory.get((MATH_REPOSITORY, pinned["source"]))
                self.assertIsNotNone(row, "committed inventory lacks the pinned path")
                self.assertEqual(row["blob"], pinned["blob"], "inventory blob differs from the pinned blob")
                self.assertEqual(row["bytes"], pinned["bytes"], "inventory byte count differs from the pinned bytes")

    def test_record_only_nodes_are_pinned(self):
        # Same rule as the generator: a symbolic fingerprint and no source value
        # (the key absent or null) makes a node record-only.
        record_only = sorted(node_id for node_id, node in self.graph["nodes"].items()
                             if fingerprint_kind(node.get("fingerprint")) == "symbolic" and node.get("source") is None)
        self.assertEqual(record_only, self.fixture["record_only_nodes"])
        self.assertEqual(len(record_only), EXPECTED_RECORD_ONLY)

    def test_directory_sources_report_inventory_drift_without_failing(self):
        # The graph binds two nodes to whole directories without a digest. The pinned
        # file list is compared with the committed inventory and the differences are
        # reported; a difference is information for a reviewer, not a failure here.
        sources = self.fixture["directory_sources"]
        self.assertTrue(sources, "fixture lists no directory source")
        for node_id, pinned in sources.items():
            self.assertTrue(pinned["files"], node_id)
            prefix = pinned["source"]
            observed = {path: row["blob"] for (repo, path), row in self.inventory.items()
                        if repo == MATH_REPOSITORY and path.startswith(prefix)}
            added = sorted(set(observed) - set(pinned["files"]))
            removed = sorted(set(pinned["files"]) - set(observed))
            changed = sorted(path for path in set(observed) & set(pinned["files"])
                             if observed[path] != pinned["files"][path])
            report = {"node": node_id, "source": prefix, "pinned_commit": self.fixture["pinned_commit"],
                      "observation_commit": self.current["graph"]["identity"]["commit"],
                      "added": added, "removed": removed, "changed": changed,
                      "meaning": "inventory rows under a directory-bound source compared with the pinned file list; byte identity only"}
            print("DIRECTORY_SOURCE_REPORT " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
