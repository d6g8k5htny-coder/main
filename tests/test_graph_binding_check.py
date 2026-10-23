"""Tests for tools/graph_binding_check.py (GB-6 proposal). Synthetic fixtures only."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "graph_binding_check.py"


def sha(b):
    return hashlib.sha256(b).hexdigest()


class GraphBindingCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "art").mkdir()
        (self.root / "art" / "a.json").write_bytes(b"A")
        (self.root / "art" / "b.json").write_bytes(b"B")
        self.nodes = self.root / "nodes.json"
        self.nodes.write_text(json.dumps(["TEST.n1", "TEST.n2"]))

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, bindings, *extra, raw=None):
        bf = self.root / "bindings.json"
        bf.write_text(raw if raw is not None else json.dumps(bindings))
        cmd = [sys.executable, "-B", str(TOOL), "--root", str(self.root),
               "--bindings", str(bf), "--graph-nodes", str(self.nodes),
               "--scope", "art/*.json", *extra]
        p = subprocess.run(cmd, capture_output=True, text=True)
        rep = json.loads(p.stdout) if p.stdout else None
        return p.returncode, rep, p.stderr

    def full(self):
        return [
            {"node_id": "TEST.n1", "artifact_path": "art/a.json", "sha256": sha(b"A")},
            {"node_id": "TEST.n2", "artifact_path": "art/b.json", "sha256": sha(b"B")},
        ]

    def test_all_bound_passes(self):
        rc, rep, _ = self.run_tool(self.full())
        self.assertEqual(rc, 0)
        self.assertEqual(rep["status"], "ALL_BOUND")
        self.assertEqual(rep["scientific_effect"], "NONE")

    def test_empty_bindings_fail_both_directions(self):
        rc, rep, _ = self.run_tool([])
        self.assertEqual(rc, 1)
        self.assertEqual(rep["unbound_nodes"], ["TEST.n1", "TEST.n2"])
        self.assertEqual(rep["unbound_artifacts"], ["art/a.json", "art/b.json"])

    def test_report_only_exits_zero(self):
        rc, rep, _ = self.run_tool([], "--report-only")
        self.assertEqual(rc, 0)
        self.assertEqual(rep["status"], "UNBOUND_ITEMS")

    def test_stale_sha_fails(self):
        b = self.full()
        b[0]["sha256"] = sha(b"old")
        rc, rep, _ = self.run_tool(b)
        self.assertEqual(rc, 1)
        self.assertEqual(rep["bindings_stale_sha"][0]["actual_sha256"], sha(b"A"))
        self.assertIn("TEST.n1", rep["unbound_nodes"])

    def test_missing_artifact_and_unknown_node_fail(self):
        b = self.full() + [
            {"node_id": "TEST.n1", "artifact_path": "art/gone.json", "sha256": sha(b"x")},
            {"node_id": "TEST.ghost", "artifact_path": "art/a.json", "sha256": sha(b"A")},
        ]
        rc, rep, _ = self.run_tool(b)
        self.assertEqual(rc, 1)
        self.assertEqual(len(rep["bindings_missing_artifact"]), 1)
        self.assertEqual(len(rep["bindings_unknown_node"]), 1)

    def test_symlink_escape_is_outside_root(self):
        outside = Path(self.tmp.name).parent / ("gb6_out_" + Path(self.tmp.name).name)
        outside.write_bytes(b"X")
        try:
            (self.root / "art" / "link.txt").symlink_to(outside)
            b = self.full() + [{"node_id": "TEST.n1", "artifact_path": "art/link.txt",
                                "sha256": sha(b"X")}]
            rc, rep, _ = self.run_tool(b)
            self.assertEqual(rc, 1)
            self.assertEqual(len(rep["bindings_outside_root"]), 1)
        finally:
            outside.unlink()

    def test_malformed_inputs_exit_2(self):
        cases = [
            '{"not": "a list"}',
            '[{"node_id": "TEST.n1", "artifact_path": "../x", "sha256": "%s"}]' % sha(b"A"),
            '[{"node_id": "TEST.n1", "artifact_path": "/abs", "sha256": "%s"}]' % sha(b"A"),
            '[{"node_id": "TEST.n1", "artifact_path": "art/a.json", "sha256": "ABC"}]',
            '[{"node_id": "TEST.n1", "node_id": "TEST.n2", "artifact_path": "a", "sha256": "%s"}]' % sha(b"A"),
            '[{"node_id": "TEST.n1", "artifact_path": "art/a.json", "sha256": "%s", "extra": 1}]' % sha(b"A"),
            '[NaN]',
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                rc, rep, err = self.run_tool(None, raw=raw)
                self.assertEqual(rc, 2)
                self.assertIsNone(rep)
                self.assertIn("GRAPH_BINDING_INPUT_ERROR", err)

    def test_site_nodes_dir(self):
        site = self.root / "site"
        site.mkdir()
        (site / "TEST.n3.html").write_text("x")
        rc, rep, _ = self.run_tool(self.full(), "--site-nodes", str(site))
        self.assertEqual(rc, 1)
        self.assertEqual(rep["unbound_nodes"], ["TEST.n3"])

    def test_deterministic_output(self):
        a = self.run_tool([], "--report-only")
        b = self.run_tool([], "--report-only")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
