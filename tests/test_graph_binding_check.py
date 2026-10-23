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

    def _raw(self, *args):
        bf = self.root / "bindings.json"
        bf.write_text(json.dumps(self.full()))
        cmd = [sys.executable, "-B", str(TOOL), "--root", str(self.root),
               "--bindings", str(bf), *args]
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

    # F1: empty or missing node sets fail closed.
    def test_f1_missing_site_nodes_dir_exits_2(self):
        rc, out, err = self._raw("--site-nodes", str(self.root / "nope"))
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("GRAPH_BINDING_INPUT_ERROR", err)

    def test_f1_empty_graph_nodes_exits_2_even_report_only(self):
        empty = self.root / "empty.json"
        empty.write_text("[]")
        for extra in ([], ["--report-only"]):
            with self.subTest(extra=extra):
                rc, out, _ = self._raw("--graph-nodes", str(empty), *extra)
                self.assertEqual((rc, out), (2, ""))

    def test_f1_empty_site_nodes_dir_exits_2(self):
        (self.root / "emptysite").mkdir()
        rc, out, _ = self._raw("--site-nodes", str(self.root / "emptysite"))
        self.assertEqual((rc, out), (2, ""))

    def test_f1_scope_matching_nothing_fails(self):
        rc, rep, _ = self.run_tool(self.full(), "--scope", "nomatch/*.json")
        self.assertEqual(rc, 1)
        self.assertEqual(rep["empty_scopes"], ["nomatch/*.json"])
        self.assertEqual(rep["status"], "UNBOUND_ITEMS")

    # F2/F3: scope patterns must stay inside root and exit 2 when malformed.
    def test_f2_f3_bad_scope_exits_2_without_traceback(self):
        (self.root.parent / ("gb6_sib_" + self.root.name)).write_text("x")
        try:
            for pat in ("../*", "art/../../*", "/etc/*", "", "art//*", "art\\*"):
                with self.subTest(pat=pat):
                    rc, out, err = self._raw("--graph-nodes", str(self.nodes),
                                             "--scope", pat)
                    self.assertEqual((rc, out), (2, ""))
                    self.assertIn("GRAPH_BINDING_INPUT_ERROR", err)
                    self.assertNotIn("Traceback", err)
        finally:
            (self.root.parent / ("gb6_sib_" + self.root.name)).unlink()

    def test_f2_scope_symlink_escape_reported(self):
        outside = self.root.parent / ("gb6_scope_out_" + self.root.name)
        outside.write_bytes(b"X")
        try:
            (self.root / "art" / "esc.json").symlink_to(outside)
            rc, rep, _ = self.run_tool(self.full())
            self.assertEqual(rc, 1)
            self.assertEqual(rep["scope_outside_root"], ["art/esc.json"])
            self.assertNotIn("art/esc.json", rep["unbound_artifacts"])
        finally:
            outside.unlink()

    def test_deterministic_output(self):
        a = self.run_tool([], "--report-only")
        b = self.run_tool([], "--report-only")
        self.assertEqual(a, b)


def blob(b):
    return hashlib.sha1(b"blob %d\x00" % len(b) + b).hexdigest()


C = "1" * 40
REPO = "d6g8k5htny-coder/main"


class GraphBindingV2ReportTest(unittest.TestCase):
    """v2 contract (GB-5, e1d1ebc2): report mode only."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "art").mkdir()
        (self.root / "art" / "a.json").write_bytes(b"A")
        (self.root / "art" / "b.json").write_bytes(b"B")
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "x.txt").write_bytes(b"x")
        self.graph = {"graph": {"data": {"nodes": {
            "TEST.n1": {"source": "art/a.json"},
            "TEST.n2": {"source": "art/b.json"},
            "TEST.dir": {"source": "pkg/"},
            "TEST.url": {"source": "https://github.com/o/r/pull/1"},
            "TEST.ext": {"source": "external:Math-PR7"},
            "hist.old": {},
            "hist.carried": {"source": "art/a.json"},
        }}}}
        (self.root / "g.json").write_text(json.dumps(self.graph))

    def tearDown(self):
        self.tmp.cleanup()

    def rec(self, node, path, data, **kw):
        r = {"node_id": node, "kind": "file", "repository": REPO, "artifact_path": path,
             "commit": C, "blob_sha": blob(data), "sha256": sha(data), "roles": ["source"]}
        r.update(kw)
        return r

    def doc(self, bindings, scope=("art/*.json",)):
        return {"schema_version": 2,
                "node_source": {"kind": "site-projection", "repository": REPO,
                                "path": "g.json", "commit": C},
                "scope": sorted(scope), "bindings": bindings}

    def run_v2(self, doc, raw=None, *extra):
        f = self.root / "v2.json"
        f.write_text(raw if raw is not None else json.dumps(doc))
        p = subprocess.run([sys.executable, "-B", str(TOOL), "--root", str(self.root),
                            "--v2", str(f), *extra], capture_output=True, text=True)
        return p.returncode, (json.loads(p.stdout) if p.stdout else None), p.stderr

    def test_bound_and_ref_format(self):
        rc, rep, _ = self.run_v2(self.doc([self.rec("TEST.n1", "art/a.json", b"A")]))
        self.assertEqual(rc, 0)
        self.assertEqual(rep["mode"], "report")
        self.assertEqual(rep["bound"][0]["ref"], f"{REPO}:art/a.json@{C}#{blob(b'A')}")
        self.assertEqual(rep["unbound_artifacts"], ["art/b.json"])
        self.assertIn("TEST.n2", rep["unbound_nodes"])

    def test_stale_still_exit_zero(self):
        r = self.rec("TEST.n1", "art/a.json", b"OLD")
        rc, rep, _ = self.run_v2(self.doc([r]))
        self.assertEqual(rc, 0)
        self.assertEqual(rep["counts"]["stale"], 1)
        self.assertEqual(rep["stale"][0]["actual_blob_sha"], blob(b"A"))

    def test_missing_unknown_and_other_kinds(self):
        recs = [
            self.rec("TEST.n1", "art/gone.json", b"A"),
            self.rec("TEST.n2", "art/b.json", b"B", repository="o/other"),
            {"node_id": "TEST.url", "kind": "external_unverified", "url": "https://x/1",
             "roles": ["review"]},
            self.rec("ZZ.nope", "art/a.json", b"A"),
            {"node_id": "hist.carried", "kind": "absent", "repository": REPO,
             "artifact_path": "old.json", "reason": "deleted", "last_seen_commit": C,
             "roles": ["source"]},
            self.rec("hist.old", "art/a.json", b"A", kind="historical"),
        ]
        recs.sort(key=lambda r: (r["node_id"], r.get("repository", ""),
                                 r.get("artifact_path", r.get("url", ""))))
        rc, rep, _ = self.run_v2(self.doc(recs))
        self.assertEqual(rc, 0)
        c = rep["counts"]
        self.assertEqual((c["missing"], c["other_repository"], c["external_unverified"],
                          c["unknown_node"], c["absent"], c["historical"], c["bound"]),
                         (1, 1, 1, 1, 1, 1, 0))

    def test_policy_open_fields_listed_not_judged(self):
        rc, rep, _ = self.run_v2(self.doc([]))
        self.assertEqual(rc, 0)
        pol = rep["policy_open"]
        self.assertEqual([d["node_id"] for d in pol["directory_sources"]], ["TEST.dir"])
        self.assertEqual(pol["hist_nodes_without_source"], ["hist.old"])
        self.assertEqual([d["node_id"] for d in pol["url_or_external_sources"]],
                         ["TEST.ext", "TEST.url"])
        # listing does not change bound/unbound state: all still unbound
        for n in ("TEST.dir", "TEST.url", "TEST.ext", "hist.old"):
            self.assertIn(n, rep["unbound_nodes"])
        self.assertEqual(rep["counts"]["bound"], 0)

    def test_tree_kind_without_git_is_unverifiable(self):
        r = {"node_id": "TEST.dir", "kind": "tree", "repository": REPO,
             "artifact_path": "pkg", "commit": C, "blob_sha": C, "roles": ["source"]}
        rc, rep, _ = self.run_v2(self.doc([r]))
        self.assertEqual(rc, 0)
        self.assertEqual(rep["counts"]["tree_unverifiable"], 1)

    def test_malformed_exit_two(self):
        bad = [
            dict(self.doc([]), schema_version=1),
            dict(self.doc([]), extra=1),
            self.doc([self.rec("TEST.n1", "../a.json", b"A")]),
            self.doc([self.rec("TEST.n1", "art/a.json", b"A", roles=[])]),
            self.doc([self.rec("TEST.n1", "art/a.json", b"A", roles=["boss"])]),
            self.doc([self.rec("TEST.n1", "art/a.json", b"A", blob_sha="ABC")]),
            self.doc([self.rec("TEST.n2", "art/b.json", b"B"),
                      self.rec("TEST.n1", "art/a.json", b"A")]),
            self.doc([self.rec("TEST.n1", "art/a.json", b"A")] * 2),
            self.doc([], scope=("/etc/*",)),
        ]
        for d in bad:
            rc, rep, err = self.run_v2(d)
            self.assertEqual(rc, 2, d)
            self.assertIn("GRAPH_BINDING_INPUT_ERROR", err)
        rc, _, _ = self.run_v2(None, raw='{"schema_version":2,"schema_version":2}')
        self.assertEqual(rc, 2)

    def test_empty_node_source_fails_closed(self):
        (self.root / "g.json").write_text(json.dumps({"nodes": {}}))
        rc, _, _ = self.run_v2(self.doc([]))
        self.assertEqual(rc, 2)

    def test_v2_rejects_mixed_v1_args(self):
        rc, _, err = self.run_v2(self.doc([]), None, "--scope", "art/*.json")
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
