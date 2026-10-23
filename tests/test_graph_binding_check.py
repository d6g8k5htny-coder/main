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

    def test_hist_nodes_unbound_historical_without_record(self):
        rc, rep, _ = self.run_v2(self.doc([]))
        self.assertEqual(rc, 0)
        self.assertNotIn("policy_open", rep)
        h = {x["node_id"]: x for x in rep["source_rules"]["hist_nodes"]}
        self.assertEqual(sorted(h), ["hist.carried", "hist.old"])
        for x in h.values():
            self.assertEqual((x["status"], x["bound"]), ("unbound_historical", False))

    def test_hist_node_never_bound_even_with_records(self):
        recs = [self.rec("hist.carried", "art/a.json", b"A"),
                self.rec("hist.old", "art/a.json", b"A", kind="historical")]
        rc, rep, _ = self.run_v2(self.doc(recs))
        self.assertEqual(rc, 0)
        self.assertEqual(rep["counts"]["bound"], 0)
        self.assertEqual(rep["counts"]["historical"], 1)
        h = {x["node_id"]: x for x in rep["source_rules"]["hist_nodes"]}
        self.assertTrue(h["hist.old"]["has_historical_record"])
        self.assertEqual(h["hist.carried"]["demoted_from_bound"], 1)
        for x in h.values():
            self.assertEqual((x["status"], x["bound"]), ("unbound_historical", False))

    def test_directory_source_loose_is_unbound(self):
        rc, rep, _ = self.run_v2(self.doc([], scope=("pkg/*",)))
        d = rep["source_rules"]["directory_sources"]
        self.assertEqual([(x["node_id"], x["status"], x["bound"]) for x in d],
                         [("TEST.dir", "loose_directory_unbound", False)])
        self.assertIn("TEST.dir", rep["unbound_nodes"])

    def test_directory_tree_record_files_still_unbound(self):
        r = {"node_id": "TEST.dir", "kind": "tree", "repository": REPO,
             "artifact_path": "pkg", "commit": C, "blob_sha": C, "roles": ["source"]}
        rc, rep, _ = self.run_v2(self.doc([r], scope=("pkg/*",)))
        self.assertEqual(rc, 0)
        d = rep["source_rules"]["directory_sources"][0]
        self.assertEqual((d["status"], d["record_kinds"]), ("pinned_by_tree_or_files", ["tree"]))
        self.assertEqual(rep["unbound_artifacts"], ["pkg/x.txt"])  # L94: only file records bind

    def test_external_source_never_bound(self):
        recs = [{"node_id": "TEST.ext", "kind": "external_unverified", "url": "external:Math-PR7",
                 "roles": ["source"]},
                {"node_id": "TEST.url", "kind": "external_unverified",
                 "url": "https://github.com/o/r/pull/1", "roles": ["source"]}]
        rc, rep, _ = self.run_v2(self.doc(recs))
        self.assertEqual(rc, 0)
        e = rep["source_rules"]["url_or_external_sources"]
        self.assertEqual([(x["node_id"], x["has_external_unverified_record"], x["bound"])
                          for x in e], [("TEST.ext", True, False), ("TEST.url", True, False)])
        self.assertEqual(rep["counts"]["bound"], 0)
        self.assertEqual(rep["counts"]["external_unverified"], 2)

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

    def test_absent_null_last_seen_commit_kept_honest(self):
        # GB-5 L104 "without faking a pin": null = never committed; still
        # counted with missing, never bound.
        a = {"node_id": "hist.old", "kind": "absent", "repository": REPO,
             "artifact_path": "gone.json", "reason": "never committed",
             "last_seen_commit": None, "roles": ["source"]}
        rc, rep, err = self.run_v2(self.doc([a]))
        self.assertEqual(rc, 0, err)
        self.assertEqual(rep["absent"][0]["last_seen_commit"], None)
        self.assertEqual(rep["bound"], [])
        del a["last_seen_commit"]
        self.assertEqual(self.run_v2(self.doc([a]))[0], 2)  # key still required
        a["last_seen_commit"] = "abc"
        self.assertEqual(self.run_v2(self.doc([a]))[0], 2)  # non-null must be 40-hex

    def test_external_unverified_may_name_repository(self):
        e = {"node_id": "TEST.ext", "kind": "external_unverified", "repository": REPO,
             "url": "external:Math-PR7", "roles": ["source"]}
        rc, rep, err = self.run_v2(self.doc([e]))
        self.assertEqual(rc, 0, err)
        self.assertEqual(rep["bound"], [])

    def test_foreign_node_source_requires_checkout_and_pin(self):
        d = self.doc([])
        d["node_source"]["repository"] = "o/other"
        self.assertEqual(self.run_v2(d)[0], 2)  # no checkout: fail closed
        src = self.root / "src"
        src.mkdir()
        g = lambda *a: subprocess.run(["git", "-C", str(src), *a], check=True,
                                      capture_output=True, text=True).stdout.strip()
        g("init", "-q")
        (src / "g.json").write_text(json.dumps(self.graph))
        g("add", "g.json")
        g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "g")
        pin = g("rev-parse", "HEAD")
        (src / "g.json").write_text(json.dumps({"nodes": {"ONLY.wt": {}}}))  # dirty tree ignored
        d["node_source"]["commit"] = pin
        rc, rep, err = self.run_v2(d, None, "--node-source-checkout", str(src))
        self.assertEqual(rc, 0, err)
        self.assertTrue(rep["node_source_read"]["pinned"])
        self.assertEqual(rep["counts"]["graph_nodes"], 7)
        d["node_source"]["commit"] = "2" * 40  # pin absent from checkout
        self.assertEqual(self.run_v2(d, None, "--node-source-checkout", str(src))[0], 2)

    def test_foreign_repository_records_never_bound(self):
        r = self.rec("TEST.n1", "art/a.json", b"A", repository="o/other")
        rc, rep, _ = self.run_v2(self.doc([r]))
        self.assertEqual(rc, 0)
        self.assertEqual(rep["bound"], [])
        self.assertEqual(rep["other_repository"][0]["node_id"], "TEST.n1")


DATA_525 = "29b7d5969e2cd707debb7449c69a71bbcd39cf63"
MATH_PIN = "dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde"


class Composition525Test(unittest.TestCase):
    """Pinned composition: main#525 data at 29b7d596 + Math- checkout holding
    dbe6c0a1 (set GB_MATH_CHECKOUT). Skips when either commit is unavailable."""

    def test_525_composition(self):
        import os
        repo = TOOL.parents[1]
        math = os.environ.get("GB_MATH_CHECKOUT")
        ok = lambda d, c: subprocess.run(["git", "-C", str(d), "cat-file", "-e", c + "^{commit}"],
                                         capture_output=True).returncode == 0
        if not math or not ok(repo, DATA_525) or not ok(math, MATH_PIN):
            self.skipTest("29b7d596 or Math- dbe6c0a1 unavailable")
        with tempfile.TemporaryDirectory() as t:
            arc = subprocess.run(["git", "-C", str(repo), "archive", DATA_525, "architecture"],
                                 capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", "-C", t], input=arc, check=True)
            f = "architecture/graph-artifact-bindings.v2.json"
            p = subprocess.run([sys.executable, "-B", str(TOOL), "--root", t, "--v2",
                                str(Path(t) / f), "--node-source-checkout", math],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            rep = json.loads(p.stdout)
            self.assertTrue(rep["node_source_read"]["pinned"])
            c = rep["counts"]
            self.assertEqual((c["bindings"], c["graph_nodes"], c["bound"], c["absent"],
                              c["historical"], c["external_unverified"], c["other_repository"],
                              c["unknown_node"], c["missing"], c["stale"]),
                             (21, 49, 0, 3, 7, 2, 9, 0, 0, 0))
            self.assertEqual([a["last_seen_commit"] for a in rep["absent"]], [None] * 3)
            self.assertIn("hist.lemma_closed", rep["unbound_nodes"])
            self.assertEqual(len(rep["unbound_nodes"]), 28)


if __name__ == "__main__":
    unittest.main()
