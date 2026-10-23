"""Structural checks for architecture/graph-artifact-bindings.v2.json (GB-5 v2 schema).

Report-level validation only. It checks shape, kinds, pins and node coverage
against the declared node_source. It never fetches the network and never treats a
binding as acceptance, review or discharge evidence. Scientific effect NONE.
"""
from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "architecture" / "graph-artifact-bindings.v2.json"
V1 = ROOT / "architecture" / "public-artifact-index-bindings.json"
SITE = ROOT / "docs" / "site" / "current.json"

HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
ROLES = {"source", "review", "kernel", "computation", "alignment", "teaching-source"}
KINDS = {"file", "tree", "historical", "absent", "external_unverified"}
BASE = {"node_id", "kind", "repository", "roles"}
FIELDS = {
    "file": BASE | {"artifact_path", "commit", "blob_sha", "sha256"},
    "historical": BASE | {"artifact_path", "commit", "blob_sha", "sha256"},
    "tree": BASE | {"artifact_path", "commit", "blob_sha"},
    "absent": BASE | {"artifact_path", "reason", "last_seen_commit"},
    "external_unverified": BASE | {"url"},
}
PROTECTED_NODES = {"hist.lemma_closed"}
# GB-7b scope: the 22 site nodes routed to GB-7b on main#475 (no in-repo file source).
GB7B_NODES = {
    "hist.ALLCELL-FDZ-Q4", "hist.CH-LIFT", "hist.CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip",
    "hist.ENV-RESCOV", "hist.LOGQ-TAIL", "hist.OBL-H5-JETMOD", "hist.Piece-2-annulus",
    "hist.SYM-Fw-jet", "hist.allcell_fdz_enclosures.json", "hist.lemma_closed", "hist.rnu_env.py",
    "math.rn-region.fixed-annulus-window", "math.rn-region.fixed-remote",
    "math.rn-region.intermediate-r-to-rho", "math.rn-region.mesoscopic-scaled-annulus",
    "math.rn-region.pin-collision", "math.rn-region.witness-collision",
    "regional.fixed-annulus.high-jet-route", "eng.hard-gate", "eng.main-pr87-crosswalk",
    "math.d1-component.reconciliation-record", "math.rn-mesoscopic-reduction",
}

# GB-7e scope (main#475 6368882683): math.d5-component nodes bound file-kind to the
# Math- source named by node_source GRAPH.json at its pinned commit. remote-window-proof
# is deliberately excluded (its source file is the GB-7c FC-10 candidate for
# math.rn-fixed-remote-window, 522/6364026501).
GB7E_NODES = {
    "math.d5-component." + n for n in (
        "annulus-proof", "annulus-review", "collar-proof", "collar-review",
        "i5-review-claude", "i5-review-xai", "intermediate-window-proof",
        "offpin-second-moment-review", "punctured-pin-continuum-check",
        "punctured-pin-continuum-review", "punctured-pin-proof", "punctured-pin-review-111")
}
GB7E_EXCLUDED = {"math.d5-component.remote-window-proof"}


def _pairs_no_dupes(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def _valid_path(p):
    return (isinstance(p, str) and p and not p.startswith("/") and "\\" not in p
            and len(p.encode("utf-8")) <= 1024
            and not any(ord(c) < 32 or ord(c) == 127 for c in p)
            and all(s not in ("", ".", "..") for s in p.split("/")))


class GraphArtifactBindingsV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(V2.read_text(encoding="utf-8"), object_pairs_hook=_pairs_no_dupes)
        site = json.loads(SITE.read_text(encoding="utf-8"))
        cls.site_nodes = set(site["graph"]["data"]["nodes"])

    def test_root_shape(self):
        self.assertEqual(set(self.doc), {"schema_version", "node_source", "scope", "bindings"})
        self.assertEqual(self.doc["schema_version"], 2)
        ns = self.doc["node_source"]
        self.assertEqual(set(ns), {"kind", "repository", "path", "commit"})
        self.assertIn(ns["kind"], {"graph-readback", "site-projection"})
        self.assertRegex(ns["repository"], REPO)
        self.assertTrue(_valid_path(ns["path"]))
        self.assertRegex(ns["commit"], HEX40)
        scope = self.doc["scope"]
        self.assertEqual(scope, sorted(set(scope)))
        self.assertFalse(any(s.startswith("tools/") for s in scope))

    def test_records(self):
        keys = []
        for rec in self.doc["bindings"]:
            with self.subTest(node=rec.get("node_id")):
                kind = rec["kind"]
                self.assertIn(kind, KINDS)
                self.assertEqual(set(rec), FIELDS[kind])
                self.assertRegex(rec["node_id"], NODE)
                self.assertRegex(rec["repository"], REPO)
                self.assertEqual(rec["roles"], sorted(set(rec["roles"])))
                self.assertTrue(rec["roles"] and set(rec["roles"]) <= ROLES)
                if "artifact_path" in rec:
                    self.assertTrue(_valid_path(rec["artifact_path"]))
                if kind in {"file", "historical", "tree"}:
                    self.assertRegex(rec["commit"], HEX40)
                    self.assertRegex(rec["blob_sha"], HEX40)
                if kind in {"file", "historical"}:
                    self.assertRegex(rec["sha256"], HEX64)
                if kind == "absent":
                    self.assertTrue(isinstance(rec["reason"], str) and rec["reason"].strip())
                    lsc = rec["last_seen_commit"]
                    self.assertTrue(lsc is None or HEX40.match(lsc))
                if kind == "external_unverified":
                    self.assertTrue(rec["url"].startswith("https://"))
                keys.append((rec["node_id"], rec["repository"], rec.get("artifact_path", "")))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(keys, sorted(keys))

    def test_nodes_come_from_node_source(self):
        bound = {r["node_id"] for r in self.doc["bindings"]}
        self.assertLessEqual(bound, self.site_nodes)

    def test_gb7b_coverage_and_protected_untouched(self):
        self.assertLessEqual(GB7B_NODES, self.site_nodes)
        bound = {r["node_id"] for r in self.doc["bindings"]}
        self.assertEqual(GB7B_NODES - PROTECTED_NODES - bound, set())
        self.assertFalse(bound & PROTECTED_NODES)

    def test_gb7e_d5_components_are_pinned_files(self):
        self.assertLessEqual(GB7E_NODES | GB7E_EXCLUDED, self.site_nodes)
        ns = self.doc["node_source"]
        recs = {r["node_id"]: r for r in self.doc["bindings"] if r["node_id"] in GB7E_NODES}
        self.assertEqual(set(recs), GB7E_NODES)
        for node, rec in recs.items():
            with self.subTest(node=node):
                self.assertEqual(rec["kind"], "file")
                self.assertEqual(rec["repository"], ns["repository"])
                self.assertEqual(rec["commit"], ns["commit"])
        bound = {r["node_id"] for r in self.doc["bindings"]}
        self.assertFalse(bound & GB7E_EXCLUDED)

    def test_no_kind_is_promoted_to_bound(self):
        # Only file/tree can ever evaluate as bound; GB-7b binds no main-repo file,
        # so every in-repo record here is historical (verified below when objects exist).
        for rec in self.doc["bindings"]:
            if rec["kind"] == "historical" and rec["repository"] == "d6g8k5htny-coder/main":
                exists = subprocess.run(
                    ["git", "-C", str(ROOT), "cat-file", "-e", rec["blob_sha"]],
                    capture_output=True).returncode == 0
                if not exists:
                    continue  # shallow clone: cannot verify bytes; never treated as bound
                data = subprocess.run(["git", "-C", str(ROOT), "cat-file", "blob", rec["blob_sha"]],
                                      capture_output=True, check=True).stdout
                import hashlib
                self.assertEqual(hashlib.sha256(data).hexdigest(), rec["sha256"])
                head = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e",
                                       "HEAD:" + rec["artifact_path"]], capture_output=True)
                self.assertNotEqual(head.returncode, 0, "historical path reappeared at HEAD")

    def test_v1_file_untouched(self):
        self.assertEqual(json.loads(V1.read_text(encoding="utf-8")), [])


if __name__ == "__main__":
    unittest.main()
