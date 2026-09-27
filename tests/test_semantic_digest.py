"""Tests for #95 v1.1 semantic_digest / evidence_digest (scientific effect NONE)."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SD = _load("semantic_digest", "tools/semantic_digest.py")
CGA = _load("claims_gate_adapter", "tools/claims_gate_adapter.py")


class SemanticDigestTests(unittest.TestCase):
    def test_statement_change_changes_semantic_digest(self):
        claims = json.loads((ROOT / "claims" / "graph.json").read_text(encoding="utf-8"))
        nid = "OBL-H5-JETMOD"
        record = claims["premises"][nid]
        before = SD.semantic_digest(nid, record)
        mutated = copy.deepcopy(record)
        mutated["statement"] = (mutated.get("statement") or "") + " [digest drift]"
        after = SD.semantic_digest(nid, mutated)
        self.assertNotEqual(before, after)
        self.assertEqual(len(before), 64)
        int(before, 16)  # hex

    def test_manual_fingerprint_alone_is_not_sole_detector_in_adapter(self):
        claims = json.loads((ROOT / "claims" / "graph.json").read_text(encoding="utf-8"))
        old = CGA.claims_to_gate_graph(claims)
        new = copy.deepcopy(old)
        # Leave semantic_digest + source_snapshot identical; only touch a
        # provenance-style fingerprint. Must NOT seed impact by itself.
        new["nodes"]["OBL-H5-JETMOD"]["fingerprint"] = "manual-only-drift"
        impact = CGA.reverse_impact_between(old, new)
        self.assertNotIn("OBL-H5-JETMOD", impact["changed_nodes"])
        self.assertNotIn("OBL-H5-JETMOD", impact["impacted"])

    def test_migration_crosswalks_legacy_fingerprint_without_rewriting_source(self):
        claims_path = ROOT / "claims" / "graph.json"
        before_bytes = claims_path.read_bytes()
        claims = json.loads(before_bytes.decode("utf-8"))
        nid = "D1-v2.2(2)"
        record = claims["claims"][nid]
        migrated = SD.migrate_legacy_fingerprint(
            nid, record, legacy_fingerprint="legacy-manual-fp"
        )
        self.assertEqual(migrated["legacy_fingerprint"], "legacy-manual-fp")
        self.assertIs(migrated["legacy_fingerprint_is_sole_detector"], False)
        self.assertEqual(migrated["semantic_digest"], SD.semantic_digest(nid, record))
        self.assertEqual(migrated["scientific_effect"], "NONE")
        # Source claims file must remain byte-identical (crosswalk, not rewrite).
        self.assertEqual(claims_path.read_bytes(), before_bytes)

    def test_duplicate_typed_edge_fails_closed(self):
        with self.assertRaises(SD.DigestError) as ctx:
            SD.semantic_digest(
                "x",
                {"depends_on": ["A", "A"], "statement": "s"},
            )
        self.assertIn("duplicate", str(ctx.exception))

    def test_required_non_boolean_in_typed_edge_fails_closed(self):
        with self.assertRaises(SD.DigestError) as ctx:
            SD.normalize_typed_edge(
                {"target_id": "A", "required": 0}, default_relation="depends_on"
            )
        self.assertIn("strict boolean", str(ctx.exception))

    def test_evidence_digest_orthogonal_to_semantic(self):
        record = {
            "statement": "same statement",
            "depends_on": ["P"],
            "evidence": [{"repo": "main", "path": "a.py", "commit": "abc"}],
        }
        other = copy.deepcopy(record)
        other["evidence"] = [{"repo": "main", "path": "b.py", "commit": "abc"}]
        self.assertEqual(
            SD.semantic_digest("n", record),
            SD.semantic_digest("n", other),
        )
        self.assertNotEqual(SD.evidence_digest(record), SD.evidence_digest(other))

    def test_digest_is_sha256_of_canonical_payload(self):
        record = {"statement": "s", "depends_on": ["P"], "domain": "d"}
        payload = SD.canonical_semantic_payload("n", record)
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()
        self.assertEqual(SD.semantic_digest("n", record), expected)


if __name__ == "__main__":
    unittest.main()
