"""Pin SIDE24's mathematical source, computation, regression, and result artifacts.

Static identity regression for the public graph binding registry. This test does
not execute the coefficient calculator, certify a theorem, or fetch remote bytes.
The actual pinned Math- bytes must be checked separately before adopting records.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ROOT / "architecture" / "graph-artifact-bindings.v2.json"
NODE = "math.side24-coefficient"
MATH_REPO = "d6g8k5htny-coder/Math-"
NODE_PIN = "dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde"

# (git blob, raw-file SHA-256, typed evidence role)
EXPECTED = {
    "coefficients/side24_v1/PROOF.md": (
        "44b66f04f89fcd87383b3603fa69f1feb64cdddd",
        "c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769",
        ("source",),
    ),
    "coefficients/side24_v1/coefficient.py": (
        "c2d3ff339f2b9953ec08e676240cdc5edcb17a3e",
        "03ae6d0f15160cb681f9bbb19a84dc861c3a0db3e881291ea621061cc90c86ab",
        ("computation",),
    ),
    "coefficients/side24_v1/test_coefficient.py": (
        "fa9beaab7d6fcb4f0a0c5a91c6d80b455c98a912",
        "81689b5394e2e949058929570ec072ccb1d4fe93cd0d36a4b258431547d02dd3",
        ("computation",),
    ),
    "coefficients/side24_v1/ENCLOSURE.json": (
        "57af39a05e14ed0ba8ebc00a9b4aca4dffb067c7",
        "72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811",
        ("computation",),
    ),
}


class Side24SourceImplementationBindings(unittest.TestCase):
    def test_one_existing_node_exact_four_object_lineage(self):
        doc = json.loads(BINDINGS.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema_version"], 2)
        self.assertEqual(doc["node_source"]["repository"], MATH_REPO)
        self.assertEqual(doc["node_source"]["commit"], NODE_PIN)

        records = [r for r in doc["bindings"] if r.get("node_id") == NODE]
        self.assertEqual(len(records), 4, "one existing node, four distinct artifacts")
        observed = {}
        for r in records:
            self.assertEqual(r["kind"], "file")
            self.assertEqual(r["repository"], MATH_REPO)
            self.assertEqual(r["commit"], NODE_PIN)
            self.assertNotIn("artifact_id", r)  # no invented Artifact Index row
            self.assertNotIn("scientific_status", r)
            path = r["artifact_path"]
            self.assertNotIn(path, observed, "duplicate artifact path")
            observed[path] = (r["blob_sha"], r["sha256"], tuple(r["roles"]))
        self.assertEqual(observed, EXPECTED)


if __name__ == "__main__":
    unittest.main()
