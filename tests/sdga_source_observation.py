"""Mandatory pinned-source control; run explicitly, outside ordinary test_*.py discovery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import unittest

from test_sdga_scope import ContractCase, FIXTURES, api, blob_id


class PinnedSourceObservation(ContractCase):
    def test_original_raw_inputs_and_all_99_current_source_bytes(self):
        required = ("SDGA_PINNED_MATH_ROOT", "SDGA_OBSERVED_MATH_COMMIT",
                    "SDGA_OBSERVED_MATH_TREE")
        missing = [name for name in required if not os.environ.get(name)]
        self.assertFalse(missing, "Required pinned-source environment unavailable: " + ", ".join(missing))
        root_value = os.environ["SDGA_PINNED_MATH_ROOT"]
        profile = json.loads((FIXTURES / "scope-f532.json").read_text(encoding="utf-8"))
        observed = dict(profile["source"],
                        commit=os.environ["SDGA_OBSERVED_MATH_COMMIT"],
                        tree=os.environ["SDGA_OBSERVED_MATH_TREE"])
        self.assertEqual(observed, profile["source"])
        package = Path(root_value) / profile["source"]["package"]
        raw = {name: (package / name).read_bytes() for name in profile["inputs"]}
        self.assertGreater(len(raw["SOURCE_MANIFEST.json"]), 2 * 1024 * 1024)
        for name, value in raw.items():
            with self.subTest(input=name):
                self.assertEqual(hashlib.sha256(value).hexdigest(), profile["inputs"][name]["sha256"])
                self.assertEqual(blob_id(value), profile["inputs"][name]["git_blob"])
        scope = api.extract_scope(raw, profile, observed)
        self.assertEqual(scope["modules"], profile["modules"])
        self.assertEqual(scope["targets"], profile["targets"])
        sources = {x["path"]: (package / x["path"]).read_bytes() for x in profile["modules"]}
        for row in profile["modules"]:
            self.assertEqual(hashlib.sha256(sources[row["path"]]).hexdigest(), row["source_sha256"])
        self.assertIsNone(api.verify_source_files(scope, sources))
        self.assert_blocked(scope)



if __name__ == "__main__":
    unittest.main()
