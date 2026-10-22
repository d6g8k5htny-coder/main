"""Synthetic fixtures for tools/check_board_links.py (LANES-20261022 Lane 4).

The records below are invented test data. They make no ownership,
acknowledgement or custody claim about any real board or agent.
"""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/check_board_links.py"
_spec = importlib.util.spec_from_file_location("check_board_links", TOOL)
cbl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cbl)

PICKUP = "https://github.com/example/repo/issues/2#issuecomment-200"
RESOLVED = "https://github.com/example/repo/issues/2#issuecomment-201"

BASE = {
    "board-old": {
        "schema_version": 0, "node_id": "board-old", "owner": None,
        "predecessor": None,
        "successor": {"node_id": "board-new", "url": "https://github.com/example/repo/issues/2"},
        "carried_forward": [
            {"id": "item-a", "text": "open review", "source_url": PICKUP, "status": "open"},
            {"id": "item-b", "text": "done", "source_url": PICKUP, "status": "resolved",
             "resolution_url": RESOLVED},
        ],
        "scientific_effect": "NONE",
    },
    "board-new": {
        "schema_version": 0, "node_id": "board-new",
        "owner": {"id": "agent-x", "evidence_url": PICKUP},
        "predecessor": {"node_id": "board-old", "url": "https://github.com/example/repo/issues/1"},
        "successor": None,
        "carried_forward": [
            {"id": "item-a", "text": "open review", "source_url": PICKUP, "status": "open"},
        ],
        "scientific_effect": "NONE",
    },
}


class BoardLinks(unittest.TestCase):
    def run_check(self, records):
        with tempfile.TemporaryDirectory() as d:
            for node, rec in records.items():
                Path(d, f"{node}.json").write_text(json.dumps(rec), encoding="utf-8")
            return cbl.check(d)

    def records(self):
        return copy.deepcopy(BASE)

    def test_consistent_chain_passes(self):
        self.assertEqual(self.run_check(self.records()), [])

    def test_missing_reciprocal_link_fails(self):
        r = self.records()
        r["board-new"]["predecessor"] = None
        errs = self.run_check(r)
        self.assertTrue(any("does not link back" in e for e in errs), errs)

    def test_link_to_absent_record_fails(self):
        r = self.records()
        del r["board-new"]
        errs = self.run_check(r)
        self.assertTrue(any("has no record" in e for e in errs), errs)

    def test_owner_without_pickup_url_fails(self):
        for bad in ({"id": "agent-x"}, {"id": "agent-x", "evidence_url": ""},
                    {"id": "agent-x", "evidence_url": "https://github.com/example/repo/issues/2"}):
            r = self.records()
            r["board-new"]["owner"] = bad
            errs = self.run_check(r)
            self.assertTrue(any("no PICKUP" in e for e in errs), (bad, errs))

    def test_null_owner_is_not_inferred(self):
        r = self.records()
        r["board-new"]["owner"] = None
        self.assertEqual(self.run_check(r), [])

    def test_open_item_dropped_by_successor_fails(self):
        r = self.records()
        r["board-new"]["carried_forward"] = []
        errs = self.run_check(r)
        self.assertTrue(any("'item-a' not carried forward" in e for e in errs), errs)

    def test_carried_open_items_are_preserved(self):
        r = self.records()
        self.assertEqual(self.run_check(r), [])
        new = r["board-new"]["carried_forward"]
        self.assertEqual([(i["id"], i["status"]) for i in new], [("item-a", "open")])
        self.assertEqual(new[0]["text"], r["board-old"]["carried_forward"][0]["text"])

    def test_resolved_without_resolution_url_fails(self):
        r = self.records()
        r["board-new"]["carried_forward"][0] = {"id": "item-a", "status": "resolved"}
        errs = self.run_check(r)
        self.assertTrue(any("without a resolution_url" in e for e in errs), errs)

    def test_cli_exit_status_and_no_writes(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.records()
            r["board-old"]["successor"] = None
            for node, rec in r.items():
                Path(d, f"{node}.json").write_text(json.dumps(rec), encoding="utf-8")
            before = sorted(p.name for p in Path(d).iterdir())
            out = subprocess.run([sys.executable, "-B", "-S", str(TOOL), d],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
            self.assertFalse(json.loads(out.stdout)["ok"])
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), before)


if __name__ == "__main__":
    unittest.main()
