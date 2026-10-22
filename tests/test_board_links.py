"""Synthetic fixtures for tools/check_board_links.py (LANES-20261022 Lane 4).

The records below are invented test data. They make no ownership,
acknowledgement or custody claim about any real board or agent.
"""

import contextlib
import copy
import importlib.util
import io
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

PICKUP = "https://github.com/d6g8k5htny-coder/main/issues/2#issuecomment-200"
RESOLVED = "https://github.com/d6g8k5htny-coder/main/issues/2#issuecomment-201"
REVIEW = "https://github.com/d6g8k5htny-coder/Math-/pull/3#pullrequestreview-300"

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

    def assert_rejected(self, records, needle):
        with tempfile.TemporaryDirectory() as d:
            for node, rec in records.items():
                Path(d, f"{node}.json").write_text(json.dumps(rec), encoding="utf-8")
            errs = cbl.check(d)
            self.assertTrue(any(needle in e for e in errs), errs)
            with contextlib.redirect_stdout(io.StringIO()):
                code = cbl.main(["check_board_links.py", d])
            self.assertEqual(code, 1, errs)

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

    def test_carried_item_without_source_url_fails(self):
        for mutate in (
            lambda item: item.pop("source_url"),
            lambda item: item.__setitem__("source_url", ""),
            lambda item: item.__setitem__("source_url", "https://example.com/notes"),
        ):
            r = self.records()
            mutate(r["board-new"]["carried_forward"][0])
            self.assert_rejected(r, "source_url")

    def test_comment_url_pattern(self):
        rejected = (
            "https://github.com/example/main/issues/2#issuecomment-200",
            "https://github.com/d6g8k5htny-coder/other/issues/2#issuecomment-200",
            "https://github.com/d6g8k5htny-coder/main/issues/2",
        )
        for url in (PICKUP, REVIEW):
            r = self.records()
            r["board-new"]["owner"]["evidence_url"] = url
            for node in r:
                for item in r[node]["carried_forward"]:
                    item["source_url"] = url
                    if item.get("status") == "resolved":
                        item["resolution_url"] = url
            self.assertEqual(self.run_check(r), [], url)
        for url in rejected:
            r = self.records()
            r["board-new"]["owner"]["evidence_url"] = url
            self.assert_rejected(r, "no PICKUP")
            r = self.records()
            r["board-new"]["carried_forward"][0]["source_url"] = url
            self.assertEqual(self.run_check(r), [], url)
            r = self.records()
            r["board-old"]["carried_forward"][1]["resolution_url"] = url
            self.assert_rejected(r, "resolution_url")

    def test_zero_leading_comment_ids_fail(self):
        rejected = (
            "https://github.com/d6g8k5htny-coder/main/issues/0#issuecomment-200",
            "https://github.com/d6g8k5htny-coder/main/pull/02#issuecomment-200",
            "https://github.com/d6g8k5htny-coder/main/issues/2#issuecomment-0",
            "https://github.com/d6g8k5htny-coder/Math-/pull/3#issuecomment-0200",
            "https://github.com/d6g8k5htny-coder/main/pull/3#pullrequestreview-0",
            "https://github.com/d6g8k5htny-coder/Math-/issues/4#pullrequestreview-01",
        )
        for url in rejected:
            r = self.records()
            r["board-new"]["owner"]["evidence_url"] = url
            self.assert_rejected(r, "no PICKUP")
            r = self.records()
            r["board-new"]["carried_forward"][0]["source_url"] = url
            self.assertEqual(self.run_check(r), [], url)
            r = self.records()
            r["board-old"]["carried_forward"][1]["resolution_url"] = url
            self.assert_rejected(r, "resolution_url")

    def test_trailing_newline_in_url_fails(self):
        r = self.records()
        r["board-new"]["owner"]["evidence_url"] = PICKUP + "\n"
        self.assert_rejected(r, "no PICKUP")
        r = self.records()
        r["board-new"]["carried_forward"][0]["source_url"] = PICKUP + "\n"
        self.assert_rejected(r, "source_url")
        r = self.records()
        r["board-old"]["carried_forward"][1]["resolution_url"] = RESOLVED + "\n"
        self.assert_rejected(r, "resolution_url")

    def test_carried_item_without_text_fails(self):
        for mutate in (
            lambda item: item.pop("text"),
            lambda item: item.__setitem__("text", ""),
        ):
            r = self.records()
            mutate(r["board-new"]["carried_forward"][0])
            self.assert_rejected(r, "text")

    def test_refresh_error_must_be_string_or_null(self):
        r = self.records()
        r["board-new"]["refresh_error"] = 5
        self.assert_rejected(r, "refresh_error must be a string or null")
        for ok in (None, "unread"):
            r = self.records()
            r["board-new"]["refresh_error"] = ok
            self.assertEqual(self.run_check(r), [], ok)

    def test_released_owner_missing_id_fails(self):
        r = self.records()
        r["board-new"]["owner"] = {"evidence_url": PICKUP, "released_url": REVIEW}
        self.assert_rejected(r, "without an id")

    def test_released_owner_missing_evidence_url_fails(self):
        r = self.records()
        r["board-new"]["owner"] = {"id": "agent-old", "released_url": REVIEW}
        self.assert_rejected(r, "no PICKUP")

    def test_owner_extra_key_fails(self):
        r = self.records()
        r["board-new"]["owner"] = {
            "id": "agent-x", "evidence_url": PICKUP, "note": "extra",
        }
        self.assert_rejected(r, "only id, evidence_url, and released_url")

    def test_allowed_owner_keys_pass(self):
        for owner in (
            {"id": "agent-x", "evidence_url": PICKUP},
            {"id": "agent-old", "evidence_url": PICKUP, "released_url": REVIEW},
        ):
            r = self.records()
            r["board-new"]["owner"] = owner
            self.assertEqual(self.run_check(r), [], owner)

    def test_released_owner_with_evidence_passes(self):
        r = self.records()
        r["board-new"]["owner"] = {
            "id": "agent-old", "evidence_url": PICKUP, "released_url": REVIEW,
        }
        self.assertEqual(self.run_check(r), [])

    def test_malformed_released_url_fails(self):
        for bad in (
            "https://github.com/example/main/issues/2#issuecomment-200",
            "https://github.com/d6g8k5htny-coder/other/pull/3#pullrequestreview-300",
            "https://github.com/d6g8k5htny-coder/main/issues/2",
            "",
        ):
            r = self.records()
            r["board-new"]["owner"]["released_url"] = bad
            self.assert_rejected(r, "released_url")

    def test_carried_item_not_dict_fails(self):
        r = self.records()
        r["board-old"]["carried_forward"].append("stray")
        self.assert_rejected(r, "is not a dict")

    def test_carried_forward_dict_instead_of_list_fails(self):
        r = self.records()
        r["board-old"]["carried_forward"] = {
            "id": "item-a", "text": "open review", "source_url": PICKUP, "status": "open",
        }
        self.assert_rejected(r, "is not a list")

    def test_duplicate_item_id_fails(self):
        r = self.records()
        r["board-old"]["carried_forward"].append(
            {"id": "item-a", "text": "different text", "source_url": PICKUP, "status": "open"})
        self.assert_rejected(r, "duplicate carried_forward id")

    def test_link_url_not_http_fails(self):
        for node, field in (("board-old", "successor"), ("board-new", "predecessor")):
            for bad in ("not-a-url", "ftp://example.com/board", ""):
                r = self.records()
                r[node][field]["url"] = bad
                self.assert_rejected(r, "not an http(s) URL")
        r = self.records()
        r["board-old"]["successor"]["url"] = "http://example.com/board"
        r["board-new"]["predecessor"]["url"] = "http://example.com/old"
        self.assertEqual(self.run_check(r), [])

    def test_malformed_link_fails(self):
        r = self.records()
        r["board-old"]["successor"] = "board-new"
        self.assert_rejected(r, "malformed")

    def test_owner_without_id_fails(self):
        r = self.records()
        r["board-new"]["owner"] = {"evidence_url": PICKUP}
        self.assert_rejected(r, "without an id")

    def test_carried_item_id_must_be_string(self):
        r = self.records()
        r["board-new"]["carried_forward"].append(
            {"id": 5, "text": "extra", "source_url": PICKUP, "status": "open"})
        self.assert_rejected(r, "without an id")

    def test_carried_item_extra_key_fails(self):
        r = self.records()
        r["board-new"]["carried_forward"][0]["extra"] = "no"
        self.assert_rejected(r, "only id, text, source_url, status, and resolution_url")

    def test_carried_item_without_id_fails(self):
        r = self.records()
        r["board-new"]["carried_forward"].append(
            {"text": "extra", "source_url": PICKUP, "status": "open"})
        self.assert_rejected(r, "without an id")

    def test_bad_item_status_fails(self):
        r = self.records()
        r["board-new"]["carried_forward"].append(
            {"id": "item-c", "text": "nope", "source_url": PICKUP, "status": "closed"})
        self.assert_rejected(r, "has status")

    def test_duplicate_node_id_fails(self):
        r = self.records()
        r["board-old-again"] = copy.deepcopy(r["board-old"])
        self.assert_rejected(r, "duplicate node_id")

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
