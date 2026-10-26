"""Raw-file regression proposal: duplicate JSON names must not erase handoffs.

All records are synthetic. Tests run the actual checker CLI in both modes;
no network calls and no real ownership/custody assertions are made.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/check_board_links.py"
PICKUP = "https://github.com/d6g8k5htny-coder/main/issues/2#issuecomment-200"
RELEASE = "https://github.com/d6g8k5htny-coder/main/issues/2#pullrequestreview-201"
ITEM = {"id": "hold-1", "text": "Unresolved prerequisite",
        "source_url": "https://github.com/example/repo/issues/1", "status": "open"}


def raw_object(pairs):
    # Intentionally allow duplicate names. json.dumps(dict(...)) would erase them.
    return "{" + ",".join(json.dumps(k) + ":" + json.dumps(v) for k, v in pairs) + "}"


class RawBoardJSON(unittest.TestCase):
    def run_files(self, texts):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, text in texts.items():
                (root / name).write_text(text, encoding="utf-8")
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            outputs = []
            for mode in ([], ["-O"]):
                command = [sys.executable, "-B", "-S", *mode, str(TOOL), directory]
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.stderr, "", command)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["scientific_effect"], "NONE")
                outputs.append((result.returncode, result.stdout, payload))
                self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in root.iterdir()})
            self.assertEqual(outputs[0], outputs[1], "normal/optimized checker behavior differs")
            return outputs[0][0], outputs[0][2]

    def assert_duplicate(self, text, key):
        code, out = self.run_files({"record.json": text})
        self.assertEqual(code, 1, out)
        self.assertFalse(out["ok"], out)
        self.assertTrue(any("duplicate JSON key: " + key in e for e in out["errors"]), out)

    def test_dropped_open_item_cannot_be_hidden(self):
        old = {"node_id": "old", "owner": None,
               "successor": {"node_id": "new", "url": "https://example.com/new"},
               "carried_forward": [ITEM]}
        new = {"node_id": "new", "owner": None,
               "predecessor": {"node_id": "old", "url": "https://example.com/old"},
               "carried_forward": []}
        code, out = self.run_files({"a.json": json.dumps(old), "b.json": json.dumps(new)})
        self.assertEqual(code, 1)
        self.assertTrue(any("not carried forward" in e for e in out["errors"]), out)
        raw = json.dumps(old)[:-1] + ', "carried_forward": []}'
        code, out = self.run_files({"a.json": raw, "b.json": json.dumps(new)})
        self.assertEqual(code, 1, out)
        self.assertTrue(any("duplicate JSON key: carried_forward" in e for e in out["errors"]), out)

    def test_owner_cannot_be_hidden_by_later_null(self):
        self.assert_duplicate(raw_object([("node_id", "n"), ("owner", {"id": "unbacked"}),
                                          ("owner", None)]), "owner")

    def test_nested_owner_evidence_cannot_be_replaced(self):
        owner = raw_object([("id", "agent"), ("evidence_url", "not-a-comment"),
                            ("evidence_url", PICKUP)])
        self.assert_duplicate('{"node_id":"n","owner":' + owner + '}', "evidence_url")

    def test_nested_item_status_cannot_be_replaced(self):
        item = raw_object([*ITEM.items(), ("status", "open")])
        self.assert_duplicate('{"node_id":"n","carried_forward":[' + item + ']}', "status")

    def test_escaped_names_are_compared_after_decoding(self):
        self.assert_duplicate('{"node_id":"n","owner":{"id":"unbacked"},"own\\u0065r":null}', "owner")

    def test_equal_valued_duplicate_is_still_ambiguous(self):
        self.assert_duplicate('{"node_id":"n","node_id":"n","owner":null}', "node_id")

    def test_array_nested_names_are_checked(self):
        # A valid resolved item must retain exactly one resolution_url key.
        item = raw_object([*dict(ITEM, status="resolved").items(),
                           ("resolution_url", "invalid"), ("resolution_url", RELEASE)])
        self.assert_duplicate('{"node_id":"n","carried_forward":[' + item + ']}', "resolution_url")

    def test_separate_objects_can_reuse_the_same_names(self):
        a = {"node_id": "a", "owner": {"id": "agent", "evidence_url": PICKUP},
             "successor": {"node_id": "b", "url": "https://example.com/b"}, "carried_forward": [ITEM]}
        b = {"node_id": "b", "owner": None,
             "predecessor": {"node_id": "a", "url": "https://example.com/a"}, "carried_forward": [ITEM]}
        code, out = self.run_files({"a.json": json.dumps(a), "b.json": json.dumps(b)})
        self.assertEqual((code, out), (0, {"ok": True, "errors": [], "scientific_effect": "NONE"}))

    def test_other_file_errors_are_not_silenced(self):
        code, out = self.run_files({"a.json": '{"node_id":"a","owner":null,"owner":null}',
                                   "b.json": '{"node_id":"b","owner":{"id":"unbacked"}}'})
        self.assertEqual(code, 1, out)
        self.assertTrue(any("duplicate JSON key: owner" in e for e in out["errors"]), out)
        self.assertTrue(any("no PICKUP" in e for e in out["errors"]), out)


if __name__ == "__main__":
    unittest.main()
