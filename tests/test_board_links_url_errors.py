"""Malformed handoff URLs must be validation errors, not uncaught exceptions.

All records are synthetic. These tests exercise the real JSON loader, checker,
and CLI without network access or claims about real boards and owners.
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
TOOL = ROOT / "tools" / "check_board_links.py"
SPEC = importlib.util.spec_from_file_location("board_links_url_errors_target", TOOL)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

BAD_URLS = (
    "http://[",
    "https://]",
    "https://[not-an-ip]/board",
    "http://[::gg]/board",
    "https://example.com\uff0fboard",
    "https://example.com\uff1aport",
    "https://user\uff20example.com/board",
    "https://example.com\uff03fragment",
)
GOOD_URLS = (
    "https://github.com/example/repo/issues/1",
    "http://localhost/board",
    "http://127.0.0.1:8080/board",
    "https://[::1]/board",
    "HTTPS://[2001:db8::1]:443/board",
    "https://b\u00fccher.example/board",
)
BASE = {
    "old": {
        "node_id": "old", "owner": None,
        "successor": {"node_id": "new", "url": GOOD_URLS[0]},
        "carried_forward": [],
    },
    "new": {
        "node_id": "new", "owner": None,
        "predecessor": {"node_id": "old", "url": GOOD_URLS[0]},
        "carried_forward": [],
    },
}
DIRECTIONS = (("old", "successor"), ("new", "predecessor"))


class BoardLinkUrlErrors(unittest.TestCase):
    def records(self, node, field, url):
        records = copy.deepcopy(BASE)
        records[node][field]["url"] = url
        return records

    def invoke(self, records, cli=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for node, record in records.items():
                (root / (node + ".json")).write_text(
                    json.dumps(record), encoding="utf-8")
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            if cli:
                command = [sys.executable, "-I", "-B", "-S"]
                if sys.flags.optimize:
                    command.append("-O")
                result = subprocess.run(
                    command + [str(TOOL), directory], capture_output=True,
                    check=False, timeout=10)
            else:
                try:
                    result = CHECKER.check(directory)
                except ValueError as exc:
                    self.fail("malformed URL escaped validation as ValueError: " + str(exc))
            after = {path.name: path.read_bytes() for path in root.iterdir()}
            self.assertEqual(after, before, "checker must not alter input records")
            return result

    def assert_cli_report(self, records, expected_errors):
        process = self.invoke(records, cli=True)
        self.assertEqual(process.returncode, 1 if expected_errors else 0)
        self.assertTrue(process.stdout.startswith(b"{"),
                        "CLI must emit its JSON report, not an empty crash result")
        self.assertEqual(process.stderr, b"")
        report = json.loads(process.stdout)
        self.assertEqual(report, {
            "ok": not expected_errors,
            "errors": expected_errors,
            "scientific_effect": "NONE",
        })

    def test_parser_errors_are_reported_for_both_link_directions(self):
        for node, field in DIRECTIONS:
            for url in BAD_URLS:
                with self.subTest(node=node, field=field, url=url):
                    errors = self.invoke(self.records(node, field, url))
                    self.assertEqual(errors, [node + ": " + field + " url is not an http(s) URL"])

    def test_cli_reports_parser_errors_for_both_link_directions(self):
        for node, field in DIRECTIONS:
            for url in BAD_URLS:
                with self.subTest(node=node, field=field, url=url):
                    self.assert_cli_report(self.records(node, field, url),
                                           [node + ": " + field + " url is not an http(s) URL"])

    def test_valid_http_urls_keep_their_existing_behavior(self):
        for node, field in DIRECTIONS:
            for url in GOOD_URLS:
                with self.subTest(node=node, field=field, url=url):
                    self.assertEqual(self.invoke(self.records(node, field, url)), [])

    def test_cli_accepts_valid_http_urls(self):
        for url in GOOD_URLS:
            with self.subTest(url=url):
                self.assert_cli_report(self.records("old", "successor", url), [])

    def test_other_invalid_inputs_remain_validation_errors(self):
        for value in (None, 7, [], {}, "", "ftp://example.com/board", "https:///board",
                      "https://example.com/a b", "https://example.com/board\n"):
            for node, field in DIRECTIONS:
                with self.subTest(value=value, node=node, field=field):
                    errors = self.invoke(self.records(node, field, value))
                    self.assertEqual(errors, [node + ": " + field + " url is not an http(s) URL"])

    def test_multiple_bad_urls_preserve_other_validation_findings(self):
        records = self.records("old", "successor", BAD_URLS[0])
        records["new"]["predecessor"]["url"] = BAD_URLS[4]
        records["new"]["refresh_error"] = 9
        expected = [
            "new: predecessor url is not an http(s) URL",
            "old: successor url is not an http(s) URL",
            "new: refresh_error must be a string or null",
        ]
        self.assertEqual(self.invoke(records), expected)
        self.assert_cli_report(records, expected)


if __name__ == "__main__":
    unittest.main()
