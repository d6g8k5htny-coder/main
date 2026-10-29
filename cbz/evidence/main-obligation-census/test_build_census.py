"""Adverse tests for build_census.py's citation checks.

Each test corrupts one citation and requires the check to fail; a final test
requires every real entry to pass. Run from a clone that contains REV:
    python3 -B -I test_build_census.py
The tests read Git objects from the repository that contains this file.
"""

import copy
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_census as bc  # noqa: E402

REPO = Path(subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                           check=True, stdout=subprocess.PIPE, text=True).stdout.strip())


def first_entry():
    return copy.deepcopy(bc.ENTRIES[0])


class CitationChecks(unittest.TestCase):
    def test_every_real_entry_passes(self):
        for entry in copy.deepcopy(bc.ENTRIES):
            with self.subTest(entry=entry["id"]):
                bc.check(REPO, entry)

    def test_wrong_blob_fails(self):
        entry = first_entry()
        entry["blob"] = "0000000"
        with self.assertRaises(AssertionError):
            bc.check(REPO, entry)

    def test_line_range_past_the_end_fails(self):
        entry = first_entry()
        entry["statement"] = [(1, 10 ** 6, None)]
        with self.assertRaises(AssertionError):
            bc.check(REPO, entry)

    def test_inverted_or_zero_range_fails(self):
        for rows in ([(20, 10, None)], [(0, 5, None)]):
            entry = first_entry()
            entry["statement"] = rows
            with self.subTest(rows=rows), self.assertRaises(AssertionError):
                bc.check(REPO, entry)

    def test_anchor_outside_its_range_fails(self):
        entry = first_entry()
        first, last, anchor = entry["statement"][0]
        entry["statement"] = [(last + 1, last + 2, anchor)]
        with self.assertRaises(AssertionError):
            bc.check(REPO, entry)

    def test_missing_path_fails(self):
        entry = first_entry()
        entry["path"] = "no/such/PROOF.md"
        with self.assertRaises(subprocess.CalledProcessError):
            bc.check(REPO, entry)


if __name__ == "__main__":
    unittest.main()
