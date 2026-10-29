"""Tests for build_census.py's citation checks.

In-process tests corrupt one citation at a time and require CensusError. Subprocess
tests run the real command line against a fixture repository in both ordinary and
optimized (-O) Python, because assert statements would vanish under -O (finding
DCL-691-F1): a valid control must pass and write output, and every invalid case must
exit 1 and write nothing. Run: python3 -B -I test_build_census.py
"""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_census as bc  # noqa: E402

REPO = Path(subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                           check=True, stdout=subprocess.PIPE, text=True).stdout.strip())
MODES = {"ordinary": [sys.executable, "-B", "-I"], "optimized": [sys.executable, "-B", "-I", "-O"]}


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, text=True).stdout.strip()


def first_entry():
    return copy.deepcopy(bc.ENTRIES[0])


class InProcessChecks(unittest.TestCase):
    def test_every_real_entry_passes(self):
        for entry in copy.deepcopy(bc.ENTRIES):
            with self.subTest(entry=entry["id"]):
                bc.check(REPO, entry)

    def test_wrong_blob_fails(self):
        entry = first_entry()
        entry["blob"] = "0000000"
        with self.assertRaises(bc.CensusError):
            bc.check(REPO, entry)

    def test_line_range_past_the_end_fails(self):
        entry = first_entry()
        entry["statement"] = [(1, 10 ** 6, None)]
        with self.assertRaises(bc.CensusError):
            bc.check(REPO, entry)

    def test_inverted_or_zero_range_fails(self):
        for rows in ([(20, 10, None)], [(0, 5, None)]):
            entry = first_entry()
            entry["statement"] = rows
            with self.subTest(rows=rows), self.assertRaises(bc.CensusError):
                bc.check(REPO, entry)

    def test_anchor_outside_its_range_fails(self):
        entry = first_entry()
        first, last, anchor = entry["statement"][0]
        entry["statement"] = [(last + 1, last + 2, anchor)]
        with self.assertRaises(bc.CensusError):
            bc.check(REPO, entry)

    def test_missing_path_fails(self):
        entry = first_entry()
        entry["path"] = "no/such/PROOF.md"
        with self.assertRaises(bc.CensusError):
            bc.check(REPO, entry)


class CommandLineInBothModes(unittest.TestCase):
    """The real command, against a fixture repository, in ordinary and -O Python."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.repo = cls.root / "repo"
        (cls.repo / "docs").mkdir(parents=True)
        (cls.repo / "formal").mkdir()
        git(cls.repo, "init", "-q", "-b", "main")
        git(cls.repo, "config", "user.email", "t@example.invalid")
        git(cls.repo, "config", "user.name", "t")
        (cls.repo / "docs" / "PROOF.md").write_text("# Title\nthe stated claim\nits limits\n")
        cls.write_manifest(31, bc.APPROVED_AXIOMS)
        git(cls.repo, "add", "-A")
        git(cls.repo, "commit", "-q", "-m", "fixture")
        cls.rev = git(cls.repo, "rev-parse", "HEAD")
        cls.blob = git(cls.repo, "rev-parse", "HEAD:docs/PROOF.md")
        cls.write_manifest(30, bc.APPROVED_AXIOMS)
        git(cls.repo, "commit", "-q", "-am", "thirty targets")
        cls.rev_30 = git(cls.repo, "rev-parse", "HEAD")
        cls.write_manifest(31, bc.APPROVED_AXIOMS + ["sorryAx"])
        git(cls.repo, "commit", "-q", "-am", "unapproved axiom")
        cls.rev_axiom = git(cls.repo, "rev-parse", "HEAD")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def write_manifest(cls, count, axioms):
        targets = [{"name": f"T.t{i}", "module": "T.lean", "title": "t", "does_not_claim": "-"}
                   for i in range(count)]
        (cls.repo / "formal" / "manifest.json").write_text(
            json.dumps({"targets": targets, "allowed_axioms": axioms}))

    def entry(self, **overrides):
        entry = {"id": "OBL-T-01", "path": "docs/PROOF.md", "blob": self.blob[:12], "kind": "k",
                 "disposition": "QUARANTINED", "claim": "c", "statement": [[2, 2, "stated claim"]],
                 "nonclaims": [[3, 3, None]], "reviews": "r", "depends": "d", "boundary": "b"}
        entry.update(overrides)
        return entry

    def run_cli(self, mode, entry, rev=None):
        with tempfile.TemporaryDirectory(dir=self.root) as work:
            work = Path(work)
            (work / "entries.json").write_text(json.dumps([entry]))
            out = work / "out"
            proc = subprocess.run(MODES[mode] + [str(HERE / "build_census.py"), "--git", str(self.repo),
                                                 "--rev", rev or self.rev, "--entries", str(work / "entries.json"),
                                                 "--out", str(out)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            written = sorted(p.name for p in out.glob("*")) if out.exists() else None
        refused = proc.stderr.startswith("CENSUS CHECK FAILED:") and "Traceback" not in proc.stderr
        return proc.returncode, written, refused

    def test_valid_control_passes_and_writes(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assertEqual(self.run_cli(mode, self.entry()), (0, ["CENSUS.md", "census.json"], False))

    def test_invalid_evidence_is_refused_with_no_output(self):
        cases = {
            "wrong blob": (self.entry(blob="0000000"), None),
            "range past the end": (self.entry(statement=[[2, 99, None]]), None),
            "inverted range": (self.entry(statement=[[3, 2, None]]), None),
            "anchor outside its range": (self.entry(statement=[[3, 3, "stated claim"]]), None),
            "missing path": (self.entry(path="docs/NONE.md"), None),
            "thirty Lean targets": (self.entry(), "rev_30"),
            "unapproved axiom": (self.entry(), "rev_axiom"),
        }
        for mode in MODES:
            for name, (entry, rev_attr) in cases.items():
                rev = getattr(self, rev_attr) if rev_attr else None
                if rev_attr:
                    # The cited file is unchanged by the later commits, so only the manifest differs.
                    entry = self.entry(blob=git(self.repo, "rev-parse", f"{rev}:docs/PROOF.md")[:12])
                with self.subTest(mode=mode, case=name):
                    # Exit 1 with the refusal message, no traceback, and no output directory at all.
                    self.assertEqual(self.run_cli(mode, entry, rev), (1, None, True))


if __name__ == "__main__":
    unittest.main()
