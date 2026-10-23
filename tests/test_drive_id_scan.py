"""Scanner tests for tools/privacy/drive_id_scan.py.

Synthetic Drive-shaped tokens are built at runtime. This file does not contain
a real Drive identifier or a contiguous synthetic one. Hexadecimal pins stand
in for git and sha values.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import contextlib
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/privacy/drive_id_scan.py"
_spec = importlib.util.spec_from_file_location("drive_id_scan", TOOL)
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)

_HEX_DIGITS = "0123456789abcdefABCDEF"


def synthetic_id(length=33, prefix="1Syn"):
    """Assemble a Drive-shaped token. Z is outside the hex alphabet."""
    if len(prefix) != 4 or prefix[0] != "1":
        raise AssertionError("synthetic prefix must be four characters and start with 1")
    token = (prefix + ("Zz" * length))[:length]
    if len(token) != length or all(ch in _HEX_DIGITS for ch in token):
        raise AssertionError("synthetic token was not Drive-shaped")
    return token


def hex_pin(length):
    """A hexadecimal pin that starts with 1, the shape a sloppy scan might keep."""
    return ("1" + ("ab" * length))[:length]


class TextScan(unittest.TestCase):
    def test_drive_shaped_lengths_and_hex_pins(self):
        drive33 = synthetic_id(33, "1Syn")
        drive44 = synthetic_id(44, "1AbC")
        samples = {
            drive33: True,
            drive44: True,
            synthetic_id(32, "1Syn"): False,
            synthetic_id(45, "1Syn")[:45]: False,
            hex_pin(33): False,
            hex_pin(40): False,
            hex_pin(44): False,
            hex_pin(64): False,
            "existing_node_candidate_record": False,
            ("C292" + ("Title" * 8))[:33]: False,
            scan.ALLOWLIST: False,
        }
        # Rebuild the 45-character case explicitly so the slice above is honest.
        longer = synthetic_id(44, "1Syn") + "Z"
        self.assertEqual(len(longer), 45)
        samples[longer] = False
        for token, expect in samples.items():
            hits = scan.hits_in_text("value %s end" % token)
            prefixes = [prefix for _line, prefix in hits]
            if expect:
                self.assertEqual(prefixes, [token[:4]], token[:4])
                self.assertNotIn(token[4:], " ".join(prefixes))
            else:
                self.assertEqual(hits, [], token[:4])

    def test_boundaries_urls_and_id_fields(self):
        token = synthetic_id(33, "1Url")
        other = synthetic_id(44, "1Fld")
        glued = "Z" + token
        self.assertEqual(scan.hits_in_text(glued), [])
        self.assertEqual(scan.hits_in_text("(" + token + ")")[0][1], token[:4])
        urls = [
            "https://drive.google.com/file/d/%s/view" % token,
            "https://drive.google.com/open?id=%s" % token,
            "https://drive.google.com/drive/folders/%s" % token,
            "https://docs.google.com/document/d/%s/edit" % token,
            "https://docs.google.com/spreadsheets/d/%s/edit" % token,
            "https://docs.google.com/presentation/d/%s/edit" % token,
        ]
        for url in urls:
            hits = scan.hits_in_text(url)
            self.assertEqual([prefix for _line, prefix in hits], [token[:4]])
            joined = " ".join("%s:%s" % item for item in hits)
            if token in joined:
                self.fail("url hit kept more than the first 4 characters")
        body = "\n".join([
            '{"file_id": "%s", "drive_file_id": "%s", "folder_id": "%s"}' % (token, other, token),
            '"blob_sha": "%s"' % hex_pin(40),
            '"sha256": "%s"' % hex_pin(64),
            '"file_id": "%s"' % scan.ALLOWLIST,
        ])
        hits = scan.hits_in_text(body)
        prefixes = [prefix for _line, prefix in hits]
        self.assertEqual(prefixes.count(token[:4]), 2)
        self.assertEqual(prefixes.count(other[:4]), 1)
        self.assertNotIn("WITH", prefixes)
        rendered = " ".join(prefixes)
        if token in rendered or other in rendered:
            self.fail("field hit kept more than the first 4 characters")

    def test_allowlist_does_not_move_line_numbers(self):
        token = synthetic_id(33, "1Row")
        text = scan.ALLOWLIST + "\nnote " + token + "\n"
        self.assertEqual(scan.hits_in_text(text), [(2, token[:4])])
        same_line = "file_id " + scan.ALLOWLIST + " plus " + token
        self.assertEqual(scan.hits_in_text(same_line), [(1, token[:4])])

    def test_sources_have_no_drive_token(self):
        for path in (TOOL, Path(__file__)):
            self.assertEqual(scan.hits_in_text(path.read_text(encoding="utf-8")), [])


class DiffParse(unittest.TestCase):
    def test_ordinary_and_quoted_paths(self):
        token = synthetic_id(33, "1Dif")
        diff = "\n".join([
            'diff --git "a/my file.txt" "b/my file.txt"',
            "new file mode 100644",
            "--- /dev/null",
            '+++ "b/my file.txt"',
            "@@ -0,0 +3 @@",
            "+" + token,
            "+" + scan.ALLOWLIST,
        ])
        rows = list(scan.added_lines(diff))
        self.assertEqual([row[0] for row in rows], ["my file.txt", "my file.txt"])
        self.assertEqual(rows[0][1], 3)
        self.assertEqual(scan.hits_in_text(rows[0][2]), [(1, token[:4])])
        self.assertEqual(scan.hits_in_text(rows[1][2]), [])

    def test_combined_merge_marker(self):
        token = synthetic_id(33, "1Mrg")
        diff = "\n".join([
            "diff --git a/keep.txt b/keep.txt",
            "+++ b/keep.txt",
            "@@@ -1 -1 +2 @@@",
            "++" + token,
            "+ " + token,
        ])
        rows = list(scan.added_lines(diff, marker_width=2))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "keep.txt")
        self.assertEqual(scan.hits_in_text(rows[0][2]), [(1, token[:4])])

    def test_plus_plus_content_line_is_not_a_header(self):
        token = synthetic_id(33, "1Pls")
        diff = "\n".join([
            "diff --git a/notes.md b/notes.md",
            "--- a/notes.md",
            "+++ b/notes.md",
            "@@ -1,0 +2,2 @@",
            "+++ heading",
            "+file_id " + token,
        ])
        rows = list(scan.added_lines(diff))
        self.assertEqual([(r[0], r[1]) for r in rows], [("notes.md", 2), ("notes.md", 3)])
        self.assertEqual(rows[0][2], "++ heading")
        self.assertEqual(scan.hits_in_text(rows[1][2]), [(1, token[:4])])

    def test_mixed_prefix_merge_rows_advance_line(self):
        token = synthetic_id(33, "1Mix")
        diff = "\n".join([
            "diff --cc conflict.txt",
            "+++ b/conflict.txt",
            "@@@ -1,1 -1,1 +1,3 @@@",
            "+ from-first",
            " +from-second",
            "- dropped",
            "++file_id " + token,
        ])
        rows = list(scan.added_lines(diff, marker_width=2))
        self.assertEqual([(r[0], r[1]) for r in rows], [("conflict.txt", 3)])


class GitRange(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.env = {
            "GIT_AUTHOR_NAME": "drive-scan-test",
            "GIT_AUTHOR_EMAIL": "drive-scan-test@example.com",
            "GIT_COMMITTER_NAME": "drive-scan-test",
            "GIT_COMMITTER_EMAIL": "drive-scan-test@example.com",
            "GIT_CONFIG_NOSYSTEM": "1",
            "PATH": "/usr/bin:/bin",
            "HOME": self.tmp.name,
        }
        self.git(["init", "-q"])
        self.git(["config", "user.email", "drive-scan-test@example.com"])
        self.git(["config", "user.name", "drive-scan-test"])
        self.git(["config", "commit.gpgsign", "false"])

    def git(self, args):
        proc = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            capture_output=True, text=True, env=self.env, check=False,
        )
        if proc.returncode != 0:
            self.fail("git failed: %s" % args[0])
        return proc.stdout.strip()

    def commit_file(self, name, content, message):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.git(["add", "--", name])
        self.git(["commit", "-q", "-m", message])
        return self.git(["rev-parse", "HEAD"])

    def test_per_commit_diff_flags_an_id_removed_later(self):
        token = synthetic_id(33, "1Old")
        base = self.commit_file("notes.txt", "base\n", "base")
        leaked = self.commit_file(
            "notes.txt",
            "\n".join([
                "blob_sha %s" % hex_pin(40),
                "sha256 %s" % hex_pin(64),
                "hex33 %s" % hex_pin(33),
                "file_id %s" % token,
                "title %s" % (("C292" + ("Title" * 8))[:33]),
            ]) + "\n",
            "introduce a drive-shaped file_id",
        )
        head = self.commit_file(
            "notes.txt",
            "\n".join([
                "blob_sha %s" % hex_pin(40),
                "sha256 %s" % hex_pin(64),
                "file_id %s" % scan.ALLOWLIST,
            ]) + "\n",
            "replace the file_id with the allow-listed placeholder",
        )
        history = scan.scan_range(self.repo, base, head)
        self.assertTrue(history)
        rendered = "\n".join(history)
        if token in rendered:
            self.fail("range output kept more than the first 4 characters")
        self.assertIn(token[:4], rendered)
        self.assertIn("\tcommit\t%s\t" % leaked, rendered)
        self.assertNotIn(hex_pin(40), rendered)
        self.assertNotIn(hex_pin(64), rendered)
        clean = scan.scan_range(self.repo, leaked, head)
        self.assertEqual(clean, [])
        still = scan.scan_range(self.repo, base, leaked)
        self.assertTrue(any(token[:4] in line for line in still))

    def test_message_space_in_path_and_empty_range(self):
        token = synthetic_id(44, "1Msg")
        base = self.commit_file("readme.txt", "base\n", "base")
        head = self.commit_file("my file.txt", token + "\n", "note " + token)
        hits = scan.scan_range(self.repo, base, head)
        rendered = "\n".join(hits)
        if token in rendered:
            self.fail("path or message output kept more than the first 4 characters")
        self.assertGreaterEqual(len(hits), 2)
        self.assertTrue(any("\tmy file.txt\t" in line for line in hits))
        self.assertTrue(any("\tmessage\t" in line for line in hits))
        self.assertEqual(scan.scan_range(self.repo, head, head), [])

    def test_cli_body_file_stdin_and_exit_codes(self):
        token = synthetic_id(33, "1Bod")
        base = self.commit_file("readme.txt", "clean\n", "base")
        head = self.commit_file("readme.txt", "still clean\n", "edit")
        body = self.repo / "body.txt"
        body.write_text('folder https://drive.google.com/drive/folders/%s\n' % token, encoding="utf-8")
        file_run = self.cli(["--base", base, "--head", head, "--body", str(body)])
        self.assertEqual(file_run.returncode, 1)
        if token in file_run.stdout or token in file_run.stderr:
            self.fail("cli output kept more than the first 4 characters")
        self.assertIn(token[:4] + "\tbody\t-\tpr-body\t1", file_run.stdout)
        stdin_run = self.cli(["--body", "-"], input_text=body.read_text(encoding="utf-8"))
        self.assertEqual(stdin_run.returncode, 1)
        self.assertEqual(stdin_run.stdout, file_run.stdout)
        clean = self.cli(["--base", base, "--head", head, "--body", "-"], input_text=scan.ALLOWLIST + "\n")
        self.assertEqual(clean.returncode, 0)
        self.assertEqual(clean.stdout, "")
        usage = self.cli(["--base", base])
        self.assertEqual(usage.returncode, 2)
        missing = self.cli(["--base", "does-not-exist", "--head", head])
        self.assertEqual(missing.returncode, 2)
        if token in missing.stderr:
            self.fail("git error kept more than the first 4 characters")

    def test_plus_plus_line_in_range_is_scanned(self):
        token = synthetic_id(33, "1Hdr")
        base = self.commit_file("notes.md", "base\n", "base")
        head = self.commit_file("notes.md", "base\n++ heading\nfile_id %s\n" % token, "add")
        hits = scan.scan_range(self.repo, base, head)
        self.assertIn("%s\tcommit\t%s\tnotes.md\t3" % (token[:4], head), hits)
        self.assertIn("%s\tfiles\t-\tnotes.md\t3" % token[:4], hits)
        run = self.cli(["--base", base, "--head", head])
        self.assertEqual(run.returncode, 1)

    def test_dedup_keeps_each_commit_and_head(self):
        first = synthetic_id(33, "1Dup")
        second = synthetic_id(33, "1Dup")[:-1] + "Q"
        self.assertNotEqual(first, second)
        base = self.commit_file("notes.txt", "base\n", "base")
        one = self.commit_file("notes.txt", "file_id %s\n" % first, "first")
        two = self.commit_file("notes.txt", "file_id %s\n" % second, "second")
        hits = scan.scan_range(self.repo, base, two)
        self.assertIn("1Dup\tcommit\t%s\tnotes.txt\t1" % one, hits)
        self.assertIn("1Dup\tcommit\t%s\tnotes.txt\t1" % two, hits)
        self.assertIn("1Dup\tfiles\t-\tnotes.txt\t1", hits)
        self.assertEqual(len(hits), len(set(hits)))

    def test_pure_rename_and_empty_file_paths_are_scanned(self):
        token = synthetic_id(33, "1Ren")
        empty = synthetic_id(33, "1Emp")
        base = self.commit_file("clean.txt", "same\n", "base")
        (self.repo / "r").mkdir()
        self.git(["mv", "clean.txt", "r/%s.txt" % token])
        self.git(["commit", "-q", "-m", "rename"])
        renamed = self.git(["rev-parse", "HEAD"])
        head = self.commit_file("e/%s" % empty, "", "empty")
        hits = scan.scan_range(self.repo, base, head)
        rendered = "\n".join(hits)
        if token in rendered or empty in rendered:
            self.fail("path output kept more than the first 4 characters")
        self.assertTrue(any(l.startswith("1Ren\tcommit\t%s\t" % renamed) for l in hits))
        self.assertTrue(any(l.startswith("1Ren\tfiles\t-\t") for l in hits))
        self.assertTrue(any(l.startswith("1Emp\tcommit\t%s\t" % head) for l in hits))
        self.assertEqual(self.cli(["--base", base, "--head", head]).returncode, 1)

    def test_body_read_error_is_redacted_exit_2(self):
        token = synthetic_id(33, "1Unr")
        body = self.repo / ("body-%s.txt" % token)
        body.write_text("x\n", encoding="utf-8")
        err = io.StringIO()
        out = io.StringIO()
        boom = PermissionError(13, "Permission denied", str(body))
        with mock.patch.object(scan.Path, "read_text", side_effect=boom), \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            code = scan.main(["--body", str(body)])
        self.assertEqual(code, 2)
        if token in err.getvalue() or token in out.getvalue():
            self.fail("body read error kept more than the first 4 characters")
        self.assertIn("cannot read PR body: Permission denied", err.getvalue())
        with mock.patch.object(scan.sys, "stdin") as fake, \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            fake.read.side_effect = OSError(5, "I/O error")
            self.assertEqual(scan.main(["--body", "-"]), 2)

    def cli(self, args, input_text=""):
        return subprocess.run(
            [sys.executable, "-B", str(TOOL), "--repo", str(self.repo), *args],
            capture_output=True, text=True, input=input_text, check=False,
        )


if __name__ == "__main__":
    unittest.main()
