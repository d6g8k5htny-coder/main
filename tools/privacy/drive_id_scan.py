#!/usr/bin/env python3
"""Scan a git range and a pull-request body for Google Drive file or folder IDs.

Local reviewer tool for OPPRIV-SCAN-01. It is not a required check and it does
not change workflow or settings. Scientific effect: none.

The scan covers three places:

  files   added lines of the net diff base..head (the changed text at head)
  commit  added lines and the message of every commit in base..head
  body    a pull-request body from a file or from stdin

Merge commits are scanned through their messages and any combined diff (lines
the merge itself introduced). Their parents' patches are scanned when those
commits fall inside the range.

A hit is a Drive/Docs URL id, a drive_file_id / file_id / *_id value, or any
other Drive-shaped token: 33 or 44 characters from the Drive alphabet, starting
with "1", and not a hexadecimal git or sha pin. The literal WITHHELD
(OP-PRIVACY) is allow-listed.

Stdout is one hit per line. The identifier itself is only its first four
characters. Exit status is 1 when any hit is found, 0 when the scan is clean,
and 2 when the invocation or git fails.

  python -B tools/privacy/drive_id_scan.py --base BASE --head HEAD
  python -B tools/privacy/drive_id_scan.py --base BASE --head HEAD --body pr.txt
  python -B tools/privacy/drive_id_scan.py --body -
"""

from __future__ import annotations

import argparse
import codecs
import re
import subprocess
import sys
from pathlib import Path

ALLOWLIST = "WITHHELD (OP-PRIVACY)"
_HEX = frozenset("0123456789abcdefABCDEF")
_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")
# 33- and 44-character Drive tokens. Boundaries keep hex pins (40 and 64)
# and longer identifiers from matching a prefix.
# Only letters and digits bound a token: a "-" or "_" before or after it (as in
# "private-<id>.txt") must not hide it from detection or redaction.
_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(1(?:[A-Za-z0-9_-]{43}|[A-Za-z0-9_-]{32}))(?![A-Za-z0-9])"
)
_HUNK = re.compile(r"\+(\d+)")


class ScanError(Exception):
    """git failed or the scan could not be read. The message is already redacted."""


def is_drive_token(token):
    """True for a Drive-shaped token that is not a hexadecimal pin."""
    if len(token) not in (33, 44) or not token.startswith("1"):
        return False
    if any(ch not in _ALPHABET for ch in token):
        return False
    if all(ch in _HEX for ch in token):
        return False
    return True


def redact(text):
    """Replace any Drive-shaped token with its four-character prefix."""

    def repl(match):
        token = match.group(1)
        if not is_drive_token(token):
            return token
        return token[:4] + "…"

    return _TOKEN.sub(repl, text)


def _mask(text):
    """Blank the allow-listed placeholder without moving later line numbers."""
    return text.replace(ALLOWLIST, " " * len(ALLOWLIST))


def hits_in_text(text):
    """Return (line_number, four-character prefix) for every hit in text.

    Line numbers are 1-based. The full token is not returned.
    """
    found = []
    masked = _mask(text)
    for line_number, line in enumerate(masked.splitlines(), 1):
        for match in _TOKEN.finditer(line):
            token = match.group(1)
            if is_drive_token(token):
                found.append((line_number, token[:4]))
    return found


def _git_path(spec):
    spec = spec.strip()
    if len(spec) >= 2 and spec[0] == '"' and spec[-1] == '"':
        spec = codecs.decode(spec[1:-1], "unicode_escape")
    if spec == "/dev/null":
        return None
    if spec.startswith("b/"):
        return spec[2:]
    return spec


def added_lines(diff_text, marker_width=1):
    """Yield (path, new_line_number, text) for lines added by a unified diff.

    marker_width is 1 for an ordinary diff and the parent count for a combined
    merge diff, whose added lines begin with that many '+' markers.
    """
    marker = "+" * marker_width
    path = None
    new_line = None
    # True between a "diff " line and its first hunk. Only there is a
    # "+++ " line a file header; inside a hunk it is an added "++" line.
    in_header = False
    # Git ends diff rows with LF only; splitlines() would also break a row at
    # \f, \r, or U+2028 and drop the tail of an added line.
    for raw in diff_text.split("\n"):
        if raw.startswith("diff "):
            path = None
            new_line = None
            in_header = True
            continue
        if in_header and raw.startswith("+++ "):
            path = _git_path(raw[4:])
            continue
        if raw.startswith("@@"):
            match = _HUNK.search(raw)
            new_line = int(match.group(1)) if match else None
            in_header = False
            continue
        if in_header:
            continue
        if path is None or new_line is None:
            continue
        if raw.startswith("\\"):
            continue
        columns = raw[:marker_width]
        if len(columns) < marker_width or "-" in columns:
            continue  # row absent from the result file
        # Every row without a "-" column exists in the result, so it
        # advances the counter; only all-"+" rows are scanned.
        if columns == marker:
            yield path, new_line, raw[marker_width:]
        new_line += 1


def _git(repo, args):
    proc = subprocess.run(
        ["git", "--no-pager", "-C", str(repo), *args],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        detail = redact(proc.stderr).strip() or "git exited %s" % proc.returncode
        raise ScanError(detail)
    return proc.stdout


def _record(prefix, channel, sha, path, line):
    shown_sha = sha if sha else "-"
    shown_path = redact(path)
    rendered = "%s\t%s\t%s\t%s\t%s" % (prefix, channel, shown_sha, shown_path, line)
    return redact(rendered)


def _add(store, seen, prefix, channel, sha, path, line):
    # Channel and commit are part of the identity: the same prefix at the
    # same place in two commits, or at head, is a separate exposure.
    place = (prefix, channel, sha or "", path, line)
    if place in seen:
        return
    seen.add(place)
    store.append((channel, sha or "", path, line, prefix))


def _scan_paths(store, seen, names, channel, sha):
    """Scan changed destination paths (NUL-separated --name-only output).

    Independent of content hunks, so pure renames and empty-file additions
    are covered. A path hit is reported at line 0.
    """
    for path in names.split("\0"):
        if not path:
            continue
        for _ignored, prefix in hits_in_text(path):
            _add(store, seen, prefix, channel, sha, redact(path), 0)


def _scan_patch(store, seen, diff_text, channel, sha, marker_width):
    for path, line, text in added_lines(diff_text, marker_width):
        safe_path = redact(path)
        for _ignored, prefix in hits_in_text(text):
            _add(store, seen, prefix, channel, sha, safe_path, line)


def scan_body(text):
    """Scan a pull-request body. Returns rendered hit lines."""
    store = []
    seen = set()
    for line, prefix in hits_in_text(text):
        _add(store, seen, prefix, "body", "", "pr-body", line)
    return [_record(*_order(item)) for item in _sorted(store)]


def _order(item):
    channel, sha, path, line, prefix = item
    return prefix, channel, sha, path, line


def _sorted(store):
    return sorted(store, key=lambda item: (item[0], item[1], item[2], item[3], item[4]))


def scan_range(repo, base, head):
    """Scan files changed in base..head and every commit in that range.

    Returns rendered hit lines. Raises ScanError when git cannot read the range.
    The full identifier is never included.
    """
    # Run every query from the top level and with --no-relative, so --repo
    # pointing at a subdirectory or diff.relative=true cannot hide changes.
    repo = Path(_git(Path(repo), ["rev-parse", "--show-toplevel"]).strip())
    store = []
    seen = set()
    commits = [
        line.strip()
        for line in _git(repo, ["rev-list", "--reverse", "%s..%s" % (base, head)]).splitlines()
        if line.strip()
    ]
    for sha in commits:
        parents = _git(repo, ["show", "-s", "--format=%P", sha]).split()
        message = _git(repo, ["show", "-s", "--format=%B", sha])
        for line, prefix in hits_in_text(message):
            _add(store, seen, prefix, "commit", sha, "message", line)
        if len(parents) >= 2:
            diff = _git(
                repo,
                ["show", "--no-relative", "-c", "-p", "--format=", "--unified=0", "--no-color",
                 "--no-ext-diff", "--no-textconv", sha],
            )
            _scan_patch(store, seen, diff, "commit", sha, len(parents))
            names = _git(repo, ["show", "--no-relative", "-c", "--format=", "--name-only", "-z", sha])
        else:
            diff = _git(
                repo,
                ["show", "--no-relative", "-p", "--format=", "--unified=0", "--no-color",
                 "--no-ext-diff", "--no-textconv", sha],
            )
            _scan_patch(store, seen, diff, "commit", sha, 1)
            names = _git(repo, ["show", "--no-relative", "--format=", "--name-only", "-z", "-M", sha])
        _scan_paths(store, seen, names, "commit", sha)
    net = _git(
        repo,
        ["diff", "--no-relative", "--unified=0", "--no-color", "--no-ext-diff", "--no-textconv",
         "%s..%s" % (base, head)],
    )
    _scan_patch(store, seen, net, "files", "", 1)
    names = _git(repo, ["diff", "--no-relative", "--name-only", "-z", "-M", "%s..%s" % (base, head)])
    _scan_paths(store, seen, names, "files", "")
    return [_record(*_order(item)) for item in _sorted(store)]


def _read_body(spec):
    try:
        if spec == "-":
            return sys.stdin.read()
        path = Path(spec)
        if not path.is_file():
            raise ScanError("PR body is not a file: %s" % redact(str(path)))
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        # The path can carry an identifier that redact() would not bound
        # (e.g. after "-" or "_"), so the diagnostic names no path at all.
        reason = exc.strerror or exc.__class__.__name__
        raise ScanError("cannot read PR body: %s" % redact(reason))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="drive_id_scan.py",
        description="Scan base..head and an optional PR body for Drive file or folder IDs. "
                    "Prints only the first four characters of each hit.",
    )
    parser.add_argument("--base", help="base revision (with --head)")
    parser.add_argument("--head", help="head revision (with --base)")
    parser.add_argument("--body", help="PR body file, or - to read stdin")
    parser.add_argument("--repo", default=".", help="git repository (default: .)")
    args = parser.parse_args(argv)
    if (args.base is None) != (args.head is None):
        parser.error("--base and --head are required together")
    if args.base is None and args.body is None:
        parser.error("provide --base and --head, --body, or both")
    try:
        hits = []
        if args.base is not None:
            hits.extend(scan_range(args.repo, args.base, args.head))
        if args.body is not None:
            hits.extend(scan_body(_read_body(args.body)))
    except ScanError as exc:
        print("drive_id_scan: %s" % exc, file=sys.stderr)
        return 2
    for line in hits:
        print(line)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
