#!/usr/bin/env python3
"""Print a line-numbered digest of each proof document, for the manual census reading.

For each document listed in extracted.json, it prints:
  - the self-description paragraph that follows the title;
  - every line that states a theorem, proposition, lemma or corollary;
  - every line that states a non-claim or limit;
  - the verdict lines of an adjacent REVIEW.md, if one exists.

Line numbers refer to the blob at the extracted revision, so a census entry can cite
exact ranges. Usage: digest.py --git DIR --extracted FILE [--only SUBSTRING]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

STATEMENT = re.compile(r"\*\*(?:Main )?(?:Theorem|Proposition|Lemma|Corollary)\b|^#{1,6}\s.*\b(?:Theorem|Proposition)\b"
                       r"|^(?:Theorem|Proposition|Corollary)\s+[\w.()-]+[.:]", re.I)
NONCLAIM = re.compile(r"\bdoes not (?:claim|prove|establish|certify)|\bnot claimed\b|\bno (?:claim|lean|formal|numerical"
                      r"|certified)\b|\bnon-claims?\b|\bexclu(?:de|sion)|\bremains? (?:open|conditional)\b"
                      r"|\bnot (?:a|an) (?:scientific|formal|lean|certified|proof)", re.I)
VERDICT = re.compile(r"\b(?:PASS|ACCEPT|AMEND|REJECT|HOLD|FAIL|BLOCK)(?:_[A-Z_]+)?\b|\bverdict\b", re.I)


def show(repo, rev, path):
    proc = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return proc.stdout if proc.returncode == 0 else None


def first_paragraph(lines):
    out, started = [], False
    for i, line in enumerate(lines, 1):
        if not started:
            if line.startswith("#"):
                started = True
            continue
        if not line.strip():
            if out:
                break
            continue
        out.append((i, line))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--git", type=Path, required=True)
    p.add_argument("--extracted", type=Path, required=True)
    p.add_argument("--only")
    args = p.parse_args(argv)
    data = json.loads(args.extracted.read_text())
    rev = data["rev"]
    for doc in data["documents"]:
        if args.only and args.only not in doc["path"]:
            continue
        lines = show(args.git, rev, doc["path"]).splitlines()
        print(f"=================== {doc['path']}  blob {doc['blob'][:12]}  {len(lines)} lines")
        print(f"TITLE: {doc['title']}")
        for i, l in first_paragraph(lines):
            print(f"  P{i:4}: {l[:300]}")
        for i, l in enumerate(lines, 1):
            if STATEMENT.search(l):
                print(f"  S{i:4}: {l[:300]}")
        for i, l in enumerate(lines, 1):
            if NONCLAIM.search(l):
                print(f"  N{i:4}: {l[:300]}")
        review_path = doc["path"].rsplit("/", 1)[0] + "/REVIEW.md"
        review = show(args.git, rev, review_path)
        if review is not None:
            hits = [(i, l) for i, l in enumerate(review.splitlines(), 1) if VERDICT.search(l)]
            print(f"  REVIEW.md present, {len(review.splitlines())} lines; verdict lines:")
            for i, l in hits[:8]:
                print(f"  R{i:4}: {l[:240]}")
        else:
            print("  REVIEW.md: none adjacent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
