#!/usr/bin/env python3
"""Extract proof-obligation evidence from one Git revision of main.

Reads only Git objects at the given revision, with no checkout and no network,
and writes extracted.json. For each proof document it records:
  - path, blob and bytes;
  - its first heading and any line that states its own status;
  - the headings of sections that state theorems, scope or limits, with those sections' text;
  - links to reviews (PR reviews, issue comments and reviews/ paths) and to other proof documents.
For each Lean target in formal/manifest.json it records the declaration and the
manifest's own formalization and alignment fields.

This is extraction, not judgement: CENSUS.md holds the reading of these
excerpts. Usage: extract.py --git DIR --rev REV --out DIR
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROOF_NAME = re.compile(r"(^|/)(PROOF|THEOREM|LEMMA)[^/]*\.md$", re.I)
STATUS_LINE = re.compile(
    r"^\s*(?:\*\*|_)?(?:status|scope|disposition|verdict|claim status)\b.*$"
    r"|research preparation|conventional analytic source|not (?:yet )?reviewed"
    r"|\b(?:ACCEPT|PASS|AMEND|REJECT|HOLD)(?:_[A-Z]+)*\b", re.I | re.M)
KEY_HEADING = re.compile(
    r"theorem|proposition|lemma|main result|statement|claim|scope|limit|does not|not claim"
    r"|exclusion|non-claim|boundary|open|status|assumption|hypothes|conclusion|summary", re.I)
REVIEW_LINK = re.compile(
    r"https://github\.com/[\w.-]+/[\w.-]+/(?:pull|issues)/\d+#(?:pullrequestreview|issuecomment)-\d+"
    r"|(?<![\w/])reviews/[\w./-]+")
PROOF_LINK = re.compile(r"(?<![\w])(?:[\w.-]+/)*(?:PROOF|THEOREM|LEMMA)[\w.-]*\.md")
SECTION_CHARS = 2400


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, text=True).stdout


def sections(text):
    """Split Markdown into (heading, body) pairs at ATX headings."""
    out, heading, buf = [], None, []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if heading is not None or buf:
                out.append((heading, "\n".join(buf).strip()))
            heading, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    out.append((heading, "\n".join(buf).strip()))
    return out


def document_evidence(repo, rev, path):
    raw = git(repo, "show", f"{rev}:{path}")
    secs = sections(raw)
    first = next((h for h, _ in secs if h), None)
    status = sorted({m.group(0).strip()[:200] for m in STATUS_LINE.finditer(raw)})
    key = [{"heading": h, "text": b[:SECTION_CHARS], "truncated": len(b) > SECTION_CHARS}
           for h, b in secs if h and KEY_HEADING.search(h)]
    return {
        "path": path,
        "blob": git(repo, "rev-parse", f"{rev}:{path}").strip(),
        "bytes": len(raw.encode()),
        "title": first,
        "status_lines": status[:40],
        "key_sections": key,
        "review_links": sorted(set(REVIEW_LINK.findall(raw))),
        "proof_links": sorted({p for p in PROOF_LINK.findall(raw) if not path.endswith(p)}),
    }


def lean_targets(repo, rev):
    manifest = json.loads(git(repo, "show", f"{rev}:formal/manifest.json"))
    rows = [{k: t.get(k) for k in ("name", "module", "title", "source", "informal_anchor", "does_not_claim")
             if k in t} for t in manifest.get("targets", [])]
    return {"package": manifest.get("package"), "count": len(rows), "targets": rows,
            "formalization_status": manifest.get("formalization_status"),
            "alignment_status": manifest.get("alignment_status"),
            "allowed_axioms": manifest.get("allowed_axioms"),
            "lean_toolchain": manifest.get("lean_toolchain"),
            "scientific_status_authority": manifest.get("scientific_status_authority")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--git", type=Path, required=True)
    p.add_argument("--rev", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rev = git(args.git, "rev-parse", "--verify", args.rev + "^{commit}").strip()
    paths = [l for l in git(args.git, "ls-tree", "-r", "--name-only", rev).splitlines()
             if PROOF_NAME.search(l)]
    data = {"rev": rev, "documents": [document_evidence(args.git, rev, p) for p in paths],
            "lean": lean_targets(args.git, rev)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "extracted.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"{len(paths)} documents, {data['lean']['count']} Lean targets at {rev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
