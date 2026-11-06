#!/usr/bin/env python3
"""Count the references tools/cbz/reference_crawler.py extracts, by kind, source and target.

Usage, from the repository root:
  python3 -B -S cbz/evidence/reference-crawler-01shcrxv/reference_stats.py WORKSPACE REV CACHE
It crawls exactly as the tool does, and prints three count maps.
"""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "cbz"))
import reference_crawler as rc  # noqa: E402


def main(workspace, rev, cache):
    nodes, refs, findings = rc.crawl(workspace, rev, cache, skipped={})
    kinds = collections.Counter()
    for r in refs:
        kinds["url" if r.kind == "url" else "json"] += 1
        if r.kind == "json":
            kinds["json_with_commit" if r.commit else "json_live"] += 1
            kinds["blob_claims"] += len(r.blobs)
            kinds["sha256_claims"] += len(r.sha256s)
            kinds["byte_count_claims"] += len(r.sizes)
            kinds["excerpts"] += bool(r.lines)
    print(dict(kinds))
    print("by source:", dict(collections.Counter(r.source for r in refs)))
    print("by target:", dict(collections.Counter(rc.name_of(r.repo) for r in refs)))


if __name__ == "__main__":
    main(*sys.argv[1:4])
