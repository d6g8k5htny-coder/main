#!/usr/bin/env python3
"""Build final MANIFEST.sha256 for the whole K3_SIDE24_LB tree.
Every deliverable file: whole-file sha256 + byte count.
Excludes: __pycache__ content, the manifest itself.
Format per line: <sha256>  <relpath>  (<bytes> bytes)   -- plus a
sha256sum-compatible header note. A companion MANIFEST.check.sha256 in
strict `sha256sum -c` format (hash + two spaces + path) is also written.
"""
import os, hashlib

ROOT = "/mnt/agents/output/K3_SIDE24_LB"
OUT = os.path.join(ROOT, "MANIFEST.sha256")
OUTC = os.path.join(ROOT, "W13_reproduction", "MANIFEST.check.sha256")

entries = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in sorted(filenames):
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, ROOT)
        if rel == "MANIFEST.sha256":
            continue
        raw = open(p, "rb").read()
        h = hashlib.sha256(raw).hexdigest()
        entries.append((rel, h, len(raw)))

entries.sort()
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write("# MANIFEST.sha256 -- K3_SIDE24_LB full deliverable tree\n")
    f.write("# built by W13_reproduction/build_manifest.py; whole-file sha256 + byte count\n")
    f.write("# excludes: MANIFEST.sha256 itself, __pycache__ contents\n")
    for rel, h, n in entries:
        f.write("%s  %s  (%d bytes)\n" % (h, rel, n))
with open(OUTC, "w", encoding="utf-8", newline="\n") as f:
    for rel, h, n in entries:
        f.write("%s  %s\n" % (h, rel))
print("entries: %d" % len(entries))
print("total bytes: %d" % sum(n for _, _, n in entries))
print("manifest: %s" % OUT)
