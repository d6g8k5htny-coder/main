#!/usr/bin/env python3
"""W13 byte-discipline audit (report only, no modification).
Checks every text artifact under K3_SIDE24_LB (excluding W13_reproduction's
own working files and __pycache__) for:
  - valid UTF-8, no BOM
  - LF only (no CR)
  - no U+00A0
  - exactly one trailing LF (nonempty files)
  - no smart quotes U+2018 U+2019 U+201C U+201D
"""
import os, sys

ROOT = "/mnt/agents/output/K3_SIDE24_LB"
EXCLUDE_DIRS = {"__pycache__", "W13_reproduction"}
SMART = {"\u2018": "U+2018", "\u2019": "U+2019", "\u201C": "U+201C", "\u201D": "U+201D"}

violations = []
nfiles = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
    for fn in sorted(filenames):
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, ROOT)
        raw = open(p, "rb").read()
        if b"\x00" in raw:
            continue  # binary, skip
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            violations.append((rel, "NOT-UTF8", str(e)))
            continue
        nfiles += 1
        if raw.startswith(b"\xef\xbb\xbf"):
            violations.append((rel, "BOM", ""))
        if b"\r" in raw:
            violations.append((rel, "CR-PRESENT", "count=%d" % raw.count(b"\r")))
        if "\u00A0" in text:
            violations.append((rel, "NBSP", f"count={text.count(chr(0xA0))}"))
        if len(raw) > 0:
            if not raw.endswith(b"\n"):
                violations.append((rel, "NO-TRAILING-LF", ""))
            elif raw.endswith(b"\n\n"):
                violations.append((rel, "MULTI-TRAILING-LF", ""))
        for ch, name in SMART.items():
            if ch in text:
                violations.append((rel, "SMART-QUOTE-" + name, f"count={text.count(ch)}"))

print(f"files audited: {nfiles}")
print(f"violations: {len(violations)}")
for rel, kind, det in violations:
    print(f"VIOLATION\t{kind}\t{rel}\t{det}")
