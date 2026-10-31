#!/usr/bin/env python3
"""Count the failing subtests per test in unittest -v output (RED-control table).

Usage: red_counts.py LOG...   (prints test name, failures, and the labels seen)
"""
import collections
import re
import sys

for path in sys.argv[1:]:
    counts = collections.Counter()
    labels = collections.defaultdict(set)
    for line in open(path, errors='replace'):
        m = re.match(r'^(FAIL|ERROR): (test_\w+) \([^)]*\)(?: \[(.*)\])?', line)
        if m:
            counts[m.group(2)] += 1
            if m.group(3):
                labels[m.group(2)].add(m.group(3)[:60])
    total = sum(counts.values())
    print(path, 'failing subtests:', total)
    for name, n in sorted(counts.items()):
        print('  %-60s %3d' % (name, n))
