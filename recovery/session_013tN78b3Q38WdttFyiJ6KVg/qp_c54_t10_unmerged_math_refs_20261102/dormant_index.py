#!/usr/bin/env python3
"""Index the PR-level dormant quarantine ids (Q-MATH-<n>..., Q-MAIN-<n>...) named in main issue and PR comments, with the Math- PR
comments that hold the records where a comment links or cites one (a PR cited only by "Math#N comment `ID`" gets
ids []). Input: a JSON list of GitHub issue-comment objects (the REST
/repos/d6g8k5htny-coder/main/issues/comments listing, saved as fetched). Usage: python3 -I -B dormant_index.py COMMENTS_JSON OUT_JSON"""
import collections, json, re, sys
comments, out = sys.argv[1:]
allc = json.load(open(comments))
ids, links = collections.defaultdict(set), collections.defaultdict(set)
for c in allc:
    b = c['body']
    for m in re.finditer(r'(Q-MATH-(\d+)[A-Z0-9-]*)', b):
        ids[('Math-', int(m.group(2)))].add(m.group(1))
    for m in re.finditer(r'(Q-MAIN-(\d+)[A-Z0-9-]*)', b):
        ids[('main', int(m.group(2)))].add(m.group(1))
    for m in re.finditer(r'github\.com/d6g8k5htny-coder/Math-/pull/(\d+)#issuecomment-(\d+)', b):
        links[('Math-', int(m.group(1)))].add(int(m.group(2)))
    for m in re.finditer(r'Math-?\s*#(\d+)\s+comment\s+`?#?(\d{9,})`?', b):  # short form "Math#557 comment `6654065083`"
        links[('Math-', int(m.group(1)))].add(int(m.group(2)))
keys = sorted(set(ids) | set(links))
res = {f'{k[0]}#{k[1]}': {'ids': sorted(ids.get(k, [])), 'record_comments': sorted(links.get(k, []))} for k in keys}
with open(out, 'x') as f:
    json.dump(res, f, indent=1)
print(len([k for k in res if k.startswith('Math-')]), 'Math- PRs;', len([k for k in res if k.startswith('main')]), 'main PRs')
