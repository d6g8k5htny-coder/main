"""Map local scratch files to exact remote blobs: repo, path, a session commit containing the blob (if any), PR."""
import json, subprocess, collections, os, re
REPOS = {"Math-": "mathf", "meta-framework": "mfrepo", "main": "/home/user/main"}
local = json.load(open("rec2/local_files.json"))
local_blobs = {r["blob"] for r in local}
blobmaps = {n: json.load(open(f"rec2/blobmap_{n}.json")) for n in REPOS}
# session commits and their PRs
def sh(d, *a):
    return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True).stdout
blob_hits = collections.defaultdict(list)   # blob -> [(repo, path, commit, pr)]
for name, d in REPOS.items():
    commits = [l.split()[0] for l in open(f"rec2/session_commits_{name}.txt") if l.strip()]
    cset = set(commits)
    # PR heads
    prs = {}
    for l in sh(d, "for-each-ref", "--format=%(objectname) %(refname)", "refs/remotes/pr/").splitlines():
        sha, ref = l.split(); prs[int(ref.rsplit("/", 1)[1])] = sha
    commit_pr = {}
    for num in sorted(prs):
        rl = sh(d, "rev-list", "--max-count=400", prs[num]).split()
        for c in rl:
            if c in cset and c not in commit_pr:
                commit_pr[c] = num
    # merged-to-default check: which session commits are ancestors of origin/main
    main_set = set(sh(d, "rev-list", "origin/main").split()) if sh(d, "rev-parse", "--verify", "-q", "origin/main").strip() else set()
    for c in commits:
        for l in sh(d, "ls-tree", "-r", c).splitlines():
            meta, path = l.split("\t", 1)
            blob = meta.split()[2]
            if blob in local_blobs:
                blob_hits[blob].append((name, path, c, commit_pr.get(c), c in main_set))
    print(name, "session commits:", len(commits), "with PR:", len(commit_pr), "on default branch:", sum(1 for c in commits if c in main_set))
# any-ref match for blobs not in session commits
out = []
stats = collections.Counter()
for r in local:
    b = r["blob"]; entry = dict(r)
    if b in blob_hits:
        hits = blob_hits[b]
        # prefer a hit on the default branch, then lowest PR
        hits = sorted(hits, key=lambda h: (not h[4], h[3] or 10**9))
        entry["match"] = {"kind": "session-commit", "repo": hits[0][0], "path": hits[0][1], "commit": hits[0][2], "pr": hits[0][3], "on_default": hits[0][4]}
        stats["session-commit"] += 1
    else:
        found = None
        for name in REPOS:
            if b in blobmaps[name]:
                found = {"kind": "repo-history", "repo": name, "path": blobmaps[name][b]}; break
        if found: entry["match"] = found; stats["repo-history"] += 1
        else: entry["match"] = None; stats["unmatched"] += 1
    out.append(entry)
json.dump(out, open("rec2/local_map.json", "w"))
print(dict(stats))
# per top-level dir summary
agg = collections.defaultdict(lambda: collections.Counter())
for e in out:
    top = e["path"].split("/")[0] if "/" in e["path"] else "(root)"
    k = e["match"]["kind"] if e["match"] else "unmatched"
    agg[top][k] += 1; agg[top][k + "_bytes"] += e["size"]
print(f"{'dir':40s} {'sess':>5s} {'hist':>5s} {'unm':>5s} {'unm_bytes':>10s}")
for top, c in sorted(agg.items(), key=lambda kv: -kv[1]["unmatched_bytes"]):
    print(f"{top:40s} {c['session-commit']:5d} {c['repo-history']:5d} {c['unmatched']:5d} {c['unmatched_bytes']:10d}")
