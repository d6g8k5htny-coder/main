"""RETRO source resolver (read-only). For every ledger unit with has_source=false,
find in-unit mathematical text and resolve every declared {path, git_blob} reference
against the named repository's main, then against that path's history.
Credit 0 · sci NONE. Inventory only; resolves identities, not correctness.
REPOS below are local clone paths of the run; edit them to replay elsewhere."""
import json, re, subprocess, sys, collections
REPOS = {"Math-": "/home/user/Math-", "main": "/home/user/main"}
def git(repo, *a):
    return subprocess.run(["git", "-C", REPOS[repo], *a], capture_output=True, text=True).stdout
hg = json.loads(git("main", "show", "origin/main:data/hypergraph/math_hypergraph.json"))
MAIN = {r: git(r, "rev-parse", "origin/main").strip() for r in REPOS}
hist_cache = {}
def blob_at(repo, rev, path):
    out = git(repo, "ls-tree", rev, "--", path).split()
    return out[2] if len(out) >= 3 else None
def blob_in_history(repo, path, blob):
    key = (repo, path)
    if key not in hist_cache:
        revs = git(repo, "log", "--all", "--format=%H", "--", path).split()[:400]
        hist_cache[key] = {}
        for c in revs:
            b = blob_at(repo, c, path)
            if b: hist_cache[key].setdefault(b, c)
    return hist_cache[key].get(blob)
def refs(obj, base):
    if isinstance(obj, dict):
        p, b = obj.get("path"), obj.get("git_blob") or obj.get("blob")
        if isinstance(p, str) and isinstance(b, str) and re.fullmatch(r"[0-9a-f]{40}", b):
            yield p, b, obj.get("repository") or obj.get("repo")
        for v in obj.values(): yield from refs(v, base)
    elif isinstance(obj, list):
        for v in obj: yield from refs(v, base)
rows = []
for e in hg["hyperedges"]:
    if e.get("has_source"): continue
    units = [k for k, v in e["roles"].items() if v == "unit"]
    for u in units:
        repo, _, path = u.split(":", 2)
        files = git(repo, "ls-tree", "-r", "--name-only", "origin/main", path + "/").split()
        if not files:
            b = blob_at(repo, "origin/main", path)
            rows.append({"row": e["row"], "unit": u, "state": "SINGLE_FILE_AT_MAIN" if b else "UNIT_ABSENT_AT_MAIN", "blob": b,
                         "math_text": [path] if path.endswith((".md", ".tex", ".lean")) else []}); continue
        md = [f[len(path)+1:] for f in files if f.endswith((".md", ".tex")) and not f.endswith("README.md")]
        res = collections.Counter(); bad = []
        for f in files:
            if not f.endswith(".json"): continue
            try: j = json.loads(git(repo, "show", f"origin/main:{f}"))
            except Exception: continue
            for p, b, rr in refs(j, path):
                r2 = "Math-" if (rr or "").endswith("Math-") else ("main" if (rr or "").endswith("/main") or rr == "main" else repo)
                cand = [p, f"{path}/{p}"]
                st = None
                for c in cand:
                    if blob_at(r2, "origin/main", c) == b: st = "AT_MAIN"; break
                if not st:
                    for c in cand:
                        h = blob_in_history(r2, c, b)
                        if h: st = "IN_HISTORY"; break
                if not st:
                    st = "UNRESOLVED"; bad.append(f"{r2}:{p}@{b[:8]}")
                res[st] += 1
        rows.append({"row": e["row"], "unit": u, "math_text": md, "refs": dict(res), "unresolved": bad[:10]})
json.dump({"main_cuts": MAIN, "rows": rows}, sys.stdout, indent=1)
