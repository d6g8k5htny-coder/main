"""Amendment 1 builder for RECOVERY_MAP.tsv (session_017Mi3hxjaxV45x6zo6o1ee3), standard library + git.

This is the script that produced RECOVERY_MAP_AMENDMENT_1.tsv, AMENDMENTS.md and SHA256SUMS2_AMENDMENT_1.txt
at 2026-10-13T02:22:30Z (commit 1926b02e), run from the increment2 directory with history_rows.json extracted
from RECOVERY_MAP.tsv (rows whose disposition is MATCHED_REPO_HISTORY). Precedence: default branches first,
then the coordinator's order (main#307 comment 6136788138); `also_in_trees` keeps the other memberships.
"""
import collections
import hashlib
import json
import subprocess
import sys
import time

S = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scratch"          # scratch root holding clones mfrepo/ and mathf/
MAIN = sys.argv[2] if len(sys.argv) > 2 else "/home/user/main"   # main clone with the peer recovery branches fetched
rows = json.load(open(S + "/rec2/history_rows.json"))
TREES = [
    ("meta-framework", "daa4be2649f027d5c7b80ddbd40324b040169ee1", S + "/mfrepo", "meta-framework default branch (main) at 13 October"),
    ("meta-framework", "ede7eb11da68d9ccd84333105de6bd0d5a91579f", S + "/mfrepo", "meta-framework branch claude/meta15-contract-compatibility-20261012 (PR #27 history)"),
    ("Math-", "8acc48566de3ae613c1ae320903b66c1d5411797", S + "/mathf", "Math- default branch (main) at 13 October"),
    ("main", "b7deb198722582df191006e4f47238f92e6e543d", MAIN, "main default branch at the recovery branch base"),
    ("main", "037436b454003593af8039907a68490331300f9e", MAIN, "main branch claude/recovery-01cz-20261013 (session 01Cz recovery copy)"),
    ("main", "43435b361202d54b18f7cb25318b6c8234217429", MAIN, "main branch claude/recovery-S-20261013 (session 01S recovery delta)"),
    ("main", "3f121217227d159b687f215a3942e1ace7f3fa7e", MAIN, "main branch claude/nice-maxwell-7i7ok0 (this session's increment 2 head)"),
]
EXPECTED = {"daa4be26": 7529, "ede7eb11": 7, "8acc4856": 6, "037436b4": 47, "43435b36": 46, "b7deb198": 14, "3f121217": 3}
trees = {}
for repo, commit, clone, _ in TREES:
    out = subprocess.run(["git", "-C", clone, "ls-tree", "-r", commit], capture_output=True, text=True, check=True).stdout
    trees[(repo, commit)] = {l.split("\t", 1)[1]: l.split("\t", 1)[0].split()[2] for l in out.splitlines()}
GH = "https://github.com/d6g8k5htny-coder/"
amend = []; per_tree = collections.Counter(); unresolved = []
for r in rows:
    hits = [commit for repo, commit, _, _ in TREES if repo == r["repo"] and trees[(repo, commit)].get(r["remote_path"]) == r["blob"]]
    if not hits:
        unresolved.append(r); continue
    per_tree[hits[0][:8]] += 1
    amend.append({**r, "commit": hits[0], "locator": f"{GH}{r['repo']}/blob/{hits[0]}/{r['remote_path']}", "also_in": ",".join(hits[1:])})
print("resolved", len(amend), "unresolved", len(unresolved), dict(per_tree))
assert len(unresolved) == 0 and dict(per_tree) == EXPECTED, "counts differ from the coordinator's"
NOW = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
cols = ["path", "git_blob", "sha256", "repo", "remote_path", "commit", "locator", "also_in_trees"]
with open("RECOVERY_MAP_AMENDMENT_1.tsv", "w") as f:
    f.write(f"# Amendment 1 to RECOVERY_MAP.tsv (session_017Mi3hxjaxV45x6zo6o1ee3), generated {NOW}: exact commit locators for every MATCHED_REPO_HISTORY row, per main#307 comment 6136788138\n")
    f.write("\t".join(cols) + "\n")
    for a in sorted(amend, key=lambda a: a["path"]):
        f.write("\t".join([a["path"], a["blob"], a["sha256"], a["repo"], a["remote_path"], a["commit"], a["locator"], a["also_in"]]) + "\n")
src_sha = hashlib.sha256(open("RECOVERY_MAP.tsv", "rb").read()).hexdigest()
lines = ["# Amendments to the increment-2 crosswalk (session_017Mi3hxjaxV45x6zo6o1ee3)\n",
 "## Amendment 1 — exact commit locators for the MATCHED_REPO_HISTORY rows\n",
 f"Requested by the coordinator in main#307 comment 6136788138 (13 October 2026, 02:20Z) after a read-only reconciliation of the crosswalk published at commit `3f121217227d159b687f215a3942e1ace7f3fa7e`. Generated {NOW} by Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`. Organizational-independence credit 0. Scientific effect NONE. No payload upload, test replay or derivation; this validates immutable locations for the history rows only, not scientific correctness and not the 7,055 MATCHED_SESSION_COMMIT rows.\n",
 f"- **Source preserved:** `RECOVERY_MAP.tsv` is unchanged (SHA-256 `{src_sha}`, as in `SHA256SUMS2.txt`). The amendment is the separate file `RECOVERY_MAP_AMENDMENT_1.tsv`; `SHA256SUMS2_AMENDMENT_1.txt` lists the amendment files.",
 f"- **Method:** for each of the {len(rows)} MATCHED_REPO_HISTORY rows ({len({r['blob'] for r in rows})} unique blobs) the tuple (repo, remote_path, git_blob) was matched against the seven immutable trees named by the coordinator, read with `git ls-tree -r <commit>` in local clones. A row's primary locator is the first tree holding the tuple in the order below (default branches first, then the coordinator's order); `also_in_trees` lists every other tree of the seven that holds it, so no precedence choice hides a location. Unresolved: {len(unresolved)}.",
 "- **Trees and primary counts** (equal to the coordinator's counts in 6136788138):\n", "| repo | commit | what it is | rows (primary) |", "|---|---|---|---:|"]
for repo, commit, _, what in TREES:
    lines.append(f"| {repo} | `{commit}` | {what} | {per_tree[commit[:8]]} |")
lines.append("\n- **Interpretation:** the 7,529 meta-framework rows at `daa4be26` are the local copy of the meta-framework tree used by the meta-framework#15 parity probe (`mf15/mathman/`), not uploaded because every byte is on that default branch; the 14 `b7deb198` rows are files identical to main's default-branch content; the `037436b4` and `43435b36` rows are files byte-identical to files in the two peer recovery copies; the 3 `3f121217` rows are files identical to content first published in this session's own increment.\n")
open("AMENDMENTS.md", "w").write("\n".join(lines))
with open("SHA256SUMS2_AMENDMENT_1.txt", "w") as f:
    for name in ("AMENDMENTS.md", "RECOVERY_MAP_AMENDMENT_1.tsv"):
        f.write(hashlib.sha256(open(name, "rb").read()).hexdigest() + "  ./" + name + "\n")
print("written; source map sha", src_sha[:12])
