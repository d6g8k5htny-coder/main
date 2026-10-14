"""List comments mentioning session 017 since a time, per repo, and whether 017 posted later in the same thread."""
import json, subprocess, sys
SINCE = sys.argv[1]
REPOS = sys.argv[2:]
KEYS = ("017Mi", "Claude017", "Claude 017", "(017", "017,", "017 ", "017's", "017:", "017.")
MINE = ("Anthropic Claude, session `session_017Mi", "Session `session_017Mi")
def gh_all(path):
    out, page = [], 1
    while True:
        p = subprocess.run(["gh", "api", "%s%spage=%d" % (path, "&" if "?" in path else "?", page)], capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            print("FAIL", path, page, (p.stderr or p.stdout)[:150]); return out
        d = json.loads(p.stdout)
        if not isinstance(d, list):
            print("NONLIST", path, str(d)[:150]); return out
        out += d
        if len(d) < 100: return out
        page += 1
for r in REPOS:
    base = "repos/d6g8k5htny-coder/%s" % r
    ic = gh_all("%s/issues/comments?since=%s&per_page=100" % (base, SINCE))
    rc = gh_all("%s/pulls/comments?since=%s&per_page=100" % (base, SINCE))
    items = []
    for c in ic:
        items.append(("issue", c["issue_url"].rsplit("/", 1)[1], c))
    for c in rc:
        items.append(("thread", c["pull_request_url"].rsplit("/", 1)[1], c))
    my_last = {}
    for kind, num, c in items:
        b = c.get("body") or ""
        if any(m in b[:600] for m in MINE):
            my_last[num] = max(my_last.get(num, ""), c["created_at"])
    print("== %s: %d issue comments, %d review-thread comments since %s; 017 posts in threads: %s" % (r, len(ic), len(rc), SINCE, {k: v[11:19] for k, v in sorted(my_last.items())}))
    for kind, num, c in sorted(items, key=lambda x: x[2]["created_at"]):
        b = c.get("body") or ""
        if any(m in b[:600] for m in MINE):
            continue
        if not any(k in b for k in KEYS) and not (kind == "thread" and c.get("in_reply_to_id")):
            continue
        later = my_last.get(num, "") > c["created_at"]
        tag = "answered-later" if later else "NO-017-POST-AFTER"
        print("  %-17s %s#%s %s %s %s | %s" % (tag, r, num, kind, c["id"], c["created_at"][11:19], b[:170].replace("\n", " ")))
