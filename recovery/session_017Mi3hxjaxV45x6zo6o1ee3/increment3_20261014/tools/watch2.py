"""Event watcher for session 017: repo-wide comment scan with seen-ID dedup.

A failed or unparsable call never advances the anchor, so a transient API
failure cannot drop an event (the defect in the earlier per-issue loop).
Exits 0 after printing the first batch of relevant ("hot") events; routine
events and call failures go to the log.

usage: python3 -I watch2.py STATE LOG ANCHOR [ITER] [SLEEP]
"""
import json
import re
import subprocess
import sys
import time

STATE, LOG, ANCHOR = sys.argv[1], sys.argv[2], sys.argv[3]
ITER = int(sys.argv[4]) if len(sys.argv) > 4 else 48
SLEEP = int(sys.argv[5]) if len(sys.argv) > 5 else 120

OWNER = "repos/d6g8k5htny-coder"
REPOS = ("main", "Math-", "meta-framework")
COMMIT = "946ffb99e3696bdaac469883a4d61d047026a4b3"
MINE = ("Anthropic Claude, session `session_017Mi", "Session `session_017Mi")
KEYS = ("017Mi", "Claude017", "Claude 017", "(017", "017,", "017 ", "017's", "017:")
HUB_388 = re.compile(r"#388\b|\b388\b|atlas")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def gh(path):
    try:
        p = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=90)
    except Exception as e:  # noqa: BLE001 - any failure is just logged
        return None, "exec: %s" % e
    if p.returncode != 0:
        return None, "rc=%d %s" % (p.returncode, (p.stderr or p.stdout).strip()[:160])
    try:
        return json.loads(p.stdout), None
    except Exception as e:  # noqa: BLE001
        return None, "json: %s" % e


def load():
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {"seen": [], "anchor": {r: ANCHOR for r in REPOS}, "prs": []}


def save(st):
    with open(STATE, "w") as f:
        json.dump(st, f)


def log(lines):
    with open(LOG, "a") as f:
        for line in lines:
            f.write(line + "\n")


def classify(repo, num, body):
    if repo == "main" and num in ("388", "399"):
        return True
    if any(k in body for k in KEYS):
        return True
    head = body.lstrip().split("\n", 1)[0][:160]
    if repo == "main" and num == "307" and head.startswith("dot") and HUB_388.search(head):
        return True
    return False


def scan(st):
    hot, routine, fails = [], [], []
    seen = set(st["seen"])
    for repo in REPOS:
        start = now()
        anchor = st["anchor"].get(repo, ANCHOR)
        page, ok, batch = 1, True, []
        while True:
            d, err = gh("%s/%s/issues/comments?since=%s&per_page=100&page=%d" % (OWNER, repo, anchor, page))
            if err or not isinstance(d, list):
                fails.append("%s FAIL %s comments p%d: %s" % (now(), repo, page, err or type(d).__name__))
                ok = False
                break
            batch += d
            if len(d) < 100:
                break
            page += 1
        for c in batch:
            cid = str(c.get("id"))
            if cid in seen or c.get("created_at", "") < ANCHOR:
                continue
            seen.add(cid)
            body = c.get("body") or ""
            if any(m in body for m in MINE):
                continue
            num = (c.get("issue_url") or "").rsplit("/", 1)[-1]
            line = "%s#%s %s %s | %s" % (repo, num, cid, c.get("created_at"), body[:280].replace("\n", " "))
            (hot if classify(repo, num, body) else routine).append(line)
        if ok:
            # keep a 10-minute overlap; seen-ID dedup makes the overlap harmless
            t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 600))
            st["anchor"][repo] = max(anchor, min(t, start))
    for prepo in ("main", "meta-framework"):
        d, err = gh("%s/%s/pulls?state=open&sort=created&direction=desc&per_page=10" % (OWNER, prepo))
        if err or not isinstance(d, list):
            fails.append("%s FAIL pulls %s: %s" % (now(), prepo, err))
            continue
        for p in d:
            key = str(p.get("number")) if prepo == "main" else prepo + "#" + str(p.get("number"))
            if p.get("created_at", "") >= ANCHOR and key not in st["prs"]:
                st["prs"].append(key)
                hot.append("NEW PR %s#%s %s %s | %s" % (prepo, p.get("number"), p.get("created_at"), p["head"]["ref"], p.get("title", "")[:120]))
    d, err = gh("%s/main/commits/%s" % (OWNER, COMMIT))
    if isinstance(d, dict) and d.get("sha") == COMMIT:
        hot.append("COMMIT 946ffb99 NOW ON GITHUB")
    elif err and "No commit found" not in err:
        fails.append("%s FAIL commit check: %s" % (now(), err))
    st["seen"] = sorted(seen)[-2000:]
    return hot, routine, fails


def main():
    st = load()
    nfail = 0
    for _ in range(ITER):
        hot, routine, fails = scan(st)
        save(st)
        nfail += len(fails)
        log(["ROUTINE " + x for x in routine] + fails)
        if hot:
            print("\n".join(hot))
            print("(detected %s; call failures this run: %d)" % (now(), nfail))
            return 0
        time.sleep(SLEEP)
    print("no relevant events through %s; call failures this run: %d" % (now(), nfail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
