#!/usr/bin/env python3
"""C8 independent exact-run evidence read for one ANNOUNCED current-site run.

Usage: python3 -I site_run_readback.py RUN_ID [OUTDIR]

Reads only: the one announced run, the receipts branch head and its parent chain,
and the three public files. It never lists the workflow's runs (the operator is
the sole run-list observer), dispatches, reruns or writes anything remote.
"""
import hashlib, json, subprocess, sys, time, urllib.request, os, ssl

OWNER_REPO = "d6g8k5htny-coder/main"
RECEIPTS = "codex/site-publication-receipts"
BASE = "https://d6g8k5htny-coder.github.io/main/site/"
PUBLIC = {"manifest": "snapshots/index.json", "metadata": "current-meta.json", "payload": "current.json"}


def gh(path):
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"gh api {path} failed: {out.stderr.strip()}")
    return json.loads(out.stdout)


def fetch(url):
    ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or "/root/.ccr/ca-bundle.crt")
    req = urllib.request.Request(url, headers={"User-Agent": "c8-readback"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
        body = r.read()
        hdr = {k.lower(): v for k, v in r.headers.items()}
        return r.status, body, hdr


def main():
    run_id = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else f"readback_{run_id}"
    os.makedirs(outdir, exist_ok=True)
    rep = {"read_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id": int(run_id)}

    run = gh(f"repos/{OWNER_REPO}/actions/runs/{run_id}")
    rep["run"] = {k: run.get(k) for k in ("event", "status", "conclusion", "head_sha", "head_branch",
                                         "run_attempt", "path", "created_at", "run_started_at", "updated_at")}

    head = gh(f"repos/{OWNER_REPO}/branches/{RECEIPTS}")["commit"]
    chain, sha = [], head["sha"]
    for _ in range(4):
        c = gh(f"repos/{OWNER_REPO}/git/commits/{sha}")
        chain.append({"sha": c["sha"], "message": c["message"].split("\n")[0], "parents": [p["sha"] for p in c["parents"]],
                      "tree": c["tree"]["sha"], "author": c["author"]["name"], "committer": c["committer"]["name"],
                      "date": c["committer"]["date"]})
        if not c["parents"]:
            break
        sha = c["parents"][0]["sha"]
    rep["receipt_chain"] = chain

    latest_raw = subprocess.run(["gh", "api", f"repos/{OWNER_REPO}/contents/latest.json?ref={head['sha']}",
                                 "--jq", ".content"], capture_output=True, text=True, check=True).stdout
    import base64
    latest = json.loads(base64.b64decode(latest_raw))
    rep["latest_json"] = latest
    rep["receipt_names_this_run"] = (latest.get("run", {}).get("id") == int(run_id))

    files = {}
    for key, rel in PUBLIC.items():
        status, body, hdr = fetch(BASE + rel)
        open(os.path.join(outdir, rel.replace("/", "_")), "wb").write(body)
        exp = latest.get(key, {})
        files[key] = {"url": BASE + rel, "status": status, "bytes": len(body),
                      "sha256": hashlib.sha256(body).hexdigest(),
                      "expected_bytes": exp.get("bytes"), "expected_sha256": exp.get("sha256"),
                      "match": (len(body) == exp.get("bytes") and hashlib.sha256(body).hexdigest() == exp.get("sha256")),
                      "age": hdr.get("age"), "last_modified": hdr.get("last-modified"),
                      "x_cache": hdr.get("x-cache"), "request_id": hdr.get("x-github-request-id"), "date": hdr.get("date")}
    rep["public_files"] = files
    rep["all_three_match"] = all(f["match"] for f in files.values())
    rep["read_finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(rep, open(os.path.join(outdir, "READBACK.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps({"run": rep["run"], "receipt_head": chain[0], "receipt_names_this_run": rep["receipt_names_this_run"],
                      "phase": latest.get("phase"), "source": latest.get("source"),
                      "public": {k: {x: v[x] for x in ("status", "bytes", "sha256", "match", "age")} for k, v in files.items()},
                      "all_three_match": rep["all_three_match"]}, indent=1))


if __name__ == "__main__":
    main()
