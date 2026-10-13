import json, re, subprocess, sys
P = "<SCRATCH>/pr282/reviews/"
for r in sys.argv[1:]:
    rv = json.loads(subprocess.run(["gh", "api", f"repos/d6g8k5htny-coder/main/pulls/282/reviews/{r}"], capture_output=True, text=True).stdout)
    b = rv.get("body") or ""
    open(P + f"{rv['id']}.md", "w").write(f"{rv['id']} {rv['state']} {rv['submitted_at']} commit {rv['commit_id']}\n\n" + b)
    print("#####", rv["id"], rv["state"], rv["submitted_at"], rv["commit_id"][:8], "bytes", len(b))
    for ln in b.split("\n"):
        l = ln.strip()
        if re.search(r"(PASS|AMEND|RESOLV|resolv|[Vv]erdict|4209\d{6}|421\d{7}|ACCESS-HANDOFF|C162|C165|C189|residual)", l):
            print("   ", l[:300])
