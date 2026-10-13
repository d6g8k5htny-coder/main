"""Build blob-id -> (repo, first path) map from every ref of a clone (stdlib + git)."""
import subprocess, sys, json
repo_name, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
p = subprocess.run(["git", "-C", path, "rev-list", "--all", "--objects", "--missing=allow-promisor"],
                   capture_output=True, text=True)
if p.returncode != 0:
    print("rev-list failed:", p.stderr[:500]); sys.exit(1)
m = {}
for line in p.stdout.splitlines():
    parts = line.split(" ", 1)
    if len(parts) == 2 and parts[1] and not parts[1].endswith("/"):
        m.setdefault(parts[0], parts[1])
json.dump(m, open(out, "w"))
print(repo_name, "objects-with-path:", len(m))
