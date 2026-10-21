# Prints a summary of a scan.py, scan2.py or filter2.py output: row count, distinct
# files per commit, and a SHA-256 over the sorted rows with the local "tree" label
# reduced to its commit (the label otherwise embeds a local path). The digest also
# depends on git's core.quotePath, which reproduce.sh pins to git's default (true).
import collections, hashlib, json, sys
rows = json.load(open(sys.argv[1]))
norm = []
files = collections.defaultdict(set)
for r in rows:
    commit = r["tree"].rsplit("@", 1)[1]
    files[commit].add(r["file"])
    norm.append(json.dumps(dict(r, tree=commit), sort_keys=True))
norm.sort()
print("rows", len(rows))
for c in sorted(files):
    print("files@" + c, len(files[c]))
print("digest", hashlib.sha256("\n".join(norm).encode()).hexdigest())
