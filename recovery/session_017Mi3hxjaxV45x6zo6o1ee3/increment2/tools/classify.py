import json, collections, os, re, subprocess
m = json.load(open("rec2/local_map.json"))
unm = [e for e in m if not e["match"]]
# A. root classification
root = [e for e in unm if "/" not in e["path"]]
ext = collections.Counter(); extb = collections.Counter()
for e in root:
    x = os.path.splitext(e["path"])[1] or "(none)"; ext[x] += 1; extb[x] += e["size"]
print("== root unmatched by extension"); 
for x, n in ext.most_common(): print(f"  {x:10s} {n:4d} {extb[x]:10d}")
print("== 45 largest root unmatched")
for e in sorted(root, key=lambda e: -e["size"])[:45]: print(f"  {e['size']:9d} {e['path']}")
# D. basename relation to session-commit paths
sess_paths = collections.defaultdict(set)
for e in m:
    if e["match"] and e["match"]["kind"] == "session-commit":
        sess_paths[os.path.basename(e["match"]["path"])].add((e["match"]["repo"], e["match"]["path"]))
rel = 0
for e in unm:
    b = os.path.basename(e["path"])
    if b in sess_paths and b not in ("README.md","SCOPE.md","RESULTS.json","GRAPH.json","run_validation.py","__init__.py"):
        e["superseded_by_candidates"] = sorted(sess_paths[b])[:3]; rel += 1
print("== unmatched files sharing a basename with a landed session file:", rel)
# B. privacy scan of unmatched text files
EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
SAFE_EMAIL = re.compile(r'(example\.(invalid|com|org)|noreply@anthropic\.com|users\.noreply\.github\.com|@users\.noreply)')
DRIVE = re.compile(r'(privaterelay|drive\.google\.com|docs\.google\.com/(document|spreadsheets)|/folders/|googleusercontent)')
flag = {}
for e in unm:
    try:
        data = open(e["path"], "rb").read()
    except Exception as ex:
        continue
    if b"\0" in data[:4096]: e["binary"] = True; continue
    t = data.decode("utf-8", "replace")
    emails = sorted({x for x in EMAIL.findall(t) if not SAFE_EMAIL.search(x)})
    drive = sorted(set(DRIVE.findall(t)))
    if emails or drive:
        flag[e["path"]] = {"emails": [x[:2] + "…@" + x.split("@")[1] for x in emails][:5], "drive": [d if isinstance(d, str) else d[0] for d in drive][:5]}
        e["privacy_flag"] = True
print("== unmatched files with privacy hits:", len(flag))
for p, f in sorted(flag.items()): print(f"  {p}: emails={f['emails']} drive={f['drive']}")
json.dump(m, open("rec2/local_map.json", "w"))
# per-dir unmatched extension summary for non-root dirs with >= 3 unmatched
print("== non-root unmatched dirs: extension mix")
byd = collections.defaultdict(collections.Counter)
for e in unm:
    if "/" in e["path"]:
        byd[e["path"].split("/")[0]][os.path.splitext(e["path"])[1] or "(none)"] += 1
for d, c in sorted(byd.items()):
    print(f"  {d:36s} " + " ".join(f"{k}:{v}" for k, v in c.most_common(6)))
